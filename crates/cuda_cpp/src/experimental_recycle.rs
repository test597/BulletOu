//! Explicit, scratch-only, bpu=1 diagnostic experiment. Not a normal training option.
//! Selection uses the current TRAINING batch only, never validation.
use super::*;

fn replace_group(ctx: &Context, buffer: &F32Buffer, state: &RangerParamState,
    values: &[f32], changes: &[(usize, f32)]) -> Result<()> {
    let mut s = state.download(ctx)?;
    for &(i, slow_value) in changes {
        s.momentum[i] = 0.0;
        s.velocity[i] = 0.0;
        s.slow_params[i] = slow_value;
    }
    buffer.upload(ctx, values)?;
    state.upload(ctx, values.len(), RangerParamHostState {
        momentum: &s.momentum, velocity: &s.velocity, slow_params: &s.slow_params,
    })
}

// Algebraic operation: two constant branch contributions become L2 bias.
fn fold(w: &mut [f32], b: &mut [f32], rows: usize, cols: usize, u: usize, h: usize, linear: f32) {
    for j in 0..rows {
        b[j] += w[j*cols+u] + linear*w[j*cols+h+u];
        w[j*cols+u] = 0.0;
        w[j*cols+h+u] = 0.0;
    }
}

pub(super) fn maybe_recycle(r: &SfnnTrainStepRunner, ctx: &Context,
    batch: &SfnnForwardDeviceBatch, entry_weights: &F32Buffer, step: u64,
    update_weights: bool) -> Result<()> {
    let raw = match std::env::var("BULLETOU_EXPERIMENT_RECYCLE_STEPS") {
        Err(std::env::VarError::NotPresent) => return Ok(()),
        Err(_) => return Err(CudaCppError::message("invalid recycle environment")),
        Ok(s) => s,
    };
    let steps: Vec<u64> = raw.split(',').map(str::parse).collect::<std::result::Result<_,_>>()
        .map_err(|_| CudaCppError::message("recycle steps must be comma separated positive update numbers"))?;
    if steps.iter().any(|&s|s==0) { return Err(CudaCppError::message("zero recycle step")); }
    if !steps.contains(&step) { return Ok(()); }
    recycle_now(r,ctx,batch,entry_weights,step,update_weights)
}

fn recycle_now(r: &SfnnTrainStepRunner, ctx: &Context, batch: &SfnnForwardDeviceBatch,
    entry_weights: &F32Buffer, step: u64, update_weights: bool) -> Result<()> {
    if !update_weights || r.shape.has_compact_l1() || r.factorizer.any_axis() || r.residual_count_gates_enabled {
        return Err(CudaCppError::message("recycle experiment requires bpu1, dense L1, no axes/count gates"));
    }
    for buffer in [&r.weights.l2fw,&r.weights.l2fb,&r.weights.l3fw,&r.weights.l3fb].into_iter().flatten() {
        if buffer.download(ctx)?.iter().any(|&v|v!=0.0) {
            return Err(CudaCppError::message("recycle experiment does not support L2/L3 factorizers"));
        }
    }
    let (n,h,f,l1out,l2out)=(r.batch_size,r.shape.l1_hidden,r.shape.ft_size,r.shape.l1_out(),r.shape.l2_size);
    let cols=2*h;
    let buckets=batch.buckets.download(ctx)?;
    let ew=entry_weights.download(ctx)?;
    let a=r.forward_workspace.l2_input.download(ctx)?;
    let mut chosen=Vec::new();
    for k in 0..r.shape.num_stacks {
        let samples:Vec<usize>=(0..n).filter(|&i|buckets[i]==k as i32 && ew[i]>0.0).collect();
        if samples.len()<512 { continue; }
        for u in 0..h {
            if !samples.iter().all(|&i|a[i*cols+u]==1.0) { continue; }
            let linear=a[samples[0]*cols+h+u];
            if (linear==0.0 || linear==1.0) && samples.iter().all(|&i|a[i*cols+h+u]==linear) {
                chosen.push((k,u,linear,samples.clone()));
            }
        }
    }
    if chosen.is_empty() {
        eprintln!("  EXPERIMENT recycle step={step}: no fully constant unit with >=512 active training positions");
        return Ok(());
    }
    let old_output=r.forward_workspace.output.download(ctx)?;
    let x=r.forward_workspace.combined.download(ctx)?;
    let mut w1=r.weights.l1w.download(ctx)?; let mut b1=r.weights.l1b.download(ctx)?;
    let mut w2=r.weights.l2w.download(ctx)?; let mut b2=r.weights.l2b.download(ctx)?;
    let (sw,sb,sloww,slowb)=if r.factorizer.shared {
        (r.weights.l1fw.as_ref().unwrap().download(ctx)?,r.weights.l1fb.as_ref().unwrap().download(ctx)?,
         r.optimizer_states.l1fw.as_ref().unwrap().slow_params.download(ctx)?,
         r.optimizer_states.l1fb.as_ref().unwrap().slow_params.download(ctx)?)
    } else { (vec![0.0;l1out*f],vec![0.0;l1out],vec![0.0;l1out*f],vec![0.0;l1out]) };
    let alpha=r.factorizer_alpha.shared;
    let (mut c1,mut cb1,mut c2,mut cb2)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());
    for (k,u,linear,samples) in &chosen {
        let mass:f64=samples.iter().map(|&i|ew[i] as f64).sum();
        let mut seed=0x9e3779b97f4a7c15u64 ^ ((*k*h+*u) as u64) ^ step;
        let mut row:Vec<f32>=(0..f).map(|_|{
            seed^=seed<<13; seed^=seed>>7; seed^=seed<<17;
            ((seed>>40) as f32/16777216.0-0.5)*2.0
        }).collect();
        let moments=|row:&[f32]| {
            let (mut sum,mut sq)=(0.0f64,0.0f64);
            for &i in samples { let y:f64=row.iter().zip(&x[i*f..(i+1)*f]).map(|(&w,&v)|w as f64*v as f64).sum();
                sum+=ew[i] as f64*y; sq+=ew[i] as f64*y*y; }
            (sum/mass,(sq/mass-(sum/mass).powi(2)).max(0.0).sqrt())
        };
        let (_,sd)=moments(&row);
        if sd<1e-8 { return Err(CudaCppError::message("recycle input has no variance")); }
        // Fresh row targets SD .2; clamp and round before setting its centering bias.
        for v in &mut row { *v=((*v*(0.2/sd) as f32).clamp(-1.0,1.0)*64.0).round()/64.0; }
        let (mean,sd)=moments(&row);
        let bias=((0.5-mean)*8128.0).round() as f32/8128.0;
        let out=k*l1out+u;
        for j in 0..f {
            let i=out*f+j;
            w1[i]=row[j]-alpha*sw[j*l1out+u];
            c1.push((i,row[j]-alpha*sloww[j*l1out+u]));
        }
        b1[out]=bias-alpha*sb[*u]; cb1.push((out,bias-alpha*slowb[*u]));
        let base=k*l2out*cols; let bb=k*l2out;
        fold(&mut w2[base..base+l2out*cols],&mut b2[bb..bb+l2out],l2out,cols,*u,h,*linear);
        for j in 0..l2out { c2.push((base+j*cols+u,0.0));c2.push((base+j*cols+h+u,0.0)); }
        eprintln!("  EXPERIMENT recycle step={step} bucket={k} unit={u} active={} constants=1/{linear} new_z_mean=.5 new_z_sd={sd:.6}",samples.len());
    }
    // Reset only changed row/column/bias moments. Keep shared state unchanged;
    // compensated residual slow rows preserve the new effective slow L1 row.
    for (k,_,_,_) in &chosen { for j in 0..l2out { let i=k*l2out+j; cb2.push((i,b2[i])); } }
    replace_group(ctx,&r.weights.l1w,&r.optimizer_states.l1w,&w1,&c1)?;
    replace_group(ctx,&r.weights.l1b,&r.optimizer_states.l1b,&b1,&cb1)?;
    replace_group(ctx,&r.weights.l2w,&r.optimizer_states.l2w,&w2,&c2)?;
    replace_group(ctx,&r.weights.l2b,&r.optimizer_states.l2b,&b2,&cb2)?;
    if let Some(qat)=&r.forward_workspace.qat_l1 { qat.refresh.set(true); }
    sfnn_forward_train_device_with_factorizer(ctx,batch,&r.weights,&r.forward_workspace,
        r.factorizer,r.factorizer_alpha,None,
        r.residual_count_gates(),r.factorizer_axis_confidences())?;
    let new_output=r.forward_workspace.output.download(ctx)?;
    let (mut max,mut sq,mut mass)=(0.0f64,0.0f64,0.0f64);
    for i in 0..n { if ew[i]>0.0 {let d=(new_output[i]-old_output[i]) as f64;
        max=max.max(d.abs());sq+=ew[i] as f64*d*d;mass+=ew[i] as f64;} }
    eprintln!("  EXPERIMENT recycle parity: units={} train_raw_max={:.6} train_raw_rms={:.6}; not a guarantee on unseen positions",chosen.len(),max*8128.0,(sq/mass).sqrt()*8128.0);
    if max>1e-4 { return Err(CudaCppError::message("recycle forward parity failed; experimental run stopped")); }
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn gpu_recycle_preserves_output_with_shared_qat_and_masks() {
        let ctx=Context::new(0).unwrap();let shape=crate::tests::tiny_sfnn_shape();
        let mut host=crate::tests::tiny_sfnn_weights(shape);
        let mut w=host.l1w.to_vec();let mut b=host.l1b.to_vec();
        for k in 0..2 { for j in 0..4 {w[k*12+j]=0.0;} b[k*3]=if k==0 {2.0} else {-2.0}; }
        host.l1w=&w;host.l1b=&b;host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,2048,1).unwrap();
        let stm:Vec<i32>=(0..2048).map(|i|((i/2)%4) as i32).collect();
        let nstm:Vec<i32>=stm.iter().map(|i|3-i).collect();
        let buckets:Vec<i32>=(0..2048).map(|i|i%2).collect();
        let ew:Vec<f32>=(0..2048).map(|i|if i%17==0 {0.0} else {1.0}).collect();
        r.device_batch.stm_indices.upload(&ctx,&stm).unwrap();r.device_batch.nstm_indices.upload(&ctx,&nstm).unwrap();
        r.device_batch.buckets.upload(&ctx,&buckets).unwrap();r.entry_weights.upload(&ctx,&ew).unwrap();
        r.prepare_l1_qat(&ctx,true).unwrap();
        let shared=r.weights.l1fw.as_ref().unwrap().download(&ctx).unwrap();
        sfnn_forward_train_device_with_factorizer(&ctx,&r.device_batch,&r.weights,&r.forward_workspace,
            r.factorizer,r.factorizer_alpha,None,None,None).unwrap();
        let before=r.forward_workspace.output.download(&ctx).unwrap();
        r.optimizer_states.l1w.momentum.fill(&ctx,0.25).unwrap();
        recycle_now(&r,&ctx,&r.device_batch,&r.entry_weights,9761,true).unwrap();
        let after=r.forward_workspace.output.download(&ctx).unwrap();
        assert!(before.iter().zip(after).all(|(a,b)|(a-b).abs()<1e-5));
        assert_eq!(shared,r.weights.l1fw.as_ref().unwrap().download(&ctx).unwrap());
        let m=r.optimizer_states.l1w.momentum.download(&ctx).unwrap();
        for k in 0..2 {for j in 0..4 {assert_eq!(m[k*12+j],0.0);assert_eq!(m[k*12+4+j],0.25);}}
        let a=r.forward_workspace.l2_input.download(&ctx).unwrap();
        assert!(a.chunks_exact(4).all(|row|row[0]<1.0));
        let w2=r.weights.l2w.download(&ctx).unwrap();
        for k in 0..2 {for j in 0..2 {assert_eq!(w2[k*8+j*4],0.0);assert_eq!(w2[k*8+j*4+2],0.0);}}
    }
    #[test] fn folding_constant_branches_preserves_preactivation() {
        for linear in [0.0,1.0] {
            let mut w=vec![0.2,-0.3,0.4,0.7,-0.2,0.1,0.5,-0.4];let mut b=vec![0.1,-0.2];
            let x=[1.0,0.3,linear,0.6];
            let before:Vec<f32>=(0..2).map(|j|b[j]+(0..4).map(|i|w[j*4+i]*x[i]).sum::<f32>()).collect();
            fold(&mut w,&mut b,2,4,0,2,linear);
            let fresh=[0.25,0.3,0.5,0.6];
            for j in 0..2 { let after=b[j]+(0..4).map(|i|w[j*4+i]*fresh[i]).sum::<f32>();assert!((after-before[j]).abs()<1e-6); }
        }
    }
}
