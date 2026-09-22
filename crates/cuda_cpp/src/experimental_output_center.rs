//! Explicit bpu=1 diagnostic: center optimizer coordinates of L2/L3 only.
//! Forward/checkpoints retain the folded affine form. Not native Ranger.
use super::*;

pub(super) struct Centers { l2: Vec<f32>, l3: Vec<f32>, device: bool }
fn mean_into(ctx:&Context,x:&F32Buffer,n:usize,d:usize,scratch:&F32Buffer)->Result<()> {
    check(unsafe {ffi::bulletou_experiment_column_mean(ctx.as_ptr(),x.as_ptr(),scratch.as_ptr(),n,d)})
}
fn mean_device(ctx:&Context,x:&F32Buffer,n:usize,d:usize)->Result<F32Buffer> {
    let scratch=F32Buffer::new(ctx,32*d)?;
    mean_into(ctx,x,n,d,&scratch)?;
    Ok(scratch)
}
fn mean(ctx:&Context,x:&F32Buffer,n:usize,d:usize)->Result<Vec<f32>> {
    let scratch=mean_device(ctx,x,n,d)?;
    let mut c=scratch.download(ctx)?;c.truncate(d);Ok(c)
}
pub(super) fn capture(r:&SfnnTrainStepRunner,ctx:&Context,p:RangerUpdateParams,lr:SfnnLayerLrMultipliers)->Result<Option<Centers>> {
    let gpu=match std::env::var("BULLETOU_EXPERIMENT_OUTPUT_CENTER") {
        Err(std::env::VarError::NotPresent)=>return Ok(None),
        Ok(s) if s=="1"=>false,
        Ok(s) if s=="gpu"=>true,
        _=>return Err(CudaCppError::message("output centering expects 1 (host reference) or gpu")),
    };
    if r.weights.l2fw.is_some() || r.weights.l3fw.is_some() || r.factorizer.any_axis()
        || r.residual_count_gates_enabled || lr.tatara_weight_clip || p.radam.decay!=0.0
        || lr.norm_loss_strength!=0.0 || lr.factorizer_residual_decay!=0.0 || lr.saturation_penalty!=0.0
        || p.radam.min_weight > -1e20 || p.radam.max_weight < 1e20
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_CENTER").is_some()
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_PROJECT_UPDATE").is_some() {
        return Err(CudaCppError::message("output centering requires none/shared, no gates/clip/penalties/other centering"));
    }
    static ANNOUNCE:std::sync::Once=std::sync::Once::new();
    ANNOUNCE.call_once(||eprintln!("  EXPERIMENT output centering: L2/L3 batch-global input means; FT/L1 untouched; folded bias fast/slow coordinates; bpu=1 diagnostic; implementation={}",if gpu {"gpu"}else{"host reference"}));
    if gpu {
        let (c2,c3)=r.experimental_output_centers.as_ref().ok_or_else(||CudaCppError::message("configure GPU output centering before constructing the runner"))?;
        mean_into(ctx,&r.forward_workspace.l2_input,r.batch_size,r.shape.l2_in(),c2)?;
        mean_into(ctx,&r.forward_workspace.l2,r.batch_size,r.shape.l2_size,c3)?;
        return Ok(Some(Centers{l2:vec![],l3:vec![],device:true}));
    }
    Ok(Some(Centers{l2:mean(ctx,&r.forward_workspace.l2_input,r.batch_size,r.shape.l2_in())?,
        l3:mean(ctx,&r.forward_workspace.l2,r.batch_size,r.shape.l2_size)?,device:false}))
}
fn group_device(ctx:&Context,w:&F32Buffer,b:&F32Buffer,ws:&RangerParamState,bs:&RangerParamState,
    gw:&F32Buffer,gb:&F32Buffer,cols:usize,c:&F32Buffer,before:bool)->Result<()> {
    check(unsafe {ffi::bulletou_experiment_center_affine(ctx.as_ptr(),w.as_ptr(),b.as_ptr(),
        ws.slow_params.as_ptr(),bs.slow_params.as_ptr(),gw.as_ptr(),gb.as_ptr(),c.as_ptr(),b.len(),cols,before as i32)})
}
pub(super) fn transform(r:&SfnnTrainStepRunner,ctx:&Context,c:&Centers,before:bool)->Result<()> {
    if c.device {
        let (c2,c3)=r.experimental_output_centers.as_ref().ok_or_else(||CudaCppError::message("missing GPU output centering workspace"))?;
        group_device(ctx,&r.weights.l2w,&r.weights.l2b,&r.optimizer_states.l2w,&r.optimizer_states.l2b,
            &r.backward_workspace.l2w_gradients,&r.backward_workspace.l2b_gradients,r.shape.l2_in(),c2,before)?;
        return group_device(ctx,&r.weights.l3w,&r.weights.l3b,&r.optimizer_states.l3w,&r.optimizer_states.l3b,
            &r.backward_workspace.l3w_gradients,&r.backward_workspace.l3b_gradients,r.shape.l2_size,c3,before);
    }
    experimental_l1_center::group(ctx,&r.weights.l2w,&r.weights.l2b,&r.optimizer_states.l2w,&r.optimizer_states.l2b,
        &r.backward_workspace.l2w_gradients,&r.backward_workspace.l2b_gradients,r.shape.l2_in(),&c.l2,before)?;
    experimental_l1_center::group(ctx,&r.weights.l3w,&r.weights.l3b,&r.optimizer_states.l3w,&r.optimizer_states.l3b,
        &r.backward_workspace.l3w_gradients,&r.backward_workspace.l3b_gradients,r.shape.l2_size,&c.l3,before)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn gpu_and_host_centering_match_ranger_fast_slow_and_moments() {
        let ctx=Context::new(0).unwrap();
        for cols in [14,64] {
            let rows=9;
            let wh:Vec<f32>=(0..rows*cols).map(|i|((i*11%61) as f32-30.0)*0.003).collect();
            let bh:Vec<f32>=(0..rows).map(|i|0.03*i as f32).collect();
            let create=|| (F32Buffer::from_host(&ctx,&wh).unwrap(),F32Buffer::from_host(&ctx,&bh).unwrap(),
                RangerParamState::from_host_weights(&ctx,&wh).unwrap(),RangerParamState::from_host_weights(&ctx,&bh).unwrap());
            let (w,b,ws,bs)=create();let (rw,rb,rws,rbs)=create();
            for step in 1..=18 {
                let c:Vec<f32>=(0..cols).map(|j|0.1+0.001*(j+step as usize) as f32).collect();
                let dc=F32Buffer::from_host(&ctx,&c).unwrap();
                let gh:Vec<f32>=(0..rows*cols).map(|i|((i*7%41) as f32-20.0)*0.001/step as f32).collect();
                let gbh:Vec<f32>=(0..rows).map(|i|((i as f32)-4.0)*0.002).collect();
                let g=F32Buffer::from_host(&ctx,&gh).unwrap();let bg=F32Buffer::from_host(&ctx,&gbh).unwrap();
                let rg=F32Buffer::from_host(&ctx,&gh).unwrap();let rbg=F32Buffer::from_host(&ctx,&gbh).unwrap();
                group_device(&ctx,&w,&b,&ws,&bs,&g,&bg,cols,&dc,true).unwrap();
                experimental_l1_center::group(&ctx,&rw,&rb,&rws,&rbs,&rg,&rbg,cols,&c,true).unwrap();
                let mut p=RangerUpdateParams::default();p.radam.step=step;
                for (w,g,s) in [(&w,&g,&ws),(&b,&bg,&bs),(&rw,&rg,&rws),(&rb,&rbg,&rbs)] {update_param_group(&ctx,p,g,w,s).unwrap();}
                group_device(&ctx,&w,&b,&ws,&bs,&g,&bg,cols,&dc,false).unwrap();
                experimental_l1_center::group(&ctx,&rw,&rb,&rws,&rbs,&rg,&rbg,cols,&c,false).unwrap();
                for (a,b) in [(&w,&rw),(&b,&rb),(&ws.slow_params,&rws.slow_params),(&bs.slow_params,&rbs.slow_params),
                    (&ws.momentum,&rws.momentum),(&ws.velocity,&rws.velocity),(&bs.momentum,&rbs.momentum),(&bs.velocity,&rbs.velocity)] {
                    for (a,b) in a.download(&ctx).unwrap().iter().zip(b.download(&ctx).unwrap()) {
                        assert!((a-b).abs()<2e-6,"cols={cols} step={step}: {a} vs {b}");
                    }
                }
            }
        }
    }
    #[test] fn output_input_means_match_cpu_for_actual_widths() {
        let ctx=Context::new(0).unwrap();
        for d in [14,64] {
            let n=1031;
            let x:Vec<f32>=(0..n*d).map(|i|((i*13%197) as f32)/197.0).collect();
            let input=F32Buffer::from_host(&ctx,&x).unwrap();
            let c=mean(&ctx,&input,n,d).unwrap();
            for j in 0..d {
                let expected=(0..n).map(|i|x[i*d+j] as f64).sum::<f64>()/n as f64;
                assert!((c[j] as f64-expected).abs()<1e-6);
            }
            let scratch=mean_device(&ctx,&input,n,d).unwrap();
            input.upload(&ctx,&vec![0.375;n*d]).unwrap();
            mean_into(&ctx,&input,n,d,&scratch).unwrap();
            for &v in &scratch.download(&ctx).unwrap()[..d] {
                assert!((v-0.375).abs()<1e-6,"reused workspace retained a previous mean");
            }
        }
    }
    #[test] fn output_center_roundtrip_and_independent_sgd() {
        let ctx=Context::new(0).unwrap();let s=crate::tests::tiny_sfnn_shape();
        let r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(s),4,1).unwrap();
        let centers=Centers{l2:vec![0.2;s.l2_in()],l3:vec![0.35;s.l2_size],device:false};
        for (w,b,gw,gb,c) in [(&r.weights.l2w,&r.weights.l2b,&r.backward_workspace.l2w_gradients,&r.backward_workspace.l2b_gradients,&centers.l2),
            (&r.weights.l3w,&r.weights.l3b,&r.backward_workspace.l3w_gradients,&r.backward_workspace.l3b_gradients,&centers.l3)] {
            let oldw=w.download(&ctx).unwrap();let oldb=b.download(&ctx).unwrap();
            let g:Vec<f32>=(0..oldw.len()).map(|i|0.01*(i as f32-3.0)).collect();
            let h:Vec<f32>=(0..oldb.len()).map(|i|-0.02*(i+1) as f32).collect();
            gw.upload(&ctx,&g).unwrap();gb.upload(&ctx,&h).unwrap();
            let (ws,bs)=if oldw.len()==r.weights.l2w.len(){(&r.optimizer_states.l2w,&r.optimizer_states.l2b)}else{(&r.optimizer_states.l3w,&r.optimizer_states.l3b)};
            experimental_l1_center::group(&ctx,w,b,ws,bs,gw,gb,c.len(),c,true).unwrap();
            let nw:Vec<f32>=w.download(&ctx).unwrap().iter().zip(gw.download(&ctx).unwrap()).map(|(w,g)|w-0.03*g).collect();
            let nb:Vec<f32>=b.download(&ctx).unwrap().iter().zip(&h).map(|(b,h)|b-0.03*h).collect();
            w.upload(&ctx,&nw).unwrap();b.upload(&ctx,&nb).unwrap();
            experimental_l1_center::group(&ctx,w,b,ws,bs,gw,gb,c.len(),c,false).unwrap();
            let actualw=w.download(&ctx).unwrap();let actualb=b.download(&ctx).unwrap();
            for row in 0..oldb.len() {
                let x:Vec<f32>=(0..c.len()).map(|j|0.07*j as f32).collect();
                let old=oldb[row]+(0..c.len()).map(|j|oldw[row*c.len()+j]*x[j]).sum::<f32>();
                let expected=old-0.03*(h[row]+(0..c.len()).map(|j|(g[row*c.len()+j]-h[row]*c[j])*(x[j]-c[j])).sum::<f32>());
                let actual=actualb[row]+(0..c.len()).map(|j|actualw[row*c.len()+j]*x[j]).sum::<f32>();
                assert!((actual-expected).abs()<1e-6);
            }
        }
    }
}
