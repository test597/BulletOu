//! Temporary controlled experiment, explicitly enabled via environment variable.
//! Centering: z = W(x-c) + beta, beta = b + Wc.
//! Checkpoints/forward retain folded b. Only optimizer coordinates change.
//! `batch` uses this batch's feature means: moments intentionally accumulate
//! centered gradients across batches (not a fixed-coordinate optimizer).
//! Experimental bpu=1 only: a last-batch mean is not an accumulation-window mean.
use super::*;

pub(super) fn batch_mean(r: &SfnnTrainStepRunner, ctx: &Context) -> Result<Vec<f32>> {
    let scratch = F32Buffer::new(ctx, 32 * r.shape.ft_size)?;
    check(unsafe { ffi::bulletou_experiment_column_mean(ctx.as_ptr(),
        r.forward_workspace.combined.as_ptr(), scratch.as_ptr(), r.batch_size, r.shape.ft_size) })?;
    let mut values = scratch.download(ctx)?;
    values.truncate(r.shape.ft_size);
    Ok(values)
}

pub(super) fn center(r: &SfnnTrainStepRunner, ctx: &Context) -> Result<Option<Vec<f32>>> {
    match std::env::var("BULLETOU_EXPERIMENT_L1_CENTER") {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(s) => {
            if s == "batch" {
                return batch_mean(r,ctx).map(Some);
            }
            let c: f32 = s.parse().map_err(|_| CudaCppError::message("invalid experimental L1 center"))?;
            if !c.is_finite() { return Err(CudaCppError::message("nonfinite experimental L1 center")); }
            Ok(Some(vec![c; r.shape.ft_size]))
        }
        Err(_) => Err(CudaCppError::message("invalid experimental L1 center environment")),
    }
}

pub(super) fn group(ctx: &Context, w: &F32Buffer, b: &F32Buffer, ws: &RangerParamState,
    bs: &RangerParamState, gw: &F32Buffer, gb: &F32Buffer, cols: usize, c: &[f32], before: bool) -> Result<()> {
    group_layout(ctx,w,b,ws,bs,gw,gb,cols,c,before,false)
}

fn group_layout(ctx: &Context, w: &F32Buffer, b: &F32Buffer, ws: &RangerParamState,
    bs: &RangerParamState, gw: &F32Buffer, gb: &F32Buffer, cols: usize, c: &[f32], before: bool,
    column_major: bool) -> Result<()> {
    let weights = w.download(ctx)?;
    let slow = ws.slow_params.download(ctx)?;
    let mut biases = b.download(ctx)?;
    let mut slow_biases = bs.slow_params.download(ctx)?;
    let sign = if before { 1.0 } else { -1.0 };
    let rows=biases.len();
    let index=|row:usize,col:usize| if column_major { col*rows+row } else { row*cols+col };
    for row in 0..biases.len() {
        biases[row] += sign * (0..cols).map(|col|weights[index(row,col)]*c[col]).sum::<f32>();
        slow_biases[row] += sign * (0..cols).map(|col|slow[index(row,col)]*c[col]).sum::<f32>();
    }
    b.upload(ctx, &biases)?;
    bs.slow_params.upload(ctx, &slow_biases)?;
    if before {
        let bg = gb.download(ctx)?;
        let mut wg = gw.download(ctx)?;
        for row in 0..rows {
            for col in 0..cols { wg[index(row,col)] -= c[col] * bg[row]; }
        }
        gw.upload(ctx, &wg)?;
    }
    Ok(())
}

pub(super) fn transform(r: &SfnnTrainStepRunner, ctx: &Context, c: &[f32], before: bool) -> Result<()> {
    if r.shape.has_compact_l1() || r.factorizer.any_axis() || r.residual_count_gates_enabled {
        return Err(CudaCppError::message("experimental centering supports dense L1 none/shared without count gates only"));
    }
    group(ctx, &r.weights.l1w, &r.weights.l1b, &r.optimizer_states.l1w, &r.optimizer_states.l1b,
        &r.backward_workspace.l1w_gradients, &r.backward_workspace.l1b_gradients, r.shape.ft_size, c, before)?;
    if r.factorizer.shared {
        group_layout(ctx, r.weights.l1fw.as_ref().unwrap(), r.weights.l1fb.as_ref().unwrap(),
            r.optimizer_states.l1fw.as_ref().unwrap(), r.optimizer_states.l1fb.as_ref().unwrap(),
            &r.backward_workspace.l1fw_gradients, &r.backward_workspace.l1fb_gradients, r.shape.ft_size, c, before,true)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_gpu_l1_centered_sgd_matches_independent_equations() {
        let ctx=Context::new(0).unwrap();let shape=crate::tests::tiny_sfnn_shape();
        let r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        r.device_batch.stm_indices.upload(&ctx,&[0,1,2,3]).unwrap();
        r.device_batch.nstm_indices.upload(&ctx,&[3,2,1,0]).unwrap();
        r.device_batch.buckets.upload(&ctx,&[0,1,0,1]).unwrap();
        let forward=||sfnn_forward_train_device_with_factorizer(&ctx,&r.device_batch,&r.weights,&r.forward_workspace,
            r.factorizer,r.factorizer_alpha,None,None,None).unwrap();
        forward();let x=r.forward_workspace.combined.download(&ctx).unwrap();
        let old=r.forward_workspace.l1.download(&ctx).unwrap();
        let c=[0.1,0.2,0.3,0.4];let lr=0.03;let rows=shape.l1_out();let cols=shape.ft_size;
        let gw:Vec<f32>=(0..shape.num_stacks*rows*cols).map(|i|(i as f32-5.0)*0.013).collect();
        let gb:Vec<f32>=(0..shape.num_stacks*rows).map(|i|(i as f32-2.0)*0.04).collect();
        let fw:Vec<f32>=(0..rows*cols).map(|i|(i as f32-3.0)*0.019).collect();
        let fb:Vec<f32>=(0..rows).map(|i|(i as f32+1.0)*-0.07).collect();
        r.backward_workspace.l1w_gradients.upload(&ctx,&gw).unwrap();
        r.backward_workspace.l1b_gradients.upload(&ctx,&gb).unwrap();
        r.backward_workspace.l1fw_gradients.upload(&ctx,&fw).unwrap();
        r.backward_workspace.l1fb_gradients.upload(&ctx,&fb).unwrap();
        transform(&r,&ctx,&c,true).unwrap();
        for (w,g) in [(&r.weights.l1w,&r.backward_workspace.l1w_gradients),
            (&r.weights.l1b,&r.backward_workspace.l1b_gradients),
            (r.weights.l1fw.as_ref().unwrap(),&r.backward_workspace.l1fw_gradients),
            (r.weights.l1fb.as_ref().unwrap(),&r.backward_workspace.l1fb_gradients)] {
            let new:Vec<f32>=w.download(&ctx).unwrap().iter().zip(g.download(&ctx).unwrap()).map(|(w,g)|w-lr*g).collect();
            w.upload(&ctx,&new).unwrap();
        }
        transform(&r,&ctx,&c,false).unwrap();forward();
        let actual=r.forward_workspace.l1.download(&ctx).unwrap();
        for i in 0..4 {let k=i%2;for u in 0..rows {
            let residual=(0..cols).map(|j|(gw[(k*rows+u)*cols+j]-gb[k*rows+u]*c[j])*(x[i*cols+j]-c[j])).sum::<f32>()+gb[k*rows+u];
            let shared=(0..cols).map(|j|(fw[j*rows+u]-fb[u]*c[j])*(x[i*cols+j]-c[j])).sum::<f32>()+fb[u];
            let expected=old[i*rows+u]-lr*(residual+shared);
            assert!((actual[i*rows+u]-expected).abs()<2e-6,"sample {i} unit {u}: {} != {expected}",actual[i*rows+u]);
        }}
    }
    #[test]
    fn shared_column_major_centered_bias_matches_layout() {
        let ctx=Context::new(0).unwrap();
        // Three inputs, two output units: shared storage is [input][output].
        let wh=[0.2,0.7,-0.1,0.4,0.3,-0.6]; let bh=[0.05,-0.2]; let c=[0.1,0.3,0.5];
        let w=F32Buffer::from_host(&ctx,&wh).unwrap();let b=F32Buffer::from_host(&ctx,&bh).unwrap();
        let ws=RangerParamState::from_host_weights(&ctx,&wh).unwrap();let bs=RangerParamState::from_host_weights(&ctx,&bh).unwrap();
        let wg=F32Buffer::from_host(&ctx,&[0.0;6]).unwrap();let bg=F32Buffer::from_host(&ctx,&[0.4,-0.2]).unwrap();
        group_layout(&ctx,&w,&b,&ws,&bs,&wg,&bg,3,&c,true,true).unwrap();
        let beta=b.download(&ctx).unwrap();let grad=wg.download(&ctx).unwrap();
        for r in 0..2 {
            let expected=bh[r]+(0..3).map(|j|wh[j*2+r]*c[j]).sum::<f32>();
            assert!((beta[r]-expected).abs()<1e-6,"row {r}: {} != {expected}",beta[r]);
            for j in 0..3 {assert!((grad[j*2+r]+[0.4,-0.2][r]*c[j]).abs()<1e-6);}
        }
    }
    #[test]
    fn gpu_mean_penalty_matches_finite_difference_and_leaves_skip_untouched() {
        let ctx=Context::new(0).unwrap();
        let z: Vec<f64> = vec![1.5,-1.0,7.0, 0.2,0.4,8.0, 2.5,-3.0,9.0, 0.0,0.2,10.0];
        let buckets=[0,1,0,1]; let strength=0.01;
        for weights in [[1.0,1.0,1.0,1.0], [0.0,1.0,2.0,0.5], [0.0;4]] {
        let loss=|z:&[f64]| {
            let mut loss=0.0;
            for b in 0..3 { for u in 0..2 {
                let mass:f64=(0..4).filter(|&i|buckets[i]==b).map(|i|weights[i]).sum();
                if mass > 0.0 { let mean=(0..4).filter(|&i|buckets[i]==b)
                    .map(|i|weights[i]*z[3*i+u]).sum::<f64>()/mass;
                    loss+=strength*mass/8.0*(mean.abs()-0.9).max(0.0).powi(2); }
            }} loss
        };
        let dz=F32Buffer::from_host(&ctx,&z.iter().map(|&v|v as f32).collect::<Vec<_>>()).unwrap();
        let db=I32Buffer::from_host(&ctx,&buckets).unwrap();
        let dg=F32Buffer::from_host(&ctx,&[0.7;12]).unwrap();
        let dw=F32Buffer::from_host(&ctx,&weights.map(|v|v as f32)).unwrap();
        check(unsafe { ffi::bulletou_experiment_mean_penalty_test(ctx.as_ptr(),dz.as_ptr(),db.as_ptr(),dw.as_ptr(),dg.as_ptr(),4,2,3,3,strength as f32) }).unwrap();
        let actual=dg.download(&ctx).unwrap();
        for i in 0..12 {
            let mut hi=z.clone(); let mut lo=z.clone(); hi[i]+=1e-5; lo[i]-=1e-5;
            let expected=(loss(&hi)-loss(&lo))/2e-5;
            assert!((actual[i] as f64-0.7-expected).abs()<1e-6,"index {i}");
        }
        }
    }
    #[test]
    fn gpu_folded_updates_match_explicit_centered_ranger_including_lookahead() {
        let ctx = Context::new(0).unwrap();
        let c = [0.1, 0.3];
        let wh = [0.2, -0.1, 0.3, 0.4];
        let bh = [0.05, -0.2];
        let beta: Vec<f32> = (0..2).map(|r| bh[r]+wh[2*r]*c[0]+wh[2*r+1]*c[1]).collect();
        let w = F32Buffer::from_host(&ctx,&wh).unwrap();
        let b = F32Buffer::from_host(&ctx,&bh).unwrap();
        let ws = RangerParamState::from_host_weights(&ctx,&wh).unwrap();
        let bs = RangerParamState::from_host_weights(&ctx,&bh).unwrap();
        let rw = F32Buffer::from_host(&ctx,&wh).unwrap();
        let rb = F32Buffer::from_host(&ctx,&beta).unwrap();
        let rws = RangerParamState::from_host_weights(&ctx,&wh).unwrap();
        let rbs = RangerParamState::from_host_weights(&ctx,&beta).unwrap();
        for step in 1..=18 {
            let g = [0.4/(step as f32), -0.2];
            let x = [0.7, 0.9];
            let wg = F32Buffer::from_host(&ctx,&[g[0]*x[0],g[0]*x[1],g[1]*x[0],g[1]*x[1]]).unwrap();
            let bg = F32Buffer::from_host(&ctx,&g).unwrap();
            // Optimizer consumes (clears) its gradient buffers; reference needs its own.
            let rbg = F32Buffer::from_host(&ctx,&g).unwrap();
            let rg = F32Buffer::from_host(&ctx,&[g[0]*(x[0]-c[0]),g[0]*(x[1]-c[1]),g[1]*(x[0]-c[0]),g[1]*(x[1]-c[1])]).unwrap();
            let mut p = RangerUpdateParams::default(); p.radam.step=step;
            group(&ctx,&w,&b,&ws,&bs,&wg,&bg,2,&c,true).unwrap();
            update_param_group(&ctx,p,&wg,&w,&ws).unwrap();
            update_param_group(&ctx,p,&bg,&b,&bs).unwrap();
            group(&ctx,&w,&b,&ws,&bs,&wg,&bg,2,&c,false).unwrap();
            update_param_group(&ctx,p,&rg,&rw,&rws).unwrap();
            update_param_group(&ctx,p,&rbg,&rb,&rbs).unwrap();
            let actual_w=w.download(&ctx).unwrap(); let expected_w=rw.download(&ctx).unwrap();
            let actual_b=b.download(&ctx).unwrap(); let expected_beta=rb.download(&ctx).unwrap();
            for i in 0..4 { assert!((actual_w[i]-expected_w[i]).abs()<1e-6); }
            for r in 0..2 {
                let expected_b=expected_beta[r]-expected_w[2*r]*c[0]-expected_w[2*r+1]*c[1];
                assert!((actual_b[r]-expected_b).abs()<1e-6, "step={step} row={r} actual={} expected={expected_b}", actual_b[r]);
            }
        }
    }
    #[test]
    fn gpu_centered_coordinates_preserve_forward_gradient_and_slow_weights() {
        let ctx = Context::new(0).unwrap();
        let wh = [0.2, -0.1, 0.3, 0.4];
        let bh = [0.05, -0.2];
        let x = [0.7, 0.9];
        let gh = [0.6, -0.3];
        let w = F32Buffer::from_host(&ctx, &wh).unwrap();
        let b = F32Buffer::from_host(&ctx, &bh).unwrap();
        let ws = RangerParamState::from_host_weights(&ctx, &wh).unwrap();
        let bs = RangerParamState::from_host_weights(&ctx, &bh).unwrap();
        let wg = F32Buffer::from_host(&ctx, &[gh[0]*x[0], gh[0]*x[1], gh[1]*x[0], gh[1]*x[1]]).unwrap();
        let bg = F32Buffer::from_host(&ctx, &gh).unwrap();
        let c = 0.25;
        group(&ctx, &w, &b, &ws, &bs, &wg, &bg, 2, &[c;2], true).unwrap();
        let beta = b.download(&ctx).unwrap();
        let grad = wg.download(&ctx).unwrap();
        for r in 0..2 {
            let original = wh[2*r]*x[0]+wh[2*r+1]*x[1]+bh[r];
            let centered = wh[2*r]*(x[0]-c)+wh[2*r+1]*(x[1]-c)+beta[r];
            assert!((original-centered).abs() < 1e-6);
            for j in 0..2 { assert!((grad[2*r+j]-gh[r]*(x[j]-c)).abs() < 1e-6); }
        }
        group(&ctx, &w, &b, &ws, &bs, &wg, &bg, 2, &[c;2], false).unwrap();
        for (actual, expected) in b.download(&ctx).unwrap().iter().zip(bh) { assert!((actual-expected).abs()<1e-6); }
        for (actual, expected) in bs.slow_params.download(&ctx).unwrap().iter().zip(bh) { assert!((actual-expected).abs()<1e-6); }
    }
    #[test]
    fn gpu_column_mean_matches_cpu() {
        let ctx = Context::new(0).unwrap();
        let (rows, cols) = (173, 67);
        let x: Vec<f32> = (0..rows*cols).map(|i| ((i*13)%101) as f32 / 100.0).collect();
        let input = F32Buffer::from_host(&ctx, &x).unwrap();
        let scratch = F32Buffer::new(&ctx, 32*cols).unwrap();
        check(unsafe { ffi::bulletou_experiment_column_mean(ctx.as_ptr(),input.as_ptr(),scratch.as_ptr(),rows,cols) }).unwrap();
        let means = scratch.download(&ctx).unwrap();
        for c in 0..cols {
            let expected = (0..rows).map(|r| x[r*cols+c] as f64).sum::<f64>() / rows as f64;
            assert!((means[c] as f64-expected).abs()<1e-6);
        }
    }
}
