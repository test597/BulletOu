//! Opt-in optimizer coordinates of L2/L3; GPU production option and host reference.
//! Forward/checkpoints retain the folded affine form. Not native Ranger.
//! Diagnostic variant: BULLETOU_EXPERIMENT_OUTPUT_CENTER_BUCKET=1 uses
//! count-weighted bucket means with the public centering option. This is not
//! the default: reduced L1 mean saturation alone did not resolve FT/unit-max
//! saturation in the progress8 experiment.
use super::*;

pub(super) struct Centers { l2: Vec<f32>, l3: Vec<f32>, device: bool }
pub(super) fn accumulate(r:&mut SfnnTrainStepRunner,ctx:&Context,enabled:bool)->Result<()> {
    accumulate_from_slot(r,ctx,enabled,None)
}
pub(super) fn accumulate_from_slot(r:&mut SfnnTrainStepRunner,ctx:&Context,enabled:bool,slot:Option<usize>)->Result<()> {
    if !enabled { r.output_center_batches=0; return Ok(()); }
    let width=|d| if r.output_center_bucketwise {r.shape.num_stacks*(d+1)} else {d};
    let (d2,d3)=(width(r.shape.l2_in()),width(r.shape.l2_size));
    if r.experimental_output_centers.is_none() {
        r.experimental_output_centers=Some((F32Buffer::new(ctx,32*d2)?,F32Buffer::new(ctx,32*d3)?));
    }
    if r.output_center_sums.is_none() {
        r.output_center_sums=Some((F32Buffer::new(ctx,d2)?,F32Buffer::new(ctx,d3)?));
    }
    let (c2,c3)=r.experimental_output_centers.as_ref().unwrap();
    let (sum2,sum3)=r.output_center_sums.as_ref().unwrap();
    if r.output_center_batches==0 {sum2.fill(ctx,0.0)?;sum3.fill(ctx,0.0)?;}
    if r.output_center_bucketwise {
        let ids=slot.map_or(&r.device_batch.buckets,|i|&r.upload_slots[i].device_batch.buckets);
        for (x,d,c) in [(&r.forward_workspace.l2_input,r.shape.l2_in(),c2),(&r.forward_workspace.l2,r.shape.l2_size,c3)] {
            check(unsafe{ffi::bulletou_experiment_bucket_mean(ctx.as_ptr(),x.as_ptr(),ids.as_ptr(),c.as_ptr(),r.batch_size,d,r.shape.num_stacks)})?;
        }
    } else {
        mean_into(ctx,&r.forward_workspace.l2_input,r.batch_size,r.shape.l2_in(),c2)?;
        mean_into(ctx,&r.forward_workspace.l2,r.batch_size,r.shape.l2_size,c3)?;
    }
    axpy_device(ctx,d2,1.0,c2,sum2,sum2)?;
    axpy_device(ctx,d3,1.0,c3,sum3,sum3)?;
    r.output_center_batches+=1;
    if r.output_center_bucketwise {
        static ANNOUNCE:std::sync::Once=std::sync::Once::new();
        ANNOUNCE.call_once(||eprintln!("  EXPERIMENT: L2/L3 centering uses per-bucket means, count-weighted across accumulated batches"));
    }
    Ok(())
}
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
    let gpu=if lr.l2_l3_center { true } else { match std::env::var("BULLETOU_EXPERIMENT_OUTPUT_CENTER") {
        Err(std::env::VarError::NotPresent)=>return Ok(None),
        Ok(s) if s=="1"=>false,
        Ok(s) if s=="gpu"=>true,
        _=>return Err(CudaCppError::message("output centering expects 1 (host reference) or gpu")),
    }};
    if r.output_center_bucketwise && !lr.l2_l3_center {
        return Err(CudaCppError::message("bucketwise centering experiment requires --sfnn-l2-l3-center"));
    }
    if lr.update_scope != SfnnUpdateScope::All {
        return Err(CudaCppError::message("L2/L3 centering requires update-scope=all"));
    }
    if r.shape.l2_in() > 256 || r.shape.l2_size > 256 || r.weights.l2b.len() > 65536 {
        return Err(CudaCppError::message("L2/L3 centering supports L2 input/output widths <=256 and stacks*L2 <=65536"));
    }
    if r.weights.l2fw.is_some() || r.weights.l3fw.is_some() || r.factorizer.any_axis()
        || r.residual_count_gates_enabled || lr.tatara_weight_clip || p.radam.decay!=0.0
        || lr.norm_loss_strength!=0.0 || lr.factorizer_residual_decay!=0.0 || lr.saturation_penalty!=0.0
        || p.radam.min_weight > -1e20 || p.radam.max_weight < 1e20
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_CENTER").is_some()
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_PROJECT_UPDATE").is_some() {
        return Err(CudaCppError::message("L2/L3 centering requires none/shared, no gates/clip/penalties/other centering"));
    }
    static ANNOUNCE:std::sync::Once=std::sync::Once::new();
    ANNOUNCE.call_once(||eprintln!("  L2/L3 centering: input means over accumulated batches; FT/L1 untouched; folded bias fast/slow coordinates; implementation={}",if gpu {"gpu"}else{"host reference"}));
    if gpu {
        let (c2,c3)=r.experimental_output_centers.as_ref().ok_or_else(||CudaCppError::message("configure GPU output centering before constructing the runner"))?;
        if lr.l2_l3_center && r.output_center_batches>0 {
            let (sum2,sum3)=r.output_center_sums.as_ref().unwrap();
            if r.output_center_bucketwise {
                for (s,c,d) in [(sum2,c2,r.shape.l2_in()),(sum3,c3,r.shape.l2_size)] {
                    check(unsafe{ffi::bulletou_experiment_bucket_centers(ctx.as_ptr(),s.as_ptr(),c.as_ptr(),d,r.shape.num_stacks)})?;
                }
                return Ok(Some(Centers{l2:vec![],l3:vec![],device:true}));
            }
            c2.fill(ctx,0.0)?; c3.fill(ctx,0.0)?;
            axpy_device(ctx,r.shape.l2_in(),1.0/r.output_center_batches as f32,sum2,c2,c2)?;
            axpy_device(ctx,r.shape.l2_size,1.0/r.output_center_batches as f32,sum3,c3,c3)?;
        } else {
            mean_into(ctx,&r.forward_workspace.l2_input,r.batch_size,r.shape.l2_in(),c2)?;
            mean_into(ctx,&r.forward_workspace.l2,r.batch_size,r.shape.l2_size,c3)?;
        }
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
        if r.output_center_bucketwise {
            for (w,b,ws,bs,gw,gb,d,rows_per_bucket,center) in [
                (&r.weights.l2w,&r.weights.l2b,&r.optimizer_states.l2w,&r.optimizer_states.l2b,&r.backward_workspace.l2w_gradients,&r.backward_workspace.l2b_gradients,r.shape.l2_in(),r.shape.l2_size,c2),
                (&r.weights.l3w,&r.weights.l3b,&r.optimizer_states.l3w,&r.optimizer_states.l3b,&r.backward_workspace.l3w_gradients,&r.backward_workspace.l3b_gradients,r.shape.l2_size,1,c3)] {
                check(unsafe{ffi::bulletou_experiment_center_affine_bucket(ctx.as_ptr(),w.as_ptr(),b.as_ptr(),ws.slow_params.as_ptr(),bs.slow_params.as_ptr(),gw.as_ptr(),gb.as_ptr(),center.as_ptr(),b.len(),d,rows_per_bucket,before as i32)})?;
            }
            return Ok(());
        }
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
    #[test] fn bucket_means_are_count_weighted_and_transform_matching_rows() {
        let ctx=Context::new(0).unwrap();let s=crate::tests::tiny_sfnn_shape();
        let mut w=crate::tests::tiny_sfnn_weights(s);w.l2fw=None;w.l2fb=None;w.l3fw=None;w.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,w,4,1).unwrap();r.output_center_bucketwise=true;
        for (ids,value) in [([0,0,0,1],0.2),([0,1,1,1],0.8)] {
            r.device_batch.buckets.upload(&ctx,&ids).unwrap();
            r.forward_workspace.l2_input.fill(&ctx,value).unwrap();r.forward_workspace.l2.fill(&ctx,value).unwrap();
            accumulate(&mut r,&ctx,true).unwrap();
        }
        let p=SfnnLayerLrMultipliers{l2_l3_center:true,..Default::default()};
        let c=capture(&r,&ctx,Default::default(),p).unwrap().unwrap();
        let values=r.experimental_output_centers.as_ref().unwrap().0.download(&ctx).unwrap();
        assert!((values[0]-0.35).abs()<1e-6);assert!((values[s.l2_in()+1]-0.65).abs()<1e-6);
        r.backward_workspace.l2w_gradients.fill(&ctx,0.0).unwrap();r.backward_workspace.l2b_gradients.fill(&ctx,1.0).unwrap();
        let old=r.weights.l2b.download(&ctx).unwrap();
        transform(&r,&ctx,&c,true).unwrap();
        let grad=r.backward_workspace.l2w_gradients.download(&ctx).unwrap();
        for row in 0..s.num_stacks*s.l2_size { for col in 0..s.l2_in() {
            let expected=if row/s.l2_size==0 {-0.35}else{-0.65};
            assert!((grad[row*s.l2_in()+col]-expected).abs()<1e-6);
        }}
        transform(&r,&ctx,&c,false).unwrap();
        for (a,b) in old.iter().zip(r.weights.l2b.download(&ctx).unwrap()) {assert!((a-b).abs()<1e-6);}
    }
    #[test] fn accumulated_centering_matches_one_large_batch() {
        let ctx=Context::new(0).unwrap(); let upload=Context::new(0).unwrap();
        for input_size in [4,133_578] { for centered in [false,true] { for bucketwise in [false,true] {
        let s=SfnnForwardShape{input_size,..crate::tests::tiny_sfnn_shape()};
        let mut weights=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        weights.shape=s;
        let ft_weights:Vec<f32>=(0..input_size*s.ft_size).map(|i|0.05+0.03*(i%9) as f32).collect();
        weights.l0w=&ft_weights;
        weights.l2fw=None; weights.l2fb=None; weights.l3fw=None; weights.l3fb=None;
        let policy=SfnnLayerLrMultipliers{l2_l3_center:centered,..Default::default()};
        for bpu in [1,2,4] { for mode in [0,1,2] {
            let n=4*bpu;
            let mut small=SfnnTrainStepRunner::new(&ctx,weights,4,1).unwrap();
            let mut large=SfnnTrainStepRunner::new(&ctx,weights,n,1).unwrap();
            small.output_center_bucketwise=bucketwise; large.output_center_bucketwise=bucketwise;
            let stm:Vec<i32>=(0..n).map(|i|((i+i/4)%4) as i32).collect();
            let nstm:Vec<i32>=(0..n).map(|i|((i*3+1)%4) as i32).collect();
            let buckets:Vec<i32>=(0..n).map(|i|(i%2) as i32).collect();
            let targets:Vec<f32>=(0..n).map(|i|0.1+0.1*(i%8) as f32).collect();
            let entries=vec![1.0;n];
            let batch=|start:usize,end:usize| SfnnTrainStepHostBatch {
                stm_indices:&stm[start..end],nstm_indices:&nstm[start..end],buckets:&buckets[start..end],
                targets:&targets[start..end],entry_weights:&entries[start..end],batch_size:end-start,max_active:1 };
            for step in 1..=12 {
                let mut p=RangerUpdateParams::default();p.radam.step=step;
                large.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,p,ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch(0,n),true,true,policy).unwrap();
                p.radam.gradient_factor=1.0/bpu as f32;
                for j in 0..bpu {
                    if mode==2 {
                        small.step_profiled_no_readback_with_update_and_lr_multipliers(&ctx,p,ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch(j*4,j*4+4),j+1==bpu,policy).unwrap();
                    } else if mode==1 {
                        small.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,&upload,p,ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch(j*4,j*4+4),true,j+1==bpu,policy).unwrap();
                    } else {
                        small.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,p,ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch(j*4,j*4+4),true,j+1==bpu,policy).unwrap();
                    }
                    assert_eq!(small.output_center_batches,if j+1==bpu || !centered {0}else{j+1});
                }
                for (tensor,(a,b)) in [(&small.weights.l0w,&large.weights.l0w),(&small.weights.l1w,&large.weights.l1w),
                    (&small.weights.l2w,&large.weights.l2w),(&small.weights.l2b,&large.weights.l2b),
                    (&small.weights.l3w,&large.weights.l3w),(&small.weights.l3b,&large.weights.l3b),
                    (&small.optimizer_states.l2w.momentum,&large.optimizer_states.l2w.momentum),
                    (&small.optimizer_states.l3b.slow_params,&large.optimizer_states.l3b.slow_params)].into_iter().enumerate() {
                    for (a,b) in a.download(&ctx).unwrap().iter().zip(b.download(&ctx).unwrap()) {
                        assert!((a-b).abs()<2e-5,"input={input_size},center={centered},tensor={tensor},bpu={bpu},mode={mode},step={step}: {a} != {b}");
                    }
                }
            }
        }}}}}
    }
    #[test] fn accumulated_means_reset_on_update_and_restore() {
        let ctx=Context::new(0).unwrap(); let s=crate::tests::tiny_sfnn_shape();
        let mut weights=crate::tests::tiny_sfnn_weights(s);
        weights.l2fw=None; weights.l2fb=None; weights.l3fw=None; weights.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,weights,4,1).unwrap();
        let snapshot=r.snapshot_device(&ctx).unwrap();
        let policy=SfnnLayerLrMultipliers{l2_l3_center:true,..Default::default()};
        for c in [0.1,0.3,0.5,0.7] {
            r.forward_workspace.l2_input.fill(&ctx,c).unwrap();
            r.forward_workspace.l2.fill(&ctx,c*0.5).unwrap();
            accumulate(&mut r,&ctx,true).unwrap();
        }
        capture(&r,&ctx,Default::default(),policy).unwrap();
        let (c2,c3)=r.experimental_output_centers.as_ref().unwrap();
        assert!((c2.download(&ctx).unwrap()[0]-0.4).abs()<1e-6);
        assert!((c3.download(&ctx).unwrap()[0]-0.2).abs()<1e-6);
        r.backward_workspace.l0w_gradients.fill(&ctx,1.0).unwrap();
        r.copy_state_from_device(&ctx,&snapshot).unwrap();
        assert_eq!(r.output_center_batches,0);
        assert!(r.backward_workspace.l0w_gradients.download(&ctx).unwrap().iter().all(|&v|v==0.0));
        r.forward_workspace.l2_input.fill(&ctx,0.9).unwrap();
        accumulate(&mut r,&ctx,true).unwrap();
        capture(&r,&ctx,Default::default(),policy).unwrap();
        assert!((r.experimental_output_centers.as_ref().unwrap().0.download(&ctx).unwrap()[0]-0.9).abs()<1e-6);
        r.update_weights_with_lr_multipliers_and_dirty_buckets(&ctx,Default::default(),policy,None).unwrap();
        assert_eq!(r.output_center_batches,0);
    }
    #[test] fn option_update_matches_explicit_centering_and_reuses_buffers() {
        let ctx=Context::new(0).unwrap(); let s=crate::tests::tiny_sfnn_shape();
        let mut weights=crate::tests::tiny_sfnn_weights(s);
        weights.l2fw=None; weights.l2fb=None; weights.l3fw=None; weights.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,weights,4,1).unwrap();
        let mut reference=SfnnTrainStepRunner::new(&ctx,weights,4,1).unwrap();
        let policy=SfnnLayerLrMultipliers{l2_l3_center:true,..Default::default()};
        assert!(r.experimental_output_centers.is_none());
        let mut address=None;
        for step in 1..=18 {
            let c2=0.1+step as f32*0.01; let c3=0.2+step as f32*0.01;
            for runner in [&r,&reference] {
                runner.forward_workspace.l2_input.fill(&ctx,c2).unwrap();
                runner.forward_workspace.l2.fill(&ctx,c3).unwrap();
                runner.backward_workspace.zero_parameter_gradients(&ctx).unwrap();
                runner.backward_workspace.l2w_gradients.fill(&ctx,0.03).unwrap();
                runner.backward_workspace.l2b_gradients.fill(&ctx,0.02).unwrap();
                runner.backward_workspace.l3w_gradients.fill(&ctx,-0.02).unwrap();
                runner.backward_workspace.l3b_gradients.fill(&ctx,0.04).unwrap();
            }
            let mut p=RangerUpdateParams::default(); p.radam.step=step;
            r.update_weights_with_lr_multipliers_and_dirty_buckets(&ctx,p,policy,None).unwrap();
            let current=r.experimental_output_centers.as_ref().unwrap().0.as_ptr();
            if let Some(old)=address {assert_eq!(old,current);} address=Some(current);
            let centers=Centers{l2:vec![c2;s.l2_in()],l3:vec![c3;s.l2_size],device:false};
            transform(&reference,&ctx,&centers,true).unwrap();
            reference.update_weights_with_lr_multipliers_and_dirty_buckets(&ctx,p,Default::default(),None).unwrap();
            transform(&reference,&ctx,&centers,false).unwrap();
            for (a,b) in [(&r.weights.l2w,&reference.weights.l2w),(&r.weights.l2b,&reference.weights.l2b),
                (&r.weights.l3w,&reference.weights.l3w),(&r.weights.l3b,&reference.weights.l3b),
                (&r.optimizer_states.l2w.momentum,&reference.optimizer_states.l2w.momentum),
                (&r.optimizer_states.l3b.slow_params,&reference.optimizer_states.l3b.slow_params)] {
                for (a,b) in a.download(&ctx).unwrap().iter().zip(b.download(&ctx).unwrap()) {assert!((a-b).abs()<2e-6);}
            }
        }
        let p=RangerUpdateParams::default();
        assert!(capture(&r,&ctx,p,Default::default()).unwrap().is_none());
        assert!(capture(&r,&ctx,p,SfnnLayerLrMultipliers{tatara_weight_clip:true,..policy}).is_err());
        let mut p=p; p.radam.gradient_factor=0.25;
        assert!(capture(&r,&ctx,p,policy).is_ok());
    }
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
