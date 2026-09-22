//! Explicit diagnostic, NOT a normal optimizer option. bpu=1 runs only.
//! Remove the component of each hidden L1 weight UPDATE parallel to this
//! batch's global mean input. Bias, skip row, gradients and moments are untouched.
//! This changes Ranger's algorithm. Lookahead synchronization is preserved.
use super::*;

struct Group { fast: Vec<f32>, slow: Vec<f32> }
pub(super) struct Snapshot { c: Vec<f32>, residual: Group, shared: Option<Group> }

fn capture_group(ctx:&Context,w:&F32Buffer,s:&RangerParamState)->Result<Group> {
    Ok(Group{fast:w.download(ctx)?,slow:s.slow_params.download(ctx)?})
}

pub(super) fn capture(r:&SfnnTrainStepRunner,ctx:&Context)->Result<Option<Snapshot>> {
    match std::env::var("BULLETOU_EXPERIMENT_L1_PROJECT_UPDATE") {
        Err(std::env::VarError::NotPresent)=>return Ok(None),
        Ok(s) if s=="1"=>{},
        _=>return Err(CudaCppError::message("L1 update projection experiment expects 1")),
    }
    if r.shape.has_compact_l1() || r.factorizer.any_axis() || r.residual_count_gates_enabled
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_CENTER").is_some() {
        return Err(CudaCppError::message("L1 projection requires dense none/shared, no count gates/centering"));
    }
    static ANNOUNCE:std::sync::Once=std::sync::Once::new();
    ANNOUNCE.call_once(||eprintln!("  EXPERIMENT L1 update projection: global batch mean, hidden rows only, bias/skip/moments unchanged, Lookahead sync preserved; bpu=1 diagnostic"));
    Ok(Some(Snapshot{
        c:experimental_l1_center::batch_mean(r,ctx)?,
        residual:capture_group(ctx,&r.weights.l1w,&r.optimizer_states.l1w)?,
        shared:if r.factorizer.shared {Some(capture_group(ctx,r.weights.l1fw.as_ref().unwrap(),r.optimizer_states.l1fw.as_ref().unwrap())?)}else{None},
    }))
}

fn project(old:&[f32],new:&mut [f32],c:&[f32],rows:usize,hidden:usize,stride:usize,col_major:bool) {
    let norm:f64=c.iter().map(|&v|(v as f64).powi(2)).sum();
    if norm<1e-20 {return;}
    let index=|row:usize,j:usize| if col_major {j*rows+row}else{row*c.len()+j};
    for row in 0..rows {
        if row%stride>=hidden {continue;} // Never alter linear skip's update.
        let dot:f64=c.iter().enumerate().map(|(j,&v)|{
            let i=index(row,j); (new[i] as f64-old[i] as f64)*v as f64
        }).sum();
        for (j,&v) in c.iter().enumerate() {let i=index(row,j);new[i]=(new[i] as f64-dot/norm*v as f64) as f32;}
    }
}

fn apply_group(ctx:&Context,w:&F32Buffer,s:&RangerParamState,old:&Group,c:&[f32],hidden:usize,stride:usize,col_major:bool)->Result<()> {
    let mut fast=w.download(ctx)?;let mut slow=s.slow_params.download(ctx)?;
    let synchronized=fast==slow;
    let rows=fast.len()/c.len();
    project(&old.fast,&mut fast,c,rows,hidden,stride,col_major);
    if synchronized {slow.clone_from(&fast);} else {project(&old.slow,&mut slow,c,rows,hidden,stride,col_major);}
    w.upload(ctx,&fast)?;s.slow_params.upload(ctx,&slow)
}
pub(super) fn apply(r:&SfnnTrainStepRunner,ctx:&Context,s:&Snapshot)->Result<()> {
    apply_group(ctx,&r.weights.l1w,&r.optimizer_states.l1w,&s.residual,&s.c,r.shape.l1_hidden,r.shape.l1_out(),false)?;
    if let Some(old)=&s.shared {apply_group(ctx,r.weights.l1fw.as_ref().unwrap(),r.optimizer_states.l1fw.as_ref().unwrap(),old,&s.c,r.shape.l1_hidden,r.shape.l1_out(),true)?;}
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn projection_preserves_mean_and_skip_for_both_layouts() {
        for col_major in [false,true] {
            let (rows,cols,stride,hidden)=(6,4,3,2);let c=[0.1,0.4,0.7,0.2];
            let old:Vec<f32>=(0..24).map(|i|(i as f32-12.0)*0.03).collect();
            let raw:Vec<f32>=old.iter().enumerate().map(|(i,&v)|v+(i as f32-4.0)*0.002).collect();
            let mut new=raw.clone();project(&old,&mut new,&c,rows,hidden,stride,col_major);
            let index=|r:usize,j:usize|if col_major {j*rows+r}else{r*cols+j};
            for r in 0..rows {
                if r%stride>=hidden {for j in 0..cols {assert_eq!(new[index(r,j)],raw[index(r,j)]);}continue;}
                let drift:f32=(0..cols).map(|j|(new[index(r,j)]-old[index(r,j)])*c[j]).sum();
                assert!(drift.abs()<1e-7);
                // Removed delta is parallel to c, not a full zeroing of updates.
                let ratio=(raw[index(r,0)]-new[index(r,0)])/c[0];
                for j in 1..cols {assert!(((raw[index(r,j)]-new[index(r,j)])/c[j]-ratio).abs()<1e-6);}
            }
        }
    }
    #[test] fn gpu_projection_fast_slow_and_moments() {
        let ctx=Context::new(0).unwrap();let r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape()),4,1).unwrap();
        for (w,state,col_major) in [(&r.weights.l1w,&r.optimizer_states.l1w,false),(r.weights.l1fw.as_ref().unwrap(),r.optimizer_states.l1fw.as_ref().unwrap(),true)] {
            let old=capture_group(&ctx,w,state).unwrap();let c=[0.1,0.4,0.7,0.2];
            let old=Group{fast:old.fast,slow:old.slow.iter().map(|v|v+0.4).collect()};
            let raw:Vec<f32>=old.fast.iter().enumerate().map(|(i,&v)|v+0.01*(i+1) as f32).collect();
            w.upload(&ctx,&raw).unwrap();state.slow_params.upload(&ctx,&raw).unwrap();state.momentum.fill(&ctx,0.3).unwrap();
            apply_group(&ctx,w,state,&old,&c,2,3,col_major).unwrap();
            let mut expected=raw;project(&old.fast,&mut expected,&c,old.fast.len()/4,2,3,col_major);
            assert_eq!(w.download(&ctx).unwrap(),expected);assert_eq!(state.slow_params.download(&ctx).unwrap(),expected);
            assert!(state.momentum.download(&ctx).unwrap().iter().all(|&v|v==0.3));
        }
    }
}
