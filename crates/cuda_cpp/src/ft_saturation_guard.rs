//! Linear pre-clamp hinge gradient, gated by consecutive high-saturation training batches.
use super::*;

#[derive(Debug)]
pub(super) struct State {
    counts: I32Buffer,
    streaks: I32Buffer,
    config: (f32, f32, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ft_guard_validation() {
        let p = SfnnLayerLrMultipliers::default();
        assert_eq!(p.ft_saturation_penalty, 0.0);
        assert!(p.validate().is_ok());
        for x in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(SfnnLayerLrMultipliers { ft_saturation_penalty: x, ..p }.validate().is_err());
        }
        for x in [0.0, -0.1, 1.1, f32::NAN] {
            assert!(SfnnLayerLrMultipliers { ft_saturation_rate: x, ..p }.validate().is_err());
        }
        assert!(SfnnLayerLrMultipliers { ft_saturation_patience: 0, ..p }.validate().is_err());
    }

    #[test]
    #[ignore = "requires CUDA; tiny synthetic buffers only"]
    fn ft_guard_gpu_factorizer_chain_rule() {
        let ctx = Context::new(0).unwrap();
        let a = F32Buffer::from_host(&ctx, &[1., 0.]).unwrap();
        let weights = F32Buffer::from_host(&ctx, &[1.]).unwrap();
        let indices = I32Buffer::from_host(&ctx, &[1630]).unwrap();
        let counts = I32Buffer::new(&ctx, 3).unwrap();
        let streaks = I32Buffer::from_host(&ctx, &[0, 0]).unwrap();
        let gw = F32Buffer::from_host(&ctx, &vec![0.; 133578 * 2]).unwrap();
        let gb = F32Buffer::from_host(&ctx, &[0.; 2]).unwrap();
        check(unsafe {
            ffi::bulletou_ft_saturation_guard(
                ctx.as_ptr(),
                a.as_ptr(),
                a.as_ptr(),
                weights.as_ptr(),
                indices.as_ptr(),
                indices.as_ptr(),
                counts.as_ptr(),
                streaks.as_ptr(),
                gw.as_ptr(),
                gb.as_ptr(),
                1,
                2,
                1,
                133578,
                0.25,
                4.,
                1.,
                1,
            )
        })
        .unwrap();
        let g = gw.download(&ctx).unwrap();
        assert_eq!(g[1630 * 2], 2.0);
        assert_eq!(g[(131949 + 1) * 2], 0.5);
        assert_eq!(g.iter().sum::<f32>(), 2.5);
        assert_eq!(gb.download(&ctx).unwrap(), vec![2., 0.]);
    }

    #[test]
    #[ignore = "requires CUDA; tiny synthetic buffers only"]
    fn ft_guard_gpu_hinge_streak_weights_and_reset() {
        let ctx = Context::new(0).unwrap();
        // Unit 0: all eligible views saturated; unit 1: half saturated.
        let a = F32Buffer::from_host(&ctx, &[1., 1., 1., 0., 0., 0.]).unwrap();
        let b = F32Buffer::from_host(&ctx, &[1., 0., 1., 1., 0., 0.]).unwrap();
        let weights = F32Buffer::from_host(&ctx, &[1., 0.5, 0.]).unwrap();
        let ai = I32Buffer::from_host(&ctx, &[0, -1, 1, -1, 0, -1]).unwrap();
        let bi = I32Buffer::from_host(&ctx, &[1, -1, 0, -1, 0, -1]).unwrap();
        let counts = I32Buffer::new(&ctx, 3).unwrap();
        let streaks = I32Buffer::from_host(&ctx, &[0, 0]).unwrap();
        let gw = F32Buffer::from_host(&ctx, &[0.; 4]).unwrap();
        let gb = F32Buffer::from_host(&ctx, &[0.; 2]).unwrap();
        let apply = |strength, rate, patience| {
            check(unsafe {
                ffi::bulletou_ft_saturation_guard(
                    ctx.as_ptr(),
                    a.as_ptr(),
                    b.as_ptr(),
                    weights.as_ptr(),
                    ai.as_ptr(),
                    bi.as_ptr(),
                    counts.as_ptr(),
                    streaks.as_ptr(),
                    gw.as_ptr(),
                    gb.as_ptr(),
                    3,
                    2,
                    2,
                    2,
                    1.0,
                    strength,
                    rate,
                    patience,
                )
            })
            .unwrap();
        };
        apply(12., 0.99, 2);
        assert_eq!(counts.download(&ctx).unwrap(), vec![4, 2, 4]);
        assert_eq!(streaks.download(&ctx).unwrap(), vec![1, 0]);
        assert_eq!(gb.download(&ctx).unwrap(), vec![0., 0.]);
        apply(12., 0.99, 2);
        assert_eq!(streaks.download(&ctx).unwrap(), vec![2, 0]);
        assert_eq!(gb.download(&ctx).unwrap(), vec![3., 0.]);
        assert_eq!(gw.download(&ctx).unwrap(), vec![1.5, 0., 1.5, 0.]);
        // Relax the rate: unit 1 becomes eligible on its second consecutive batch.
        apply(12., 0.5, 2);
        apply(12., 0.5, 2);
        assert_eq!(gb.download(&ctx).unwrap(), vec![9., 1.5]);
        apply(0., 0.99, 2);
        assert_eq!(streaks.download(&ctx).unwrap(), vec![2, 0]);
        assert_eq!(gb.download(&ctx).unwrap(), vec![9., 1.5]);
        weights.upload(&ctx, &[0., 0., 0.]).unwrap();
        apply(12., 0.99, 2);
        assert_eq!(streaks.download(&ctx).unwrap(), vec![0, 0]);
        assert_eq!(gb.download(&ctx).unwrap(), vec![9., 1.5]);
    }
}

pub(super) fn apply(
    r: &mut SfnnTrainStepRunner,
    ctx: &Context,
    p: SfnnLayerLrMultipliers,
    slot: Option<usize>,
) -> Result<()> {
    if p.ft_saturation_penalty == 0.0 || p.l0 == 0.0 {
        r.ft_saturation_guard = None;
        return Ok(());
    }
    p.validate()?;
    let config = (p.ft_saturation_penalty, p.ft_saturation_rate, p.ft_saturation_patience);
    if r.ft_saturation_guard.as_ref().is_none_or(|s| s.config != config) {
        let streaks = I32Buffer::from_host(ctx, &vec![0; r.shape.ft_size])?;
        r.ft_saturation_guard = Some(State { counts: I32Buffer::new(ctx, r.shape.ft_size + 1)?, streaks, config });
    }
    let state = r.ft_saturation_guard.as_ref().unwrap();
    let (batch, weights) = match slot {
        Some(i) => (&r.upload_slots[i].device_batch, &r.upload_slots[i].entry_weights),
        None => (&r.device_batch, &r.entry_weights),
    };
    check(unsafe {
        ffi::bulletou_ft_saturation_guard(
            ctx.as_ptr(),
            r.forward_workspace.stm_l0.as_ptr(),
            r.forward_workspace.nstm_l0.as_ptr(),
            weights.as_ptr(),
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            state.counts.as_ptr(),
            state.streaks.as_ptr(),
            r.backward_workspace.l0w_gradients.as_ptr(),
            r.backward_workspace.l0b_gradients.as_ptr(),
            r.batch_size,
            r.shape.ft_size,
            r.max_active,
            r.shape.input_size,
            r.factorizer_alpha.ft,
            p.ft_saturation_penalty,
            p.ft_saturation_rate,
            p.ft_saturation_patience as i32,
        )
    })
}
