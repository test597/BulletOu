//! Linear pre-clamp hinge gradient, gated by consecutive high-saturation training batches.
use super::*;

#[derive(Debug)]
pub(super) struct State {
    counts: I32Buffer,
    streaks: I32Buffer,
    config: (f32, f32, usize),
    audit_step: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; compares tiled gradient with scalar CPU reference"]
    fn ft_guard_tiled_gradient_matches_cpu_with_tails_and_repeated_features() {
        let ctx = Context::new(0).unwrap();
        let (batch, ft, active, inputs) = (2059usize, 35usize, 3usize, 7usize);
        let weights: Vec<f32> = (0..batch)
            .map(|i| match i % 5 {
                0 => 0.0,
                1 => -1.0,
                2 => 0.5,
                _ => 1.0,
            })
            .collect();
        let a: Vec<f32> = (0..batch * ft).map(|j| if j % 7 < 4 { 1.0 } else { 0.5 }).collect();
        let b: Vec<f32> = (0..batch * ft).map(|j| if j % 11 < 3 { 1.0 } else { 0.0 }).collect();
        let ai: Vec<i32> = (0..batch * active).map(|j| if j % 3 == 2 { -1 } else { (j / 3 % inputs) as i32 }).collect();
        let bi: Vec<i32> = (0..batch * active)
            .map(|j| if j % 3 == 2 { inputs as i32 } else { ((j / 3 + 2) % inputs) as i32 })
            .collect();
        let mut counts = vec![0i32; ft + 1];
        for i in 0..batch {
            if weights[i] <= 0.0 {
                continue;
            }
            counts[ft] += 2;
            for u in 0..ft {
                counts[u] += (a[i * ft + u] >= 1.0) as i32 + (b[i * ft + u] >= 1.0) as i32;
            }
        }
        let rate = 0.4f32;
        let enabled: Vec<bool> = (0..ft).map(|u| counts[u] as f32 / counts[ft] as f32 >= rate).collect();
        assert!(enabled.iter().any(|&x| x) && enabled.iter().any(|&x| !x));
        let mut expected_w = vec![0.25f32; inputs * ft];
        let mut expected_b = vec![0.25f32; ft];
        for i in 0..batch {
            if weights[i] <= 0.0 {
                continue;
            }
            let delta = 2.0 * weights[i] / (2.0 * batch as f32 * ft as f32);
            for (activation, indices) in [(&a, &ai), (&b, &bi)] {
                for u in 0..ft {
                    if !enabled[u] || activation[i * ft + u] < 1.0 {
                        continue;
                    }
                    expected_b[u] += delta;
                    for &idx in &indices[i * active..(i + 1) * active] {
                        if idx >= 0 && (idx as usize) < inputs {
                            expected_w[idx as usize * ft + u] += delta;
                        }
                    }
                }
            }
        }
        let a = F32Buffer::from_host(&ctx, &a).unwrap();
        let b = F32Buffer::from_host(&ctx, &b).unwrap();
        let weights = F32Buffer::from_host(&ctx, &weights).unwrap();
        let ai = I32Buffer::from_host(&ctx, &ai).unwrap();
        let bi = I32Buffer::from_host(&ctx, &bi).unwrap();
        let gpu_counts = I32Buffer::new(&ctx, ft + 1).unwrap();
        let streaks = I32Buffer::from_host(&ctx, &vec![0; ft]).unwrap();
        let gw = F32Buffer::from_host(&ctx, &vec![0.25; inputs * ft]).unwrap();
        let gb = F32Buffer::from_host(&ctx, &vec![0.25; ft]).unwrap();
        check(unsafe {
            ffi::bulletou_ft_saturation_guard(
                ctx.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                weights.as_ptr(),
                ai.as_ptr(),
                bi.as_ptr(),
                gpu_counts.as_ptr(),
                streaks.as_ptr(),
                gw.as_ptr(),
                gb.as_ptr(),
                batch,
                ft,
                active,
                inputs,
                1.0,
                2.0,
                rate,
                1,
            )
        })
        .unwrap();
        assert_eq!(gpu_counts.download(&ctx).unwrap(), counts);
        assert_eq!(streaks.download(&ctx).unwrap(), enabled.iter().map(|&x| x as i32).collect::<Vec<_>>());
        for (got, want) in
            gw.download(&ctx).unwrap().iter().zip(&expected_w).chain(gb.download(&ctx).unwrap().iter().zip(&expected_b))
        {
            assert!((got - want).abs() < 2e-5, "got={got} expected={want}");
        }
    }

    #[test]
    fn ft_guard_validation() {
        let p = SfnnLayerLrMultipliers::default();
        assert_eq!(p.ft_saturation_penalty, 0.0);
        assert_eq!(p.ft_saturation_rate, 0.2);
        assert_eq!(p.ft_saturation_patience, 1);
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
    fn ft_guard_default_includes_twenty_percent_in_first_batch() {
        let ctx = Context::new(0).unwrap();
        let a = F32Buffer::from_host(&ctx, &[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        let weights = F32Buffer::from_host(&ctx, &[1.; 5]).unwrap();
        let indices = I32Buffer::from_host(&ctx, &[0; 5]).unwrap();
        let counts = I32Buffer::new(&ctx, 3).unwrap();
        let streaks = I32Buffer::from_host(&ctx, &[0, 0]).unwrap();
        let gw = F32Buffer::from_host(&ctx, &[0.; 2]).unwrap();
        let gb = F32Buffer::from_host(&ctx, &[0.; 2]).unwrap();
        let p = SfnnLayerLrMultipliers::default();
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
                5,
                2,
                1,
                1,
                1.,
                20.,
                p.ft_saturation_rate,
                p.ft_saturation_patience as i32,
            )
        })
        .unwrap();
        assert_eq!(counts.download(&ctx).unwrap(), vec![2, 0, 10]);
        assert_eq!(streaks.download(&ctx).unwrap(), vec![1, 0]);
        assert_eq!(gb.download(&ctx).unwrap(), vec![2., 0.]);
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

pub(super) struct Binding<'a> {
    ctx: &'a Context,
    reference: bool,
}

impl Drop for Binding<'_> {
    fn drop(&mut self) {
        // Clear borrowed device pointers even when backward returns an error.
        unsafe {
            ffi::bulletou_bind_ft_saturation_guard(
                self.ctx.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                0,
                0.0,
                0.0,
                1,
            );
        }
    }
}

impl Binding<'_> {
    pub(super) fn finish(
        self,
        r: &mut SfnnTrainStepRunner,
        p: SfnnLayerLrMultipliers,
        slot: Option<usize>,
    ) -> Result<()> {
        if self.reference {
            apply(r, self.ctx, p, slot)?;
        }
        Ok(())
    }
}

pub(super) fn prepare<'a>(
    r: &mut SfnnTrainStepRunner,
    ctx: &'a Context,
    p: SfnnLayerLrMultipliers,
    slot: Option<usize>,
) -> Result<Binding<'a>> {
    // The opt-in audit deliberately retains the separate reference pass so it
    // can measure before/after gradients. Normal training always uses fusion.
    let reference = std::env::var("BULLETOU_FT_GUARD_AUDIT").as_deref() == Ok("1");
    let binding = Binding { ctx, reference };
    check(unsafe {
        ffi::bulletou_bind_ft_saturation_guard(
            ctx.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            0,
            0.0,
            0.0,
            1,
        )
    })?;
    if p.ft_saturation_penalty == 0.0 || p.l0 == 0.0 {
        r.ft_saturation_guard = None;
        return Ok(binding);
    }
    if reference {
        return Ok(binding);
    }
    p.validate()?;
    let config = (p.ft_saturation_penalty, p.ft_saturation_rate, p.ft_saturation_patience);
    if r.ft_saturation_guard.as_ref().is_none_or(|s| s.config != config) {
        r.ft_saturation_guard = Some(State {
            counts: I32Buffer::new(ctx, r.shape.ft_size + 1)?,
            streaks: I32Buffer::from_host(ctx, &vec![0; r.shape.ft_size])?,
            config,
            audit_step: 0,
        });
    }
    let state = r.ft_saturation_guard.as_ref().unwrap();
    let weights = match slot {
        Some(i) => &r.upload_slots[i].entry_weights,
        None => &r.entry_weights,
    };
    check(unsafe {
        ffi::bulletou_bind_ft_saturation_guard(
            ctx.as_ptr(),
            weights.as_ptr(),
            state.counts.as_ptr(),
            state.streaks.as_ptr(),
            r.batch_size,
            r.shape.ft_size,
            p.ft_saturation_penalty,
            p.ft_saturation_rate,
            p.ft_saturation_patience as i32,
        )
    })?;
    Ok(binding)
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
        r.ft_saturation_guard =
            Some(State { counts: I32Buffer::new(ctx, r.shape.ft_size + 1)?, streaks, config, audit_step: 0 });
    }
    let s = r.ft_saturation_guard.as_mut().unwrap();
    s.audit_step += 1;
    let step = s.audit_step;
    let audit =
        std::env::var("BULLETOU_FT_GUARD_AUDIT").as_deref() == Ok("1") && [1, 10, 100, 610, 1220].contains(&step);
    let before = if audit {
        Some((r.backward_workspace.l0b_gradients.download(ctx)?, r.backward_workspace.l0w_gradients.download(ctx)?))
    } else {
        None
    };
    let events = if audit { Some((Event::new(ctx)?, Event::new(ctx)?)) } else { None };
    let state = r.ft_saturation_guard.as_ref().unwrap();
    let (batch, weights) = match slot {
        Some(i) => (&r.upload_slots[i].device_batch, &r.upload_slots[i].entry_weights),
        None => (&r.device_batch, &r.entry_weights),
    };
    if let Some((start, _)) = &events {
        start.record(ctx)?;
    }
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
    })?;
    if let Some((start, end)) = &events {
        end.record(ctx)?;
        end.synchronize()?;
        let ms = end.elapsed_ms_since(start)?;
        let counts = state.counts.download(ctx)?;
        let streaks = state.streaks.download(ctx)?;
        let active = streaks.iter().filter(|&&n| n >= p.ft_saturation_patience as i32).count();
        let after = r.backward_workspace.l0b_gradients.download(ctx)?;
        let gw_after = r.backward_workspace.l0w_gradients.download(ctx)?;
        let (gb_before, gw_before) = before.unwrap();
        let w = r.weights.l0w.download(ctx)?;
        let bias = r.weights.l0b.download(ctx)?;
        let mom = r.optimizer_states.l0b.momentum.download(ctx)?;
        let vel = r.optimizer_states.l0b.velocity.download(ctx)?;
        let ai = batch.stm_indices.download(ctx)?;
        let bi = batch.nstm_indices.download(ctx)?;
        let ew = weights.download(ctx)?;
        let f = r.shape.ft_size;
        let mut order: Vec<usize> = (0..f).collect();
        order.sort_by_key(|&u| std::cmp::Reverse(counts[u]));
        order.truncate(5);
        order.extend([68, 907, 913].into_iter().filter(|&u| u < f));
        order.sort();
        order.dedup();
        eprintln!(
            "  [FT-AUDIT] step={step} guard_ms={ms:.4} active={active}/{f} eligible_views={} strength={} rate={}",
            counts[f], p.ft_saturation_penalty, p.ft_saturation_rate
        );
        for u in order {
            let mut zs = Vec::new();
            for i in 0..r.batch_size.min(4096) {
                if ew[i] <= 0.0 {
                    continue;
                }
                for indices in [&ai, &bi] {
                    let mut z = bias[u];
                    for &idx in &indices[i * r.max_active..(i + 1) * r.max_active] {
                        if idx < 0 || idx as usize >= r.shape.input_size {
                            continue;
                        }
                        let idx = idx as usize;
                        z += w[idx * f + u];
                        if r.shape.input_size == 133578 && idx < 131949 {
                            z += r.factorizer_alpha.ft * w[(131949 + idx % 1629) * f + u];
                        }
                    }
                    zs.push(z);
                }
            }
            zs.sort_by(f32::total_cmp);
            let n = zs.len();
            let mut task = 0f64;
            let mut penalty = 0f64;
            let mut dot = 0f64;
            for row in 0..r.shape.input_size {
                let j = row * f + u;
                let a = gw_before[j] as f64;
                let d = (gw_after[j] - gw_before[j]) as f64;
                task += a * a;
                penalty += d * d;
                dot += a * d;
            }
            eprintln!(
                "  [FT-AUDIT-UNIT] step={step} unit={u} hits={} rate={:.6} streak={} bias={:.7} bias_task={:.9e} bias_penalty={:.9e} momentum={:.9e} velocity={:.9e} weight_task_l2={:.9e} weight_penalty_l2={:.9e} dot={:.9e} z_p50={:.6} z_p99={:.6} z_max={:.6}",
                counts[u],
                counts[u] as f64 / counts[f].max(1) as f64,
                streaks[u],
                bias[u],
                gb_before[u],
                after[u] - gb_before[u],
                mom[u],
                vel[u],
                task.sqrt(),
                penalty.sqrt(),
                dot,
                if n > 0 { zs[n / 2] } else { 0. },
                if n > 0 { zs[(n - 1) * 99 / 100] } else { 0. },
                zs.last().copied().unwrap_or(0.)
            );
        }
    }
    Ok(())
}
