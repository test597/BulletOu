# Training with BCE with Logits

[日本語](../../ja/advanced/bce-with-logits.md)

Supported by the cuda-cpp trainer, worker, validation, and GPU/CPU quantized validation. The default remains the existing squared probability-space WRM loss.

## Configuration

Add these fields to your training JSON, preserving the other training settings:

```json
{
  "loss_bce_with_logits": true,
  "wrm_in_offset": 0
}
```

The CLI equivalent is `--loss-bce-with-logits --wrm-in-offset 0`.
Remove or disable `win_rate_model` and `loss_sigmoid_mse`; explicit selection of either conflicts with BCE.

Standard BCE logits do not use a prediction-side WRM offset. A nonzero `wrm_in_offset` is rejected rather than silently ignored (the ordinary WRM default is270, so specify0). `loss_pow_exp` is unused for BCE and must remain at its default2.

## Formula

Let `y` be the train-scale network output, `s` the teacher score, and `σ(x)=1/(1+exp(-x))`.

```text
z = y * wrm_nnue2score / wrm_in_scaling
p = σ(z)
F(s) = (1 + σ((s - wrm_target_offset)/wrm_target_scaling)
          - σ((-s - wrm_target_offset)/wrm_target_scaling)) / 2
t_score = ε + (1 - 2ε) * F(s)       ε = wrm_target_epsilon
t = lambda * t_score + (1-lambda) * game_result
L = -t*log(p) - (1-t)*log(1-p)
  = max(z,0) - t*z + log(1+exp(-abs(z)))
dL/dy = (p-t) * wrm_nnue2score / wrm_in_scaling
```

`game_result` is1 for wins,0.5 for draws, and0 for losses. Teacher offset, scaling, epsilon, and lambda retain their existing meanings. The implementation evaluates a stable logit-space expression, avoiding log(0) for large outputs. Training applies the existing per-position weight and batch mean.

Loss/qloss now report BCE, excluding Norm loss and other regularizers. Accuracy definitions do not change. GPU approximate versus CPU integer qvalid inference remains unchanged. Pass the same BCE and WRM options to `quantized-test`. Train-scale qloss uses raw/8128 without FV_SCALE; CPU mode's separately reported engine-scale loss uses raw/FV_SCALE with nnue2score=1. GPU mode retains the existing behavior: only train-scale loss is calculated, and the engine-scale field repeats that value.

**BCE values are not directly comparable with squared-error loss values.** BCE does not automatically prevent saturation: targets0/1 have no finite optimal logit. Consider a separate teacher-epsilon experiment if needed. Gradient magnitudes differ, so the same learning rate is not necessarily appropriate.

## Grid search

Set `wrm_in_offset: 0`, `win_rate_model: false`, and `loss_sigmoid_mse: false` in the common JSON:

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce `
  --grid loss_bce_with_logits false true
```

False selects the existing WRM squared error; true selects BCE. `grid_summary.csv` records `loss_bce_with_logits`. Do not rank different loss definitions by their numerical loss values. For continued-training comparisons, start each condition from the same checkpoint and teacher position.
