# L2/L3 optimizer centering

`--sfnn-l2-l3-center` (JSON: `"sfnn_l2_l3_center": true`) enables input-mean-centered optimizer coordinates for L2/L3. It is **off by default** and does not center FT/L1.

## Settings

Use these fields in your existing training JSON, keeping other training conditions unchanged:

```json
{
  "sfnn_l2_l3_center": true,
  "batches_per_update": 1,
  "sfnn_factorizer": "shared",
  "optimizer_weight_clip": 0,
  "optimizer_weight_decay": 0,
  "sfnn_norm_loss_strength": 0,
  "sfnn_saturation_penalty": 0,
  "sfnn_factorizer_residual_decay": 0
}
```

CLI: `--sfnn-l2-l3-center --batches-per-update 1 --optimizer-weight-clip 0`, also satisfying all constraints below. No user settings are automatically rewritten.

With a compatible common JSON, compare using:

```powershell
python .\grid_search.py `
  --settings-file <common-settings.json> `
  --output-folder <comparison-folder> `
  --grid sfnn-l2-l3-center false true
```

Use `--grid sfnn-l2-l3-center true` to run only the enabled condition. Both hyphenated and underscore key names are supported. The flag is recorded in `grid_summary.csv`. Epoch schedules such as `{"epoch1": false, "epoch2": true}` are supported.

## Comparing Glorot initialization with centering

`--sfnn-init-l2-l3-glorot` (JSON: `"sfnn_init_l2_l3_glorot": true`, default false) changes **scratch L2/L3 weights only** to Glorot uniform. FT, L1, all biases and random seeds are unchanged. This option is independent of centering.

- L2: `U(-a,a)`, where `a = sqrt(6 / (2*H1 + H2))`.
- L3: `U(-a,a)`, where `a = sqrt(6 / (H2 + 1))`.
- With the flag off, both base half-widths remain 0.01.
- In either mode, multiply the half-width by `nnue_pytorch_init_scale` and the layer's initialization scale (`sfnn_init_l2_scale` / `sfnn_init_l3_scale`, falling back to `sfnn_init_l2_l3_scale`, default 1.0). Set all applicable multipliers to 1.0 for standard Glorot widths.

For 1024/7/64 with unit multipliers, widths are approximately +/-0.2773501 (L2) and +/-0.3038218 (L3). Bucket count is not part of fan-in/out. Scratch initialization prints the chosen method and actual widths. This does not reproduce Conductor's entire initialization.

Four-way comparison (the common JSON must satisfy centering constraints):

```powershell
python .\grid_search.py `
  --settings-file <scratch-common-settings.json> `
  --output-folder <new-comparison-folder> `
  --grid sfnn-init-l2-l3-glorot false true `
  --grid sfnn-l2-l3-center false true
```

Both flags appear in `grid_summary.csv`. Remove checkpoint inputs such as `initial_state` to compare initializations. Loading a checkpoint never reinitializes its weights. Epoch-wise switching of this initialization option is unsupported; retain the original initialization setting on resume.

## Centered update equations

Compute each layer's batch-global input mean `c` on GPU (not a separate mean per bucket). Before updating, transform `beta = b + W*c` and `gW_center = gW - gb*c`. Run the optimizer in these coordinates, then restore `b = beta - W*c`. Lookahead slow weights/biases undergo the same coordinate conversion.

Forward, validation and nn.bin still use `W*x+b`. This is not BatchNorm or variance normalization. Moments are updated using centered gradients, so this is a different optimization algorithm from ordinary Ranger. It adds no penalty to the reported task loss.

GPU mean scratch is allocated on activation and reused (about 10 KiB for 1024/7/64). It does not copy entire weight tensors to the CPU.

## Supported combinations and resume

- cuda-cpp SFNN, `batches_per_update=1`, `sfnn_update_scope=all`.
- L1 factorizer none/shared. FT factorization and L1 QAT are supported.
- Explicit `optimizer_weight_clip=0`; zero weight decay, Norm loss, saturation penalty and factorizer residual decay.
- No bucket counts/count gates or axis/pair factors yet.
- L2 input/output widths each <=256; `buckets * L2 width <=65536`.
- Unsupported combinations fail explicitly; clipping is never silently disabled.

Checkpoint and nn.bin formats are unchanged; weights/biases are folded back before saving. You may switch the flag on resume, but optimizer moments are retained, not automatically reset. This changes training conditions and is not equivalent to an A/B comparison from scratch. Specify the desired flag in the resumed CLI/JSON.

Use a normal shell without the older `BULLETOU_EXPERIMENT_*` research environment variables. Saturation and validation improvements were observed in bounded experiments; long-term playing-strength improvement is not guaranteed.
