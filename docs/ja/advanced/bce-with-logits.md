# BCE with Logitsで学習する

[English](../../en/advanced/bce-with-logits.md)

通常の `bulletou.exe`（cuda-cpp）、worker、検証、GPU/CPUの量子化後検証で使用できます。デフォルトは従来のWRM確率空間の二乗誤差のままです。

## 指定方法

学習用JSONに次を追加します。既存の他の学習条件はそのまま使えます。

```json
{
  "loss_bce_with_logits": true,
  "wrm_in_offset": 0
}
```

CLIでは `--loss-bce-with-logits --wrm-in-offset 0` です。
`--win-rate-model` と `--loss-sigmoid-mse` は同時指定できません。JSONでは削除するかfalseにしてください。

予測側のoffsetは通常のBCE logitには適用しません。黙って無視せず、非0ならエラーにします（通常のWRMの既定offsetは270なので、0の指定が必要です）。`loss_pow_exp` はBCEでは使用せず、既定値2以外ならエラーです。

## 計算式

学習スケールのネットワーク出力を `y`、教師評価値を `s` とします。sigmoidは `σ(x)=1/(1+exp(-x))` です。

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

`game_result` は勝ち1、引き分け0.5、負け0です。教師側のoffset、scaling、epsilon、lambdaは従来どおり有効です。実装はlogitから安定な式で直接計算するため、大きい出力でも `log(0)` を計算しません。学習時は既存の局面weightを掛け、batch内で平均します。

標準のloss/qlossはこのBCEになり、Norm loss等の正則化項は含みません。acc/qaccの定義は変更しません。qvalidのGPU近似/CPU整数推論という違いも従来どおりです。`quantized-test` にも同じBCEとWRMパラメーターを指定してください。train-scale qlossではraw/8128に上の式を適用し、FV_SCALEは使いません。CPUモードの別表示engine-scale lossはraw/FV_SCALEを入力とし、nnue2score=1として計算します。GPUモードは従来どおりtrain-scaleのみを計算し、engine-scale欄にも同じ値を表示します。

**BCEのloss値と二乗誤差のloss値は直接比較できません。** BCEは飽和を自動的に抑制する機能ではありません。特に教師確率0/1では有限の最適logitがないため、必要なら教師側のepsilonなどを別途比較してください。損失の勾配スケールが変わるので、同じLRが最適とは限りません。

## Grid search

共通JSONを `wrm_in_offset: 0`、`win_rate_model: false`、`loss_sigmoid_mse: false` として、次で二乗誤差とBCEを比較できます。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce `
  --grid loss_bce_with_logits false true
```

falseは従来のWRM二乗誤差、trueはBCEです。`grid_summary.csv` に `loss_bce_with_logits` が記録されます。lossの定義が変わるため、値の大小で方式間の優劣を決めないでください。既存モデルから試す場合も、条件ごとに同じcheckpoint/教師位置から始めてください。
