# epochごとの設定変更

SFNNの通常学習・`grid_search.py` の共通設定JSONでは、途中epochから数値・真偽値を切り替えられます。重み・optimizer stateは維持し、プロセス再起動は行いません。

```json
{
  "lr": {"epoch1": 0.000400, "epoch11": 0.000200},
  "lr_min": {"epoch1": 0.000050, "epoch11": 0.000030},
  "batches_per_update": {"epoch1": 1, "epoch11": 4},
  "sfnn_qat_l1": {"epoch1": true, "epoch11": false}
}
```

既存の学習設定に加える項目の例です。epoch 1～10は前半、11以降は後半の値です。`true`→`false`、`false`→`true`のどちらも可能です。

- `epoch1` は必須。キーは `epoch1`, `epoch2`, …（先頭ゼロ不可）。順序は問いません。
- 未指定epochは直前の値を引き継ぎます。従来の数値・真偽値だけの指定は全epoch共通です。
- `lr` / `lr_min` は各epoch内のLRスケジュールの開始値／下限値。stepの自動gammaも再計算します。
- `--resume` は再開先epochの値を使います。別runを `--initial-state` から開始する場合は、新runのepoch 1からです。
- 明示的なCLI値・grid軸の値は、その項目のepoch別設定全体を上書きします。
- 設定は起動時に読み込み、実行中のJSON編集は反映しません。
- epoch開始時に `[epoch settings]` で有効値を表示します。`grid_summary.csv` の条件列も各epochの値です。checkpointの `bulletou-settings.json` は元のスケジュールを保存します。

## 対応項目

| 項目 | 意味 |
|---|---|
| `lr`, `lr_min` | 学習率の開始値・下限値 |
| `batches_per_update` | 1更新あたりの累積batch数 |
| `sfnn_qat_l1` | L1 QATの有効・無効 |
| `sfnn_freeze_l1` | L1の固定・解除 |
| `sfnn_l1_lr_mult` | L1の学習率倍率 |
| `sfnn_norm_loss_strength` | ノルム正則化係数 |
| `sfnn_saturation_penalty`, `sfnn_saturation_threshold` | 飽和penaltyの係数・閾値 |
| `optimizer_weight_clip` | 重みclip幅（0は無効） |
| `optimizer_weight_decay` | weight decay係数 |
| `bce_error_weight_k` | BCEの誤差重み付け係数（BCE有効時のみ） |

未対応項目のオブジェクト指定はエラーです。arch、batch size、教師データ、factorizer構造、loss種類などは途中切替できません。worker/tuning、direct-step smoke、plateauにも未対応です。

## bpu変更時の丸め

保存時に未反映の累積勾配を残さないため、全epoch共通のbatches/sbを、指定された全bpuの最小公倍数で割り切れる数に切り下げます。

40M局面/sb・batch size 65,536・bpu 1→4なら、全epochで608 batches/sb（39,845,888局面）です。bpu=1固定時の610とはわずかに異なります。再開時にbpuの集合を変更した場合も起動時のbatches/sb表示を確認してください。

[Grid search](grid-search.md) / [English](../../en/advanced/epoch-settings.md)
