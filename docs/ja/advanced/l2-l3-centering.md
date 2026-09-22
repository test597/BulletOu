# L2/L3の中心化

`--sfnn-l2-l3-center`（JSON: `"sfnn_l2_l3_center": true`）は、L2/L3のoptimizer更新を入力平均で中心化した座標で行う比較実験用オプションです。デフォルトは無効です。FT/L1を中心化する機能ではありません。

## 指定方法

既存の学習JSONに以下を指定します。他の学習条件はそのままです。

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

CLIでは `--sfnn-l2-l3-center --batches-per-update 1 --optimizer-weight-clip 0` を指定します。他の対応条件も満たす必要があります。既存の設定ファイルを自動で変更することはありません。

比較用の共通JSONを上記の対応条件にした上で、次のgrid指定が使えます。

```powershell
python .\grid_search.py `
  --settings-file <共通設定.json> `
  --output-folder <比較結果フォルダ> `
  --grid sfnn-l2-l3-center false true
```

`--grid sfnn-l2-l3-center true` ならONだけを試します。ハイフン表記と `sfnn_l2_l3_center` の両方を指定できます。`grid_summary.csv`にもON/OFFが記録されます。epoch指定の `{"epoch1": false, "epoch2": true}` も使用できます。

## Glorot初期化との組み合わせ比較

`--sfnn-init-l2-l3-glorot`（JSON: `"sfnn_init_l2_l3_glorot": true`、デフォルトfalse）は、**新規学習のL2/L3重みだけ**をGlorot uniformに変更します。FT・L1・各bias・乱数seedは変更しません。中心化とは独立したオプションです。

- L2: `U(-a, a)`、`a = sqrt(6 / (2*H1 + H2))`。
- L3: `U(-a, a)`、`a = sqrt(6 / (H2 + 1))`。
- OFF時の基本半幅は両層とも従来どおり0.01。
- ON/OFFとも、半幅に既存の `nnue_pytorch_init_scale` と層別初期化scaleを掛けます。層別scaleは `sfnn_init_l2_scale` / `sfnn_init_l3_scale`、未指定なら `sfnn_init_l2_l3_scale`（デフォルト1.0）です。標準Glorot幅にするにはこれらを1.0にしてください。

1024/7/64なら倍率1でL2が約±0.2773501、L3が約±0.3038218です。bucket数はfan-in/outに含めません。適用時にはstdoutへ方式と実際の半幅を表示します。Conductor全体の初期化を再現するものではありません。

4通りの比較例（共通JSONは上記の中心化対応条件にしてください）：

```powershell
python .\grid_search.py `
  --settings-file <新規学習用の共通設定.json> `
  --output-folder <新しい比較結果フォルダ> `
  --grid sfnn-init-l2-l3-glorot false true `
  --grid sfnn-l2-l3-center false true
```

`grid_summary.csv`には両オプションが出ます。初期化はscratch時だけなので、`initial_state`等のcheckpoint指定は外して比較してください。既存checkpointを読み込むと重みを再初期化しません。途中epochからGlorotに切り替える指定は非対応です。再開時は元の初期化設定を維持してください。

## 中心化の計算

各batchで、L2とL3それぞれの入力平均ベクトル `c` をGPU上で計算します。bucket別平均ではなく、batch全体の平均です。更新直前に `beta = b + W*c`、`gW_center = gW - gb*c` とし、この座標で既存のoptimizer更新を行います。更新後に `b = beta - W*c` に戻します。Lookaheadのslow weight/biasも同様に変換します。

forward・validation・nn.binは従来どおり `W*x+b` です。BatchNormや入力の分散正規化ではありません。中心化した勾配に対してmomentum等を更新するため、通常のRangerと同じ学習アルゴリズムではありません。報告されるlossに追加の罰則は加えません。

平均計算用GPU領域は有効時に確保して再利用します。1024/7/64構成では約10KiBで、重み全体をCPUへ転送する処理はありません。

## 現在の対応範囲と再開

- cuda-cppのSFNN、`batches_per_update=1`、`sfnn_update_scope=all`。
- L1 factorizerはnone/shared。FT factorizerとL1 QATは使用可能。
- weight clipは明示的に0。weight decay・Norm loss・saturation penalty・factorizer residual decayは0。
- bucket counts/count gates、axis/pair factorizerとの併用は現在非対応。
- L2入力幅・出力幅は各256以下、`bucket数 × L2幅` は65536以下。
- 非対応の組み合わせはエラーにします。clip等を暗黙に無効化しません。

checkpointとnn.binの形式は変更しません。checkpoint保存時には通常のweight/bias表現に戻っています。ON/OFFを変えて再開できますが、optimizerのmomentum等は引き継ぎ、自動リセットしません。これは学習条件変更なので、新規学習でのA/B比較と同じではありません。再開時もJSONまたはCLIでON/OFFを明示してください。

旧調査用の `BULLETOU_EXPERIMENT_*` 環境変数を設定したシェルでは使用せず、通常の環境からこのオプションを指定してください。検証では飽和率・検証精度の改善を確認しましたが、長期学習の棋力改善を保証するものではありません。
