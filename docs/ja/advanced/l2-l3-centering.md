# L2/L3の中心化

## L1中心化のA/Bテスト

`--sfnn-l1-center`（JSON: `"sfnn_l1_center": true`、デフォルトOFF）で、L1にも同じoptimizer座標の中心化を適用できます。L2/L3中心化とは独立しており、併用も可能です。

```powershell
python .\grid_search.py `
  --settings-file <共通設定.json> `
  --output-folder <新しい比較フォルダ> `
  --grid sfnn-l2-l3-center true `
  --grid sfnn-l1-center false true
```

L1に入力されるFT結合特徴の平均をGPU上で求め、bucket個別重みとshared重みに適用します。bucket別平均ではなく、更新対象の全バッチにわたる共通平均です。BPU>1、L1 QATに対応します。dense L1・factorizer none/sharedに限定し、その他の併用条件は下記と同じです。FT自身の中心化・出力の正規化・飽和率への罰則ではありません。飽和率低下は保証されないため、A/Bで確認してください。

L1の追加GPU作業領域はFT幅1024で約132 KiBです。重み全体のCPU転送はしません。forwardとnn.binは従来のfold済み形式です。epoch指定によるON/OFFとresume時の切替も可能ですが、optimizer状態はリセットしません。

GPU平均計算は列方向だけでなく行方向も並列化します。L1のbucket個別・sharedの座標変換は1カーネルにまとめ、更新前後それぞれの起動回数を削減しています。平均を取る局面の間引きはしません。加算順序の変更による浮動小数点の微差はありますが、中心化の式・学習率・optimizerの定義は変えていません。

中心化処理では、BPU=1の不要な平均値の加算・ゼロ初期化・コピーを省き、BPU>1でも平均の確定を1カーネルにまとめています。速度改善量は環境と構成によるため、実測で比較してください。A/Bでは両条件とも `optimizer_weight_clip: 0` にして、clipの有無まで変わらないようにしてください。

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

L2とL3それぞれの入力平均ベクトル `c` をGPU上で計算します。bucket別平均ではなく、optimizer更新に使う全batch・全局面の平均です。`batches_per_update=4` なら4batch分の平均と累積勾配を使い、最後のbatchだけでは計算しません。更新直前に `beta = b + W*c`、`gW_center = gW - gb*c` とし、この座標で既存のoptimizer更新を行います。更新後に `b = beta - W*c` に戻します。Lookaheadのslow weight/biasも同様に変換します。

forward・validation・nn.binは従来どおり `W*x+b` です。BatchNormや入力の分散正規化ではありません。中心化した勾配に対してmomentum等を更新するため、通常のRangerと同じ学習アルゴリズムではありません。報告されるlossに追加の罰則は加えません。

平均計算用GPU領域は有効時に確保して再利用します。1024/7/64構成では約10KiBで、重み全体をCPUへ転送する処理はありません。

## 現在の対応範囲と再開

- cuda-cppのSFNN、`batches_per_update>=1`、`sfnn_update_scope=all`。bpuは比較対象と同じ値を使用できます。
- L1 factorizerはnone/shared。FT factorizerとL1 QATは使用可能。
- weight clipは中心化中は無効。省略または正の値なら警告を出して無効化し、学習を続行します。明示的に0なら警告は出しません。JSON自体は書き換えません。weight decay・Norm loss・saturation penalty・factorizer residual decayは0。
- bucket counts/count gates、axis/pair factorizerとの併用は現在非対応。
- L2入力幅・出力幅は各256以下、`bucket数 × L2幅` は65536以下。
- clip以外の非対応の組み合わせは引き続きエラーにします。中心化OFFのepochでは設定されたclip動作に戻ります。中心化だけを比較したい場合は、共通設定で `optimizer_weight_clip=0` としてください。

checkpointとnn.binの形式は変更しません。checkpoint保存時には通常のweight/bias表現に戻っています。ON/OFFを変えて再開できますが、optimizerのmomentum等は引き継ぎ、自動リセットしません。これは学習条件変更なので、新規学習でのA/B比較と同じではありません。再開時もJSONまたはCLIでON/OFFを明示してください。

旧調査用の `BULLETOU_EXPERIMENT_*` 環境変数を設定したシェルでは使用せず、通常の環境からこのオプションを指定してください。検証では飽和率・検証精度の改善を確認しましたが、長期学習の棋力改善を保証するものではありません。
