# FTの継続飽和に対する線形penalty（実験用）

FTのあるunitがほぼ常に上限に達する現象を抑えるための、デフォルトOFFの実験機能です。unitのリセットでも、通常のclamp backwardをSTEに変更する機能でもありません。常時飽和が必ず棋力低下を意味するわけではないので、A/B比較してください。

| JSON設定（CLIでは `_` を `-` に変更） | デフォルト | 意味 |
|---|---:|---|
| `sfnn_ft_saturation_penalty` | 0 | penalty強度 λ。0は完全OFF。有限の非負値 |
| `sfnn_ft_saturation_rate` | 0.2 | 対象unitの上限到達率の閾値。(0,1]。0.2は20% |
| `sfnn_ft_saturation_patience` | 1 | 閾値以上が連続する学習microbatch数。正の整数 |

3項目ともepoch別設定に対応します。未来epochの設定は現在の判定に影響しません。

## 判定と勾配

各学習batchの両視点で、FT各unitの上限到達率を集計します。検証データは判定に使用しません。entry weightが0の局面は判定から除外し、正のentry weightは大きさによらず1局面として数えます。閾値以上が指定batch数連続したunitだけを対象にします。デフォルトでは20%以上になったそのbatchから有効です。閾値未満、または有効局面がないbatchでは連続数を0に戻します。`batches_per_update=4`でも4microbatchとして数え、sb数やoptimizer更新回数では数えません。

FTのclamp前の値を $z_{isu}$、局面数を $B$、FT幅を $F$、両視点を $s$、entry weightを $w_i$、対象unitのマスクを $m_u$ とすると、追加勾配に対応するpenaltyは

$$
L_{\mathrm{FT}}=\frac{\lambda}{2BF}\sum_{i,s,u}w_i m_u\max(0,z_{isu}-1).
$$

判定マスクは微分しません。$z>1$ の追加勾配は $\lambda w_i m_u/(2BF)$、$z<1$ は0です。$z=1$では同じ正のsubgradientを選びます。したがって上限に張り付いて通常のclamp勾配が0でも、このpenaltyは下げる方向に働きます。FP32のclamp後activationから上限到達を判定できるので、巨大なclamp前activation配列は追加しません。penaltyの数値自体は集計せず、対応する勾配だけをFT weight/biasへ加えます。FT factorizerのαを含む連鎖則も適用します。

通常のtask勾配は変更しません。勾配蓄積時も各microbatchで追加し、既存の更新時の平均化を使用します。L1/L2/L3には直接penaltyを加えません。`loss/qloss`の報告値にこのpenaltyは含めません。既存の `sfnn_saturation_penalty`（別の重みpenalty）とは別機能です。

## A/B実験

既存のgridコマンドに次を加えます（値は比較の出発点であり最適値ではありません）。

```powershell
  --grid sfnn-ft-saturation-penalty 0 0.0001 0.001
```

同じ初期状態、lr、学習量などを使い、`--verbose`でFT平均飽和率とunit最大飽和率も比較してください。短期の飽和率改善が棋力改善を保証するものではありません。

cuda-cpp SFNN、update scope=allに対応します。FT factorizer、QAT、L1/L2/L3中心化、L1 effective weight clip、bpu>1と併用できます。FT更新を凍結した場合は動作しません。

判定はGPU内で完結し、CPU readbackはありません。追加履歴はFT=1024で約8KiBですが、activationの集計と追加勾配計算の時間はかかります。OFFでは追加の集計・GPU領域確保はありません。

判定履歴はcheckpointに保存しません。resume、worker snapshot復元、設定変更、無効化では連続数をリセットして再判定します。nn.binや推論処理は変更しません。学習中の設定ファイル・実行ファイルは自動更新しません。新しい実行ファイルで学習を開始／再開してください。
