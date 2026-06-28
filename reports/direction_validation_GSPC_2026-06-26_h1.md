─────────────────────────────────────────────────────────────────────────────

• レポートは direction_validation_GSPC_2026-06-26_h10.md と
  direction_validation_N225_2026-06-26_h10.md を見ました。結論は、両方とも
  「方向の並び替え能力は一部あるが、閾値設計と特徴量の非定常性で最終判定が崩
  れている」です。

  分析

  - ^GSPC は balanced_accuracy mean = 0.5512 で最低基準 0.52 は超えています
    が、roc_auc mean = 0.5192 と beats_baseline_fold_ratio = 0.2222 が弱く、
    ベースライン超えが 9 Fold 中 2 Fold しかありません。予測能力そのものが弱
    い Fold が多いです。
  - ^N225 は roc_auc mean = 0.5870 で ^GSPC より序列付けは良いのに、
    balanced_accuracy mean = 0.5027、beats_baseline_fold_ratio = 0.2222 で最
    終精度が出ていません。こちらは「モデルより閾値側の問題」がより強いです。
  - 両銘柄とも Fold ごとの decision_threshold の振れ幅が極端です。^GSPC は
    0.0157 から 0.9891、^N225 は 0.0156 から 0.9525 まで飛んでいます。これは
    安定した分類境界ではありません。
  - 実際に単一クラス予測が何度も出ています。^GSPC では Fold 1, 2, 7 が全件
    DOWN 側、Fold 9 が全件 UP 側です。^N225 でも Fold 7 が全件 DOWN、Fold 8,
    9 が全件 UP です。balanced_accuracy = 0.5 が並ぶのはこのためです。
  - それでも ^N225 Fold 8 は AUC = 0.7596、Fold 3 は AUC = 0.8035 です。順位
    付けはできているのに、しきい値が 0.0156 や 0.2674 で全件 UP に寄って壊れ
    ています。ここは改善余地が大きいです。
  - 重要特徴量は両方とも ema_12, ema_26, volatility_60, return_60,
    macd_signal に集中しています。特に ema_12 と ema_26 は生の価格水準で、
    predictors_direction_lgbm.py:147 でそのまま特徴量化されています。指数の絶
    対水準は時期で大きく変わるので、非定常で過学習しやすいです。
  - 目的変数は config.py:40 の target_return_threshold = 0.005 と、
    predictors_direction_lgbm.py:165 の 10営業日先リターンで作っています。こ
    れだと「小幅上昇でも 0.5% 未満なら DOWN 扱い」になり、相場局面ごとにクラ
    ス比率が崩れやすいです。
  - さらに閾値探索は _fit_direction_model_with_threshold:564 で fit_df に学習
    した threshold_model の確率から求めていますが、実際の予測には別の model
    を train_df 全体で再学習して使っています。閾値を決めたモデルと本番モデル
    が違うので、確率分布がズレて極端なしきい値を誘発しやすいです。

  精度向上施策

  - 最優先は閾値ロジックの修正です。閾値を決めたモデルと本番モデルを同一にし
    てください。実装としては「fit_df 学習モデルを calibration に使い、その同
    じモデルで Fold 予測する」
  - ema_12 と ema_26 の生値はやめて、相対値に置き換えるべきです。close /
    ema_12 - 1, close / ema_26 - 1, ema_12 / ema_26 - 1, macd / close のよう
    に価格水準依存を落とすのが先です。ここは効果が出やすいです。
  