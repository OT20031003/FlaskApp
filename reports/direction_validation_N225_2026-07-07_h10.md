# Direction Validation Report: ^N225

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^N225 |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-08 14:07:06 NDT |
| Last Date | 2026-07-07 |
| Last Close | 68256.9609 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.9366 |
| Probability Down | 0.0634 |
| Decision Threshold | 0.0553 |
| Model Valid | False |
| Validation Method | purged_rolling_walk_forward |
| Target Return Threshold | 0.0050 |
| Buy Probability Threshold | 0.6000 |
| Sell Probability Threshold | 0.4000 |

## Rolling Configuration
| Item | Value |
| --- | --- |
| train_size | 504 |
| test_size | 63 |
| step_size | 63 |
| purge_size | 10 |
| calibration_ratio | 0.2500 |
| threshold_metric | balanced_accuracy |
| ridge_stack_alpha | 1.0000 |
| ridge_stack_train_window | 252 |
| ridge_stack_min_train_samples | 120 |

## Validation Thresholds
| Item | Value |
| --- | --- |
| min_rolling_folds | 3 |
| min_balanced_accuracy_mean | 0.5200 |
| min_roc_auc_mean | 0.5200 |
| min_beats_baseline_fold_ratio | 0.5000 |

## Aggregate Validation Metrics
| Metric | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| fold_count | 28 | N/A | N/A | N/A |
| total_fold_count | 28 | N/A | N/A | N/A |
| beats_baseline_fold_ratio | 0.3571 | N/A | N/A | N/A |
| confusion_matrix_sum | [[510, 323], [497, 434]] | N/A | N/A | N/A |
| accuracy | 0.5352 | 0.1298 | 0.2698 | 0.7937 |
| balanced_accuracy | 0.5589 | 0.0938 | 0.4056 | 0.7333 |
| precision | 0.5963 | 0.2628 | 0.0000 | 1.0000 |
| recall | 0.4824 | 0.3178 | 0.0000 | 1.0000 |
| f1 | 0.4557 | 0.2308 | 0.0000 | 0.8571 |
| roc_auc | 0.6144 | 0.1191 | 0.3926 | 0.8174 |
| log_loss | 1.1357 | 0.3736 | 0.6061 | 2.0982 |
| baseline_accuracy | 0.6026 | 0.0750 | 0.5079 | 0.7460 |
| decision_threshold | 0.5218 | 0.3317 | 0.0196 | 0.9644 |
| calibration_score | 0.5934 | 0.0507 | 0.5082 | 0.7047 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 398 | 942.5528 | 0.1823 |
| 2 | volatility_60 | 215 | 551.6690 | 0.1067 |
| 3 | ma_gap_60 | 91 | 488.4146 | 0.0944 |
| 4 | volatility_20 | 191 | 402.3574 | 0.0778 |
| 5 | ma_gap_120 | 137 | 329.4304 | 0.0637 |
| 6 | macd_hist_to_close | 137 | 286.2261 | 0.0553 |
| 7 | close_to_ema_12 | 159 | 267.3829 | 0.0517 |
| 8 | ma_gap_5 | 106 | 247.0717 | 0.0478 |
| 9 | ema_12_to_ema_26 | 197 | 238.1598 | 0.0461 |
| 10 | close_to_ema_26 | 70 | 150.6865 | 0.0291 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-03-01 | 2019-06-10 | 0.5397 | 0.5203 | 0.5284 | 0.5397 | 0.9079 | ok |
| 2 | ok | 2019-06-11 | 2019-09-09 | 0.4444 | 0.4471 | 0.4486 | 0.5079 | 0.7750 | ok |
| 3 | ok | 2019-09-11 | 2019-12-16 | 0.4286 | 0.5195 | 0.6084 | 0.6032 | 0.5613 | ok |
| 4 | ok | 2019-12-17 | 2020-03-24 | 0.3651 | 0.4353 | 0.5707 | 0.6349 | 0.3701 | ok |
| 5 | ok | 2020-03-25 | 2020-06-25 | 0.5397 | 0.6360 | 0.7407 | 0.6825 | 0.8351 | ok |
| 6 | ok | 2020-06-26 | 2020-10-02 | 0.4286 | 0.5000 | 0.5473 | 0.5714 | 0.9461 | ok |
| 7 | ok | 2020-10-05 | 2021-01-05 | 0.7143 | 0.5000 | 0.5481 | 0.7143 | 0.0359 | ok |
| 8 | ok | 2021-01-06 | 2021-04-07 | 0.5079 | 0.5303 | 0.6475 | 0.5238 | 0.7471 | ok |
| 9 | ok | 2021-04-08 | 2021-07-09 | 0.2698 | 0.5000 | 0.4297 | 0.7302 | 0.0196 | ok |
| 10 | ok | 2021-07-12 | 2021-10-13 | 0.5714 | 0.5636 | 0.5444 | 0.5238 | 0.6848 | ok |
| 11 | ok | 2021-10-14 | 2022-01-17 | 0.5238 | 0.5272 | 0.5801 | 0.6190 | 0.9644 | ok |
| 12 | ok | 2022-01-18 | 2022-04-19 | 0.4921 | 0.6145 | 0.7058 | 0.6825 | 0.4339 | ok |
| 13 | ok | 2022-04-20 | 2022-07-22 | 0.6032 | 0.6429 | 0.7796 | 0.5556 | 0.9412 | ok |
| 14 | ok | 2022-07-25 | 2022-10-25 | 0.5556 | 0.5299 | 0.4787 | 0.5397 | 0.9390 | ok |
| 15 | ok | 2022-10-26 | 2023-01-27 | 0.6508 | 0.6502 | 0.6784 | 0.5079 | 0.1268 | ok |
| 16 | ok | 2023-01-30 | 2023-04-28 | 0.6984 | 0.7156 | 0.7428 | 0.6508 | 0.1401 | ok |
| 17 | ok | 2023-05-01 | 2023-08-01 | 0.6190 | 0.6435 | 0.7315 | 0.5714 | 0.1982 | ok |
| 18 | ok | 2023-08-02 | 2023-11-01 | 0.5079 | 0.4959 | 0.5548 | 0.5397 | 0.4391 | ok |
| 19 | ok | 2023-11-02 | 2024-02-06 | 0.7937 | 0.7285 | 0.8174 | 0.6825 | 0.0711 | ok |
| 20 | ok | 2024-02-07 | 2024-05-13 | 0.5238 | 0.5045 | 0.7747 | 0.5238 | 0.0584 | ok |
| 21 | ok | 2024-05-14 | 2024-08-09 | 0.4603 | 0.4948 | 0.4927 | 0.5873 | 0.1819 | ok |
| 22 | ok | 2024-08-13 | 2024-11-13 | 0.7302 | 0.7333 | 0.7081 | 0.5238 | 0.6069 | ok |
| 23 | ok | 2024-11-14 | 2025-02-18 | 0.6190 | 0.5000 | 0.6635 | 0.6190 | 0.8772 | ok |
| 24 | ok | 2025-02-19 | 2025-05-23 | 0.6667 | 0.6561 | 0.6394 | 0.5238 | 0.5884 | ok |
| 25 | ok | 2025-05-26 | 2025-08-22 | 0.3651 | 0.5349 | 0.6802 | 0.6825 | 0.8515 | ok |
| 26 | ok | 2025-08-25 | 2025-11-26 | 0.3810 | 0.4202 | 0.4202 | 0.7460 | 0.5998 | ok |
| 27 | ok | 2025-11-27 | 2026-03-03 | 0.6667 | 0.6991 | 0.7479 | 0.5714 | 0.0459 | ok |
| 28 | ok | 2026-03-04 | 2026-06-05 | 0.3175 | 0.4056 | 0.3926 | 0.7143 | 0.6649 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-01-05 |
| Train End | 2019-02-13 |
| Test Start | 2019-03-01 |
| Test End | 2019-06-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5203 |
| Precision | 0.5000 |
| Recall | 0.2759 |
| F1 | 0.3556 |
| ROC-AUC | 0.5284 |
| Log Loss | 1.3407 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9079 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6045 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[26, 8], [21, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0028 |
| p25 | 0.0567 |
| median | 0.5190 |
| p75 | 0.8988 |
| max | 0.9813 |
| mean | 0.5138 |
| std | 0.3780 |
| count_ge_threshold | 16 |
| count_lt_threshold | 47 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 282 | 1066.2990 | 0.2001 |
| 2 | ma_gap_120 | 384 | 862.2506 | 0.1618 |
| 3 | volatility_20 | 220 | 407.8299 | 0.0765 |
| 4 | macd_hist_to_close | 251 | 325.6536 | 0.0611 |
| 5 | return_10 | 240 | 316.8460 | 0.0595 |
| 6 | ema_12_to_ema_26 | 91 | 244.2376 | 0.0458 |
| 7 | return_60 | 95 | 208.2286 | 0.0391 |
| 8 | macd_signal_to_close | 122 | 198.4676 | 0.0372 |
| 9 | close_to_low | 162 | 143.5397 | 0.0269 |
| 10 | close_to_ema_26 | 106 | 140.5662 | 0.0264 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-04-06 |
| Train End | 2019-05-24 |
| Test Start | 2019-06-11 |
| Test End | 2019-09-09 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.4471 |
| Precision | 0.4286 |
| Recall | 0.2812 |
| F1 | 0.3396 |
| ROC-AUC | 0.4486 |
| Log Loss | 0.9550 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.7750 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6026 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 12], [23, 9]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0744 |
| p25 | 0.2865 |
| median | 0.5892 |
| p75 | 0.7929 |
| max | 0.9433 |
| mean | 0.5629 |
| std | 0.2641 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 264 | 926.0799 | 0.1787 |
| 2 | ma_gap_120 | 263 | 680.9268 | 0.1314 |
| 3 | ridge_pred_future_log_return | 211 | 505.0083 | 0.0974 |
| 4 | macd_hist_to_close | 116 | 421.0357 | 0.0812 |
| 5 | volatility_20 | 194 | 277.5999 | 0.0536 |
| 6 | volume_change_5 | 118 | 197.7055 | 0.0381 |
| 7 | return_60 | 151 | 194.0275 | 0.0374 |
| 8 | macd_signal_to_close | 111 | 191.3281 | 0.0369 |
| 9 | return_10 | 142 | 158.6632 | 0.0306 |
| 10 | return_20 | 123 | 157.4229 | 0.0304 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-07-13 |
| Train End | 2019-08-26 |
| Test Start | 2019-09-11 |
| Test End | 2019-12-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.5195 |
| Precision | 0.7500 |
| Recall | 0.0789 |
| F1 | 0.1429 |
| ROC-AUC | 0.6084 |
| Log Loss | 1.6391 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.5613 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5482 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 1], [35, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0029 |
| p25 | 0.0220 |
| median | 0.0513 |
| p75 | 0.1139 |
| max | 0.8962 |
| mean | 0.1300 |
| std | 0.1956 |
| count_ge_threshold | 4 |
| count_lt_threshold | 59 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 321 | 1396.0548 | 0.2634 |
| 2 | volatility_20 | 366 | 935.6121 | 0.1765 |
| 3 | ma_gap_120 | 191 | 600.6466 | 0.1133 |
| 4 | return_60 | 231 | 379.9340 | 0.0717 |
| 5 | return_20 | 210 | 322.5885 | 0.0609 |
| 6 | return_5 | 97 | 251.8778 | 0.0475 |
| 7 | macd_signal_to_close | 154 | 196.1055 | 0.0370 |
| 8 | volume_change_5 | 123 | 172.9508 | 0.0326 |
| 9 | ridge_pred_future_log_return | 122 | 126.6074 | 0.0239 |
| 10 | return_2 | 107 | 90.5914 | 0.0171 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-10-16 |
| Train End | 2019-12-02 |
| Test Start | 2019-12-17 |
| Test End | 2020-03-24 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.4353 |
| Precision | 0.3265 |
| Recall | 0.6957 |
| F1 | 0.4444 |
| ROC-AUC | 0.5707 |
| Log Loss | 1.1987 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.3701 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5839 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 33], [7, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0030 |
| p25 | 0.4795 |
| median | 0.6428 |
| p75 | 0.8356 |
| max | 0.9787 |
| mean | 0.5851 |
| std | 0.3257 |
| count_ge_threshold | 49 |
| count_lt_threshold | 14 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 351 | 893.6720 | 0.1755 |
| 2 | volatility_20 | 313 | 592.0582 | 0.1163 |
| 3 | volatility_60 | 239 | 548.5209 | 0.1077 |
| 4 | ridge_pred_future_log_return | 232 | 396.3092 | 0.0778 |
| 5 | macd_signal_to_close | 230 | 396.2634 | 0.0778 |
| 6 | volume_change_5 | 145 | 231.7395 | 0.0455 |
| 7 | ma_gap_60 | 102 | 186.5374 | 0.0366 |
| 8 | volume_z_20 | 124 | 151.5840 | 0.0298 |
| 9 | return_10 | 103 | 147.9000 | 0.0290 |
| 10 | return_20 | 115 | 145.3841 | 0.0285 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-01-22 |
| Train End | 2020-03-09 |
| Test Start | 2020-03-25 |
| Test End | 2020-06-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.6360 |
| Precision | 0.8889 |
| Recall | 0.3721 |
| F1 | 0.5246 |
| ROC-AUC | 0.7407 |
| Log Loss | 1.2157 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.8351 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5238 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[18, 2], [27, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0067 |
| p25 | 0.0243 |
| median | 0.1190 |
| p75 | 0.8988 |
| max | 0.9915 |
| mean | 0.4169 |
| std | 0.4146 |
| count_ge_threshold | 18 |
| count_lt_threshold | 45 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 267 | 857.1847 | 0.1701 |
| 2 | volatility_60 | 218 | 731.9332 | 0.1453 |
| 3 | ema_12_to_ema_26 | 182 | 317.8534 | 0.0631 |
| 4 | volatility_20 | 137 | 298.1411 | 0.0592 |
| 5 | volume_change_20 | 203 | 249.7870 | 0.0496 |
| 6 | return_5 | 144 | 244.0181 | 0.0484 |
| 7 | ma_gap_60 | 122 | 232.7087 | 0.0462 |
| 8 | ma_gap_120 | 155 | 224.2884 | 0.0445 |
| 9 | ma_gap_20 | 92 | 208.3271 | 0.0413 |
| 10 | return_60 | 111 | 183.7843 | 0.0365 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-04-24 |
| Train End | 2020-06-11 |
| Test Start | 2020-06-26 |
| Test End | 2020-10-02 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5473 |
| Log Loss | 0.8163 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.9461 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5714 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[27, 0], [36, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0316 |
| p25 | 0.4343 |
| median | 0.6739 |
| p75 | 0.8209 |
| max | 0.9287 |
| mean | 0.5923 |
| std | 0.2595 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 373 | 946.0924 | 0.1892 |
| 2 | macd_signal_to_close | 266 | 435.5673 | 0.0871 |
| 3 | volatility_20 | 192 | 334.9584 | 0.0670 |
| 4 | macd_hist_to_close | 133 | 295.9527 | 0.0592 |
| 5 | return_20 | 111 | 289.0279 | 0.0578 |
| 6 | ema_12_to_ema_26 | 152 | 257.1759 | 0.0514 |
| 7 | volume_change_20 | 162 | 217.7376 | 0.0436 |
| 8 | ma_gap_60 | 129 | 216.6737 | 0.0433 |
| 9 | ma_gap_120 | 139 | 210.8099 | 0.0422 |
| 10 | volume_change_5 | 157 | 193.8384 | 0.0388 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-07-26 |
| Train End | 2020-09-15 |
| Test Start | 2020-10-05 |
| Test End | 2021-01-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7143 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.7143 |
| Recall | 1.0000 |
| F1 | 0.8333 |
| ROC-AUC | 0.5481 |
| Log Loss | 1.0008 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.0359 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5560 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 18], [0, 45]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0519 |
| p25 | 0.1729 |
| median | 0.2977 |
| p75 | 0.6698 |
| max | 0.9377 |
| mean | 0.3987 |
| std | 0.2800 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 270 | 519.2322 | 0.1024 |
| 2 | ma_gap_120 | 270 | 506.5030 | 0.0999 |
| 3 | volatility_60 | 263 | 443.3471 | 0.0874 |
| 4 | return_60 | 136 | 294.5594 | 0.0581 |
| 5 | macd_signal_to_close | 139 | 283.9487 | 0.0560 |
| 6 | ma_gap_60 | 98 | 277.9999 | 0.0548 |
| 7 | macd_hist_to_close | 175 | 248.9999 | 0.0491 |
| 8 | volume_change_20 | 148 | 225.1746 | 0.0444 |
| 9 | volatility_20 | 219 | 218.3087 | 0.0430 |
| 10 | close_to_high | 170 | 209.5556 | 0.0413 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-10-29 |
| Train End | 2020-12-18 |
| Test Start | 2021-01-06 |
| Test End | 2021-04-07 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.5303 |
| Precision | 1.0000 |
| Recall | 0.0606 |
| F1 | 0.1143 |
| ROC-AUC | 0.6475 |
| Log Loss | 1.2087 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.7471 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6106 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 0], [31, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0081 |
| p25 | 0.0305 |
| median | 0.0773 |
| p75 | 0.2563 |
| max | 0.8684 |
| mean | 0.1581 |
| std | 0.1800 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 234 | 648.0258 | 0.1232 |
| 2 | volume_change_20 | 288 | 434.8915 | 0.0827 |
| 3 | volatility_60 | 227 | 424.1794 | 0.0806 |
| 4 | macd_signal_to_close | 177 | 387.9657 | 0.0738 |
| 5 | ma_gap_60 | 120 | 379.4707 | 0.0721 |
| 6 | volatility_20 | 160 | 367.3416 | 0.0698 |
| 7 | return_20 | 100 | 322.7301 | 0.0614 |
| 8 | ridge_pred_future_log_return | 218 | 309.8940 | 0.0589 |
| 9 | macd_hist_to_close | 84 | 209.3195 | 0.0398 |
| 10 | volume_change_5 | 200 | 205.2224 | 0.0390 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-02-14 |
| Train End | 2021-03-24 |
| Test Start | 2021-04-08 |
| Test End | 2021-07-09 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2698 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.2698 |
| Recall | 1.0000 |
| F1 | 0.4250 |
| ROC-AUC | 0.4297 |
| Log Loss | 1.1464 |
| Baseline Accuracy | 0.7302 |
| Decision Threshold | 0.0196 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6484 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 46], [0, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0821 |
| p25 | 0.3891 |
| median | 0.6481 |
| p75 | 0.8230 |
| max | 0.9848 |
| mean | 0.5928 |
| std | 0.2554 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 157 | 674.2635 | 0.1281 |
| 2 | ridge_pred_future_log_return | 266 | 671.8586 | 0.1277 |
| 3 | close_to_ema_12 | 207 | 502.4263 | 0.0955 |
| 4 | macd_signal_to_close | 214 | 373.3094 | 0.0709 |
| 5 | volume_change_20 | 219 | 355.3283 | 0.0675 |
| 6 | volatility_60 | 191 | 239.1594 | 0.0454 |
| 7 | volatility_20 | 190 | 220.7413 | 0.0419 |
| 8 | return_60 | 137 | 198.7809 | 0.0378 |
| 9 | return_10 | 151 | 193.9853 | 0.0369 |
| 10 | volume_z_20 | 243 | 169.7036 | 0.0322 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-05-28 |
| Train End | 2021-06-25 |
| Test Start | 2021-07-12 |
| Test End | 2021-10-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5714 |
| Balanced Accuracy | 0.5636 |
| Precision | 0.5714 |
| Recall | 0.4000 |
| F1 | 0.4706 |
| ROC-AUC | 0.5444 |
| Log Loss | 0.9918 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.6848 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6591 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 9], [18, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0154 |
| p25 | 0.1891 |
| median | 0.4789 |
| p75 | 0.7435 |
| max | 0.9898 |
| mean | 0.4835 |
| std | 0.3180 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 298 | 484.8959 | 0.0930 |
| 2 | macd_signal_to_close | 221 | 473.2918 | 0.0908 |
| 3 | macd_hist_to_close | 155 | 447.6541 | 0.0859 |
| 4 | ridge_pred_future_log_return | 124 | 412.7954 | 0.0792 |
| 5 | close_to_ema_12 | 127 | 328.9095 | 0.0631 |
| 6 | return_60 | 210 | 294.4305 | 0.0565 |
| 7 | ma_gap_120 | 144 | 286.1980 | 0.0549 |
| 8 | volatility_20 | 137 | 281.8740 | 0.0541 |
| 9 | close_to_ema_26 | 94 | 221.0850 | 0.0424 |
| 10 | return_10 | 225 | 211.1128 | 0.0405 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-08-27 |
| Train End | 2021-09-29 |
| Test Start | 2021-10-14 |
| Test End | 2022-01-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5272 |
| Precision | 0.4062 |
| Recall | 0.5417 |
| F1 | 0.4643 |
| ROC-AUC | 0.5801 |
| Log Loss | 2.0544 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.9644 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6432 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 19], [11, 13]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1957 |
| p25 | 0.9073 |
| median | 0.9679 |
| p75 | 0.9892 |
| max | 0.9981 |
| mean | 0.9060 |
| std | 0.1446 |
| count_ge_threshold | 32 |
| count_lt_threshold | 31 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 201 | 727.7635 | 0.1393 |
| 2 | ma_gap_60 | 140 | 596.3036 | 0.1141 |
| 3 | ma_gap_120 | 188 | 380.8973 | 0.0729 |
| 4 | ema_12_to_ema_26 | 147 | 297.4926 | 0.0569 |
| 5 | macd_signal_to_close | 165 | 282.4957 | 0.0541 |
| 6 | rsi_14 | 64 | 224.9485 | 0.0431 |
| 7 | ma_gap_20 | 38 | 216.2478 | 0.0414 |
| 8 | close_to_ema_12 | 138 | 215.2620 | 0.0412 |
| 9 | close_to_high | 164 | 210.4557 | 0.0403 |
| 10 | volume_change_5 | 96 | 190.5610 | 0.0365 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-12-03 |
| Train End | 2021-12-29 |
| Test Start | 2022-01-18 |
| Test End | 2022-04-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.6145 |
| Precision | 0.3800 |
| Recall | 0.9500 |
| F1 | 0.5429 |
| ROC-AUC | 0.7058 |
| Log Loss | 1.3821 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.4339 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6042 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 31], [1, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0234 |
| p25 | 0.5166 |
| median | 0.8992 |
| p75 | 0.9781 |
| max | 0.9949 |
| mean | 0.7239 |
| std | 0.3140 |
| count_ge_threshold | 50 |
| count_lt_threshold | 13 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 368 | 1061.6648 | 0.2012 |
| 2 | ma_gap_20 | 74 | 471.0079 | 0.0892 |
| 3 | macd_signal_to_close | 202 | 282.1185 | 0.0535 |
| 4 | volume_change_5 | 170 | 268.6453 | 0.0509 |
| 5 | ridge_pred_future_log_return | 121 | 266.0726 | 0.0504 |
| 6 | volume_change_20 | 187 | 258.6796 | 0.0490 |
| 7 | macd_hist_to_close | 137 | 258.6522 | 0.0490 |
| 8 | ma_gap_60 | 155 | 242.7853 | 0.0460 |
| 9 | close_to_high | 172 | 189.7981 | 0.0360 |
| 10 | volatility_20 | 174 | 173.8210 | 0.0329 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-10 |
| Train End | 2022-04-05 |
| Test Start | 2022-04-20 |
| Test End | 2022-07-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.6429 |
| Precision | 1.0000 |
| Recall | 0.2857 |
| F1 | 0.4444 |
| ROC-AUC | 0.7796 |
| Log Loss | 0.6061 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9412 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6122 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 0], [25, 10]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0196 |
| p25 | 0.3059 |
| median | 0.6582 |
| p75 | 0.8945 |
| max | 0.9884 |
| mean | 0.5903 |
| std | 0.3118 |
| count_ge_threshold | 10 |
| count_lt_threshold | 53 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 337 | 944.1320 | 0.1786 |
| 2 | volatility_20 | 262 | 523.7025 | 0.0990 |
| 3 | macd_signal_to_close | 134 | 346.4363 | 0.0655 |
| 4 | ridge_pred_future_log_return | 185 | 336.8102 | 0.0637 |
| 5 | volume_change_5 | 208 | 318.7925 | 0.0603 |
| 6 | return_10 | 181 | 306.1798 | 0.0579 |
| 7 | high_low_range | 132 | 294.9753 | 0.0558 |
| 8 | macd_to_close | 97 | 248.0372 | 0.0469 |
| 9 | return_60 | 175 | 229.4671 | 0.0434 |
| 10 | volume_change_20 | 167 | 188.7187 | 0.0357 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-12 |
| Train End | 2022-07-07 |
| Test Start | 2022-07-25 |
| Test End | 2022-10-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5299 |
| Precision | 0.5455 |
| Recall | 0.2069 |
| F1 | 0.3000 |
| ROC-AUC | 0.4787 |
| Log Loss | 1.3323 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9390 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7047 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[29, 5], [23, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0046 |
| p25 | 0.0724 |
| median | 0.3670 |
| p75 | 0.8575 |
| max | 0.9929 |
| mean | 0.4706 |
| std | 0.3724 |
| count_ge_threshold | 11 |
| count_lt_threshold | 52 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 336 | 794.7740 | 0.1531 |
| 2 | return_20 | 119 | 374.7913 | 0.0722 |
| 3 | volatility_20 | 201 | 353.7142 | 0.0682 |
| 4 | return_5 | 183 | 335.2883 | 0.0646 |
| 5 | volume_change_5 | 230 | 330.6528 | 0.0637 |
| 6 | macd_to_close | 64 | 310.6814 | 0.0599 |
| 7 | macd_signal_to_close | 184 | 277.5890 | 0.0535 |
| 8 | return_60 | 199 | 259.7635 | 0.0501 |
| 9 | high_low_range | 135 | 230.0163 | 0.0443 |
| 10 | ema_12_to_ema_26 | 104 | 222.9488 | 0.0430 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-16 |
| Train End | 2022-10-11 |
| Test Start | 2022-10-26 |
| Test End | 2023-01-27 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.6502 |
| Precision | 0.6552 |
| Recall | 0.6129 |
| F1 | 0.6333 |
| ROC-AUC | 0.6784 |
| Log Loss | 1.1969 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.1268 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5664 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 10], [12, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0015 |
| p25 | 0.0140 |
| median | 0.0716 |
| p75 | 0.8977 |
| max | 0.9951 |
| mean | 0.3747 |
| std | 0.4135 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 212 | 878.6948 | 0.1692 |
| 2 | volatility_60 | 285 | 579.1844 | 0.1115 |
| 3 | macd_signal_to_close | 204 | 472.9945 | 0.0911 |
| 4 | return_5 | 198 | 355.6294 | 0.0685 |
| 5 | return_10 | 163 | 353.0764 | 0.0680 |
| 6 | close_to_high | 253 | 343.2355 | 0.0661 |
| 7 | volatility_20 | 192 | 278.3877 | 0.0536 |
| 8 | ma_gap_20 | 120 | 172.3526 | 0.0332 |
| 9 | return_60 | 100 | 169.6443 | 0.0327 |
| 10 | ema_12_to_ema_26 | 115 | 144.5146 | 0.0278 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-21 |
| Train End | 2023-01-13 |
| Test Start | 2023-01-30 |
| Test End | 2023-04-28 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6984 |
| Balanced Accuracy | 0.7156 |
| Precision | 0.8438 |
| Recall | 0.6585 |
| F1 | 0.7397 |
| ROC-AUC | 0.7428 |
| Log Loss | 1.0094 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.1401 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5292 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 5], [14, 27]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0372 |
| p25 | 0.0967 |
| median | 0.1560 |
| p75 | 0.4416 |
| max | 0.9774 |
| mean | 0.2897 |
| std | 0.2441 |
| count_ge_threshold | 32 |
| count_lt_threshold | 31 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 421 | 901.1321 | 0.1738 |
| 2 | return_20 | 249 | 818.6832 | 0.1579 |
| 3 | macd_signal_to_close | 310 | 772.8165 | 0.1491 |
| 4 | ema_12_to_ema_26 | 72 | 278.4667 | 0.0537 |
| 5 | close_to_high | 163 | 249.2476 | 0.0481 |
| 6 | close_to_low | 195 | 180.7434 | 0.0349 |
| 7 | ma_gap_120 | 99 | 179.2131 | 0.0346 |
| 8 | ma_gap_20 | 27 | 176.3388 | 0.0340 |
| 9 | return_5 | 183 | 163.5498 | 0.0315 |
| 10 | ma_gap_60 | 62 | 124.3961 | 0.0240 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-25 |
| Train End | 2023-04-14 |
| Test Start | 2023-05-01 |
| Test End | 2023-08-01 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6435 |
| Precision | 0.7727 |
| Recall | 0.4722 |
| F1 | 0.5862 |
| ROC-AUC | 0.7315 |
| Log Loss | 1.0084 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.1982 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5767 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 5], [19, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0057 |
| p25 | 0.0823 |
| median | 0.1159 |
| p75 | 0.2611 |
| max | 0.7640 |
| mean | 0.1944 |
| std | 0.1731 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 204 | 648.1142 | 0.1303 |
| 2 | macd_signal_to_close | 204 | 459.5750 | 0.0924 |
| 3 | volatility_60 | 251 | 458.7703 | 0.0923 |
| 4 | close_to_high | 154 | 340.7555 | 0.0685 |
| 5 | ema_12_to_ema_26 | 209 | 292.3609 | 0.0588 |
| 6 | return_60 | 174 | 271.5619 | 0.0546 |
| 7 | volatility_20 | 196 | 229.8191 | 0.0462 |
| 8 | ma_gap_120 | 119 | 201.7079 | 0.0406 |
| 9 | macd_hist_to_close | 121 | 188.1011 | 0.0378 |
| 10 | volume_z_20 | 115 | 174.2431 | 0.0350 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-28 |
| Train End | 2023-07-18 |
| Test Start | 2023-08-02 |
| Test End | 2023-11-01 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.4959 |
| Precision | 0.4545 |
| Recall | 0.3448 |
| F1 | 0.3922 |
| ROC-AUC | 0.5548 |
| Log Loss | 0.9399 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.4391 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5301 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 12], [19, 10]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0071 |
| p25 | 0.1101 |
| median | 0.2862 |
| p75 | 0.6499 |
| max | 0.8799 |
| mean | 0.3654 |
| std | 0.2861 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 173 | 638.5113 | 0.1290 |
| 2 | volatility_60 | 217 | 589.4251 | 0.1190 |
| 3 | ema_12_to_ema_26 | 194 | 384.8435 | 0.0777 |
| 4 | ma_gap_60 | 145 | 310.1084 | 0.0626 |
| 5 | volume_z_20 | 130 | 232.6349 | 0.0470 |
| 6 | ma_gap_20 | 113 | 229.2652 | 0.0463 |
| 7 | macd_to_close | 102 | 228.1808 | 0.0461 |
| 8 | return_20 | 127 | 226.6815 | 0.0458 |
| 9 | ma_gap_120 | 162 | 187.1385 | 0.0378 |
| 10 | macd_hist_to_close | 92 | 178.9470 | 0.0361 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-30 |
| Train End | 2023-10-18 |
| Test Start | 2023-11-02 |
| Test End | 2024-02-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7937 |
| Balanced Accuracy | 0.7285 |
| Precision | 0.8125 |
| Recall | 0.9070 |
| F1 | 0.8571 |
| ROC-AUC | 0.8174 |
| Log Loss | 0.9086 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.0711 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5082 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 9], [4, 39]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0039 |
| p25 | 0.0814 |
| median | 0.2063 |
| p75 | 0.5193 |
| max | 0.9716 |
| mean | 0.3211 |
| std | 0.2828 |
| count_ge_threshold | 48 |
| count_lt_threshold | 15 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 202 | 662.2477 | 0.1282 |
| 2 | ridge_pred_future_log_return | 208 | 509.7295 | 0.0987 |
| 3 | volatility_60 | 250 | 481.6197 | 0.0933 |
| 4 | ma_gap_60 | 111 | 331.6969 | 0.0642 |
| 5 | return_2 | 160 | 267.8082 | 0.0519 |
| 6 | volatility_20 | 129 | 215.8902 | 0.0418 |
| 7 | volume_change_20 | 175 | 214.2309 | 0.0415 |
| 8 | ma_gap_120 | 154 | 211.6756 | 0.0410 |
| 9 | close_to_ema_12 | 47 | 201.0476 | 0.0389 |
| 10 | volume_z_20 | 205 | 191.2051 | 0.0370 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-30 |
| Train End | 2024-01-23 |
| Test Start | 2024-02-07 |
| Test End | 2024-05-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5045 |
| Precision | 0.5263 |
| Recall | 0.9091 |
| F1 | 0.6667 |
| ROC-AUC | 0.7747 |
| Log Loss | 0.7189 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.0584 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5317 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[3, 27], [3, 30]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0159 |
| p25 | 0.1114 |
| median | 0.2428 |
| p75 | 0.4915 |
| max | 0.8975 |
| mean | 0.3028 |
| std | 0.2330 |
| count_ge_threshold | 57 |
| count_lt_threshold | 6 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 317 | 778.2186 | 0.1497 |
| 2 | ridge_pred_future_log_return | 266 | 648.8900 | 0.1249 |
| 3 | return_60 | 169 | 610.9492 | 0.1176 |
| 4 | volatility_20 | 170 | 398.3416 | 0.0766 |
| 5 | macd_hist_to_close | 187 | 346.4350 | 0.0667 |
| 6 | close_to_high | 210 | 306.9260 | 0.0591 |
| 7 | ma_gap_60 | 134 | 269.6383 | 0.0519 |
| 8 | ma_gap_120 | 58 | 170.5779 | 0.0328 |
| 9 | ma_gap_5 | 136 | 164.9721 | 0.0317 |
| 10 | macd_signal_to_close | 186 | 154.5277 | 0.0297 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-04-06 |
| Train End | 2024-04-24 |
| Test Start | 2024-05-14 |
| Test End | 2024-08-09 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.4948 |
| Precision | 0.4091 |
| Recall | 0.6923 |
| F1 | 0.5143 |
| ROC-AUC | 0.4927 |
| Log Loss | 0.9686 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.1819 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6475 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 26], [8, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0279 |
| p25 | 0.1394 |
| median | 0.2731 |
| p75 | 0.5739 |
| max | 0.9370 |
| mean | 0.3858 |
| std | 0.2906 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 457 | 1206.6790 | 0.2271 |
| 2 | return_60 | 254 | 705.9032 | 0.1329 |
| 3 | volatility_60 | 278 | 529.7663 | 0.0997 |
| 4 | ridge_pred_future_log_return | 181 | 415.3579 | 0.0782 |
| 5 | macd_hist_to_close | 252 | 357.8984 | 0.0674 |
| 6 | ma_gap_120 | 146 | 357.4049 | 0.0673 |
| 7 | volatility_20 | 248 | 222.1948 | 0.0418 |
| 8 | ma_gap_20 | 50 | 173.8875 | 0.0327 |
| 9 | return_20 | 112 | 162.4578 | 0.0306 |
| 10 | volume_change_20 | 181 | 153.4844 | 0.0289 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-07-08 |
| Train End | 2024-07-26 |
| Test Start | 2024-08-13 |
| Test End | 2024-11-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7302 |
| Balanced Accuracy | 0.7333 |
| Precision | 0.6857 |
| Recall | 0.8000 |
| F1 | 0.7385 |
| ROC-AUC | 0.7081 |
| Log Loss | 0.7714 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.6069 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6043 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 11], [6, 24]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0029 |
| p25 | 0.1586 |
| median | 0.7494 |
| p75 | 0.9277 |
| max | 0.9886 |
| mean | 0.5752 |
| std | 0.3817 |
| count_ge_threshold | 35 |
| count_lt_threshold | 28 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_margin_to_threshold | 141 | 931.8425 | 0.1765 |
| 2 | volatility_60 | 258 | 533.4003 | 0.1010 |
| 3 | return_60 | 266 | 414.0313 | 0.0784 |
| 4 | rsi_14 | 117 | 410.0971 | 0.0777 |
| 5 | ma_gap_120 | 143 | 307.1207 | 0.0582 |
| 6 | volatility_20 | 156 | 282.5598 | 0.0535 |
| 7 | macd_signal_to_close | 160 | 276.2814 | 0.0523 |
| 8 | ridge_pred_future_log_return | 185 | 267.7831 | 0.0507 |
| 9 | return_10 | 188 | 239.7399 | 0.0454 |
| 10 | ma_gap_20 | 98 | 153.1429 | 0.0290 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-10-12 |
| Train End | 2024-10-29 |
| Test Start | 2024-11-14 |
| Test End | 2025-02-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6635 |
| Log Loss | 0.6512 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.8772 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6602 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[39, 0], [24, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0326 |
| p25 | 0.1567 |
| median | 0.3178 |
| p75 | 0.5244 |
| max | 0.7462 |
| mean | 0.3414 |
| std | 0.2120 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 300 | 679.3129 | 0.1284 |
| 2 | volatility_60 | 256 | 593.1748 | 0.1121 |
| 3 | macd_signal_to_close | 289 | 582.2250 | 0.1101 |
| 4 | ridge_pred_margin_to_threshold | 96 | 425.5231 | 0.0804 |
| 5 | ridge_pred_future_log_return | 170 | 405.1402 | 0.0766 |
| 6 | return_60 | 153 | 348.1971 | 0.0658 |
| 7 | ma_gap_60 | 130 | 221.3725 | 0.0418 |
| 8 | rsi_14 | 58 | 212.8387 | 0.0402 |
| 9 | ma_gap_120 | 97 | 208.6850 | 0.0394 |
| 10 | ma_gap_5 | 178 | 198.0009 | 0.0374 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-01-16 |
| Train End | 2025-02-03 |
| Test Start | 2025-02-19 |
| Test End | 2025-05-23 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6561 |
| Precision | 0.6304 |
| Recall | 0.8788 |
| F1 | 0.7342 |
| ROC-AUC | 0.6394 |
| Log Loss | 0.8802 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.5884 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6470 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 17], [4, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1132 |
| p25 | 0.5824 |
| median | 0.8008 |
| p75 | 0.9097 |
| max | 0.9863 |
| mean | 0.7170 |
| std | 0.2388 |
| count_ge_threshold | 46 |
| count_lt_threshold | 17 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 219 | 631.8174 | 0.1233 |
| 2 | macd_signal_to_close | 151 | 523.0009 | 0.1021 |
| 3 | ma_gap_120 | 206 | 494.0112 | 0.0964 |
| 4 | ma_gap_60 | 150 | 490.2002 | 0.0957 |
| 5 | volume_change_5 | 207 | 375.7453 | 0.0733 |
| 6 | volatility_20 | 246 | 371.8030 | 0.0726 |
| 7 | return_60 | 218 | 305.1287 | 0.0596 |
| 8 | ema_12_to_ema_26 | 120 | 212.2357 | 0.0414 |
| 9 | return_5 | 134 | 201.9296 | 0.0394 |
| 10 | ridge_pred_margin_to_threshold | 112 | 196.6498 | 0.0384 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-04-17 |
| Train End | 2025-05-09 |
| Test Start | 2025-05-26 |
| Test End | 2025-08-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.5349 |
| Precision | 1.0000 |
| Recall | 0.0698 |
| F1 | 0.1304 |
| ROC-AUC | 0.6802 |
| Log Loss | 0.7605 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.8515 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6463 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [40, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0182 |
| p25 | 0.2225 |
| median | 0.4469 |
| p75 | 0.6119 |
| max | 0.9281 |
| mean | 0.4462 |
| std | 0.2502 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 260 | 575.1299 | 0.1104 |
| 2 | return_60 | 231 | 556.0451 | 0.1068 |
| 3 | volatility_20 | 159 | 484.4458 | 0.0930 |
| 4 | ma_gap_60 | 176 | 446.5914 | 0.0858 |
| 5 | return_20 | 138 | 381.6389 | 0.0733 |
| 6 | ema_12_to_ema_26 | 186 | 315.5227 | 0.0606 |
| 7 | volume_z_20 | 146 | 315.2180 | 0.0605 |
| 8 | macd_signal_to_close | 150 | 280.5231 | 0.0539 |
| 9 | ma_gap_20 | 95 | 214.7164 | 0.0412 |
| 10 | macd_hist_to_close | 154 | 211.5161 | 0.0406 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-19 |
| Train End | 2025-08-07 |
| Test Start | 2025-08-25 |
| Test End | 2025-11-26 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3810 |
| Balanced Accuracy | 0.4202 |
| Precision | 0.6667 |
| Recall | 0.3404 |
| F1 | 0.4507 |
| ROC-AUC | 0.4202 |
| Log Loss | 1.2674 |
| Baseline Accuracy | 0.7460 |
| Decision Threshold | 0.5998 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5443 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 8], [31, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0096 |
| p25 | 0.1231 |
| median | 0.4140 |
| p75 | 0.7370 |
| max | 0.9772 |
| mean | 0.4480 |
| std | 0.3194 |
| count_ge_threshold | 24 |
| count_lt_threshold | 39 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 303 | 713.3314 | 0.1352 |
| 2 | ridge_pred_future_log_return | 245 | 517.2140 | 0.0980 |
| 3 | volatility_60 | 273 | 492.9463 | 0.0934 |
| 4 | macd_signal_to_close | 168 | 444.1751 | 0.0842 |
| 5 | ma_gap_60 | 99 | 399.8382 | 0.0758 |
| 6 | ma_gap_20 | 77 | 343.1124 | 0.0650 |
| 7 | volatility_20 | 188 | 322.1114 | 0.0610 |
| 8 | volume_change_20 | 170 | 170.6659 | 0.0323 |
| 9 | rsi_14 | 58 | 151.8033 | 0.0288 |
| 10 | return_5 | 193 | 149.2356 | 0.0283 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-19 |
| Train End | 2025-11-11 |
| Test Start | 2025-11-27 |
| Test End | 2026-03-03 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6991 |
| Precision | 0.8947 |
| Recall | 0.4722 |
| F1 | 0.6182 |
| ROC-AUC | 0.7479 |
| Log Loss | 1.7310 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.0459 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5336 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 2], [19, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0037 |
| p25 | 0.0095 |
| median | 0.0223 |
| p75 | 0.0629 |
| max | 0.9587 |
| mean | 0.0879 |
| std | 0.1727 |
| count_ge_threshold | 19 |
| count_lt_threshold | 44 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 285 | 1062.8858 | 0.2013 |
| 2 | volatility_60 | 380 | 843.7614 | 0.1598 |
| 3 | ridge_pred_future_log_return | 197 | 323.1465 | 0.0612 |
| 4 | ma_gap_60 | 185 | 303.7170 | 0.0575 |
| 5 | ma_gap_20 | 99 | 266.3430 | 0.0504 |
| 6 | volatility_20 | 191 | 233.1472 | 0.0442 |
| 7 | high_low_range | 128 | 225.4357 | 0.0427 |
| 8 | macd_signal_to_close | 148 | 205.3078 | 0.0389 |
| 9 | return_2 | 147 | 194.8745 | 0.0369 |
| 10 | ma_gap_120 | 110 | 192.5055 | 0.0365 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-24 |
| Train End | 2026-02-16 |
| Test Start | 2026-03-04 |
| Test End | 2026-06-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3175 |
| Balanced Accuracy | 0.4056 |
| Precision | 0.5625 |
| Recall | 0.2000 |
| F1 | 0.2951 |
| ROC-AUC | 0.3926 |
| Log Loss | 2.0982 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.6649 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6172 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 7], [36, 9]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0037 |
| p25 | 0.0205 |
| median | 0.0467 |
| p75 | 0.6278 |
| max | 0.9065 |
| mean | 0.2953 |
| std | 0.3314 |
| count_ge_threshold | 16 |
| count_lt_threshold | 47 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 430 | 816.1061 | 0.1547 |
| 2 | macd_hist_to_close | 159 | 661.8804 | 0.1255 |
| 3 | macd_signal_to_close | 199 | 424.9229 | 0.0806 |
| 4 | return_60 | 139 | 422.7567 | 0.0802 |
| 5 | ma_gap_120 | 190 | 411.0048 | 0.0779 |
| 6 | macd_to_close | 99 | 314.4251 | 0.0596 |
| 7 | return_20 | 183 | 296.1636 | 0.0562 |
| 8 | ema_12_to_ema_26 | 142 | 247.0343 | 0.0468 |
| 9 | volatility_20 | 146 | 191.8184 | 0.0364 |
| 10 | ma_gap_60 | 140 | 174.5079 | 0.0331 |
