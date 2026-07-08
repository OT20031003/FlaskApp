# Direction Validation Report: NVDA

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | NVDA |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-08 15:06:10 NDT |
| Last Date | 2026-07-08 |
| Last Close | 203.4900 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.1858 |
| Probability Down | 0.8142 |
| Decision Threshold | 0.2871 |
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
| fold_count | 29 | N/A | N/A | N/A |
| total_fold_count | 29 | N/A | N/A | N/A |
| beats_baseline_fold_ratio | 0.1724 | N/A | N/A | N/A |
| confusion_matrix_sum | [[459, 290], [588, 490]] | N/A | N/A | N/A |
| accuracy | 0.5194 | 0.1718 | 0.1270 | 0.9206 |
| balanced_accuracy | 0.5629 | 0.0991 | 0.4090 | 0.7956 |
| precision | 0.5098 | 0.3483 | 0.0000 | 1.0000 |
| recall | 0.4747 | 0.3627 | 0.0000 | 1.0000 |
| f1 | 0.4358 | 0.3052 | 0.0000 | 0.9587 |
| roc_auc | 0.5875 | 0.1750 | 0.1273 | 0.8408 |
| log_loss | 1.3967 | 0.7120 | 0.1304 | 3.6228 |
| baseline_accuracy | 0.6639 | 0.1111 | 0.5238 | 0.9841 |
| decision_threshold | 0.6508 | 0.3340 | 0.0321 | 0.9900 |
| calibration_score | 0.5696 | 0.0611 | 0.4833 | 0.7611 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 323 | 811.2479 | 0.1521 |
| 2 | ema_12_to_ema_26 | 108 | 583.9533 | 0.1095 |
| 3 | ma_gap_120 | 216 | 512.0816 | 0.0960 |
| 4 | ma_gap_60 | 114 | 498.0629 | 0.0934 |
| 5 | volatility_60 | 190 | 408.5925 | 0.0766 |
| 6 | return_20 | 167 | 303.1739 | 0.0568 |
| 7 | volume_change_5 | 197 | 277.0256 | 0.0519 |
| 8 | macd_signal_to_close | 193 | 273.1448 | 0.0512 |
| 9 | ridge_pred_future_log_return | 137 | 266.4420 | 0.0500 |
| 10 | macd_hist_to_close | 200 | 220.8180 | 0.0414 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-01-14 | 2019-04-12 | 0.5397 | 0.6705 | 0.6675 | 0.6984 | 0.9184 | ok |
| 2 | ok | 2019-04-15 | 2019-07-15 | 0.8095 | 0.7956 | 0.8408 | 0.5397 | 0.9138 | ok |
| 3 | ok | 2019-07-16 | 2019-10-11 | 0.6190 | 0.5321 | 0.6143 | 0.6190 | 0.0321 | ok |
| 4 | ok | 2019-10-14 | 2020-01-13 | 0.1270 | 0.5000 | 0.4250 | 0.8730 | 0.5000 | ok |
| 5 | ok | 2020-01-14 | 2020-04-14 | 0.4762 | 0.5000 | 0.5778 | 0.5238 | 0.9891 | ok |
| 6 | ok | 2020-04-15 | 2020-07-14 | 0.5556 | 0.6547 | 0.7396 | 0.8413 | 0.3106 | ok |
| 7 | ok | 2020-07-15 | 2020-10-12 | 0.7302 | 0.7321 | 0.7392 | 0.6984 | 0.7484 | ok |
| 8 | ok | 2020-10-13 | 2021-01-12 | 0.3810 | 0.4359 | 0.4754 | 0.6190 | 0.5065 | ok |
| 9 | ok | 2021-01-13 | 2021-04-14 | 0.6667 | 0.6190 | 0.6798 | 0.5873 | 0.5485 | ok |
| 10 | ok | 2021-04-15 | 2021-07-14 | 0.4603 | 0.4090 | 0.2911 | 0.5873 | 0.0782 | ok |
| 11 | ok | 2021-07-15 | 2021-10-12 | 0.4127 | 0.4858 | 0.5295 | 0.6032 | 0.9821 | ok |
| 12 | ok | 2021-10-13 | 2022-01-11 | 0.4444 | 0.4667 | 0.1273 | 0.5238 | 0.9885 | ok |
| 13 | ok | 2022-01-12 | 2022-04-12 | 0.7143 | 0.7225 | 0.7869 | 0.5873 | 0.9083 | ok |
| 14 | ok | 2022-04-13 | 2022-07-14 | 0.3492 | 0.5000 | 0.4169 | 0.6508 | 0.2102 | ok |
| 15 | ok | 2022-07-15 | 2022-10-12 | 0.4921 | 0.5611 | 0.5148 | 0.7143 | 0.8283 | ok |
| 16 | ok | 2022-10-13 | 2023-01-12 | 0.7619 | 0.7750 | 0.8181 | 0.7619 | 0.0687 | ok |
| 17 | ok | 2023-01-13 | 2023-04-14 | 0.2381 | 0.5000 | 0.6194 | 0.7619 | 0.9900 | ok |
| 18 | ok | 2023-04-17 | 2023-07-17 | 0.4921 | 0.6565 | 0.6871 | 0.8254 | 0.6653 | ok |
| 19 | ok | 2023-07-18 | 2023-10-13 | 0.6032 | 0.5000 | 0.5789 | 0.6032 | 0.9121 | ok |
| 20 | ok | 2023-10-16 | 2024-01-16 | 0.3016 | 0.5000 | 0.3062 | 0.6984 | 0.8845 | ok |
| 21 | ok | 2024-01-17 | 2024-04-16 | 0.5556 | 0.5207 | 0.4098 | 0.6349 | 0.3727 | ok |
| 22 | ok | 2024-04-17 | 2024-07-17 | 0.3968 | 0.5383 | 0.7165 | 0.6984 | 0.9849 | ok |
| 23 | ok | 2024-07-18 | 2024-10-15 | 0.3333 | 0.5227 | 0.4725 | 0.6984 | 0.9884 | ok |
| 24 | ok | 2024-10-16 | 2025-01-16 | 0.6349 | 0.5647 | 0.5978 | 0.6349 | 0.9503 | ok |
| 25 | ok | 2025-01-17 | 2025-04-17 | 0.4603 | 0.5750 | 0.8293 | 0.6349 | 0.7208 | ok |
| 26 | ok | 2025-04-21 | 2025-07-21 | 0.9206 | 0.4677 | 0.7742 | 0.9841 | 0.7203 | ok |
| 27 | ok | 2025-07-22 | 2025-10-17 | 0.4762 | 0.5000 | 0.7758 | 0.5238 | 0.8884 | ok |
| 28 | ok | 2025-10-20 | 2026-01-20 | 0.5079 | 0.5231 | 0.4897 | 0.5714 | 0.1431 | ok |
| 29 | ok | 2026-01-21 | 2026-04-21 | 0.6032 | 0.5964 | 0.5357 | 0.5556 | 0.1202 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2016-12-27 |
| Train End | 2018-12-27 |
| Test Start | 2019-01-14 |
| Test End | 2019-04-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.6705 |
| Precision | 1.0000 |
| Recall | 0.3409 |
| F1 | 0.5085 |
| ROC-AUC | 0.6675 |
| Log Loss | 0.7067 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.9184 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6539 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [29, 15]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0438 |
| p25 | 0.3548 |
| median | 0.6481 |
| p75 | 0.9092 |
| max | 0.9937 |
| mean | 0.6135 |
| std | 0.3032 |
| count_ge_threshold | 15 |
| count_lt_threshold | 48 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 284 | 732.9387 | 0.1388 |
| 2 | volatility_20 | 248 | 555.1106 | 0.1051 |
| 3 | macd_signal_to_close | 252 | 445.6676 | 0.0844 |
| 4 | return_60 | 214 | 396.3367 | 0.0751 |
| 5 | ma_gap_60 | 83 | 361.8083 | 0.0685 |
| 6 | ridge_pred_future_log_return | 207 | 355.2165 | 0.0673 |
| 7 | macd_hist_to_close | 190 | 312.2490 | 0.0591 |
| 8 | return_10 | 179 | 263.2133 | 0.0498 |
| 9 | volume_change_20 | 154 | 184.1832 | 0.0349 |
| 10 | return_5 | 110 | 182.9298 | 0.0346 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-03-29 |
| Train End | 2019-03-29 |
| Test Start | 2019-04-15 |
| Test End | 2019-07-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.8095 |
| Balanced Accuracy | 0.7956 |
| Precision | 0.7500 |
| Recall | 0.9706 |
| F1 | 0.8462 |
| ROC-AUC | 0.8408 |
| Log Loss | 0.8642 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9138 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5317 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[18, 11], [1, 33]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0145 |
| p25 | 0.5309 |
| median | 0.9820 |
| p75 | 0.9932 |
| max | 0.9991 |
| mean | 0.7656 |
| std | 0.3625 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 522 | 1221.7389 | 0.2312 |
| 2 | ma_gap_120 | 188 | 489.8904 | 0.0927 |
| 3 | ma_gap_60 | 171 | 448.6406 | 0.0849 |
| 4 | volatility_20 | 208 | 406.7704 | 0.0770 |
| 5 | return_10 | 143 | 358.9906 | 0.0679 |
| 6 | macd_signal_to_close | 179 | 343.1732 | 0.0649 |
| 7 | return_60 | 192 | 272.5923 | 0.0516 |
| 8 | return_20 | 165 | 264.3563 | 0.0500 |
| 9 | volume_change_5 | 95 | 218.6976 | 0.0414 |
| 10 | macd_hist_to_close | 137 | 167.0794 | 0.0316 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-06-28 |
| Train End | 2019-06-28 |
| Test Start | 2019-07-16 |
| Test End | 2019-10-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5321 |
| Precision | 0.6364 |
| Recall | 0.8974 |
| F1 | 0.7447 |
| ROC-AUC | 0.6143 |
| Log Loss | 0.9938 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.0321 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5306 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 20], [4, 35]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0056 |
| p25 | 0.1659 |
| median | 0.4597 |
| p75 | 0.7583 |
| max | 0.9840 |
| mean | 0.4605 |
| std | 0.3330 |
| count_ge_threshold | 55 |
| count_lt_threshold | 8 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 265 | 710.9595 | 0.1378 |
| 2 | return_20 | 258 | 437.1404 | 0.0848 |
| 3 | macd_signal_to_close | 261 | 426.2990 | 0.0827 |
| 4 | return_5 | 143 | 397.9234 | 0.0771 |
| 5 | volatility_20 | 151 | 327.1006 | 0.0634 |
| 6 | volume_z_20 | 106 | 277.9039 | 0.0539 |
| 7 | ma_gap_60 | 145 | 267.2473 | 0.0518 |
| 8 | volatility_60 | 160 | 254.8571 | 0.0494 |
| 9 | close_to_ema_26 | 67 | 246.9855 | 0.0479 |
| 10 | return_10 | 146 | 241.8492 | 0.0469 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-09-27 |
| Train End | 2019-09-27 |
| Test Start | 2019-10-14 |
| Test End | 2020-01-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.1270 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.4250 |
| Log Loss | 3.6228 |
| Baseline Accuracy | 0.8730 |
| Decision Threshold | 0.5000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6394 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 0], [55, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0019 |
| p25 | 0.0077 |
| median | 0.0162 |
| p75 | 0.0308 |
| max | 0.2428 |
| mean | 0.0291 |
| std | 0.0419 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 429 | 1064.1775 | 0.1995 |
| 2 | macd_signal_to_close | 204 | 690.1760 | 0.1294 |
| 3 | volatility_60 | 275 | 666.3581 | 0.1249 |
| 4 | return_20 | 277 | 562.8616 | 0.1055 |
| 5 | volatility_20 | 188 | 367.3608 | 0.0689 |
| 6 | return_5 | 151 | 282.9895 | 0.0530 |
| 7 | macd_hist_to_close | 122 | 199.4447 | 0.0374 |
| 8 | return_60 | 101 | 191.7352 | 0.0359 |
| 9 | volume_z_20 | 130 | 165.4016 | 0.0310 |
| 10 | ridge_pred_future_log_return | 114 | 108.2276 | 0.0203 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-12-27 |
| Train End | 2019-12-27 |
| Test Start | 2020-01-14 |
| Test End | 2020-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5778 |
| Log Loss | 1.5520 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.9891 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4938 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 0], [33, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0056 |
| p25 | 0.0187 |
| median | 0.0420 |
| p75 | 0.0832 |
| max | 0.6545 |
| mean | 0.1110 |
| std | 0.1635 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 360 | 983.4899 | 0.1849 |
| 2 | return_20 | 267 | 612.5532 | 0.1152 |
| 3 | volatility_20 | 314 | 587.5656 | 0.1105 |
| 4 | volatility_60 | 164 | 492.6314 | 0.0926 |
| 5 | return_5 | 372 | 365.7521 | 0.0688 |
| 6 | macd_signal_to_close | 66 | 271.1271 | 0.0510 |
| 7 | rsi_14 | 132 | 245.5900 | 0.0462 |
| 8 | volume_change_5 | 225 | 198.8339 | 0.0374 |
| 9 | ma_gap_20 | 63 | 182.3293 | 0.0343 |
| 10 | volume_change_20 | 92 | 173.7232 | 0.0327 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-03-29 |
| Train End | 2020-03-30 |
| Test Start | 2020-04-15 |
| Test End | 2020-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.6547 |
| Precision | 0.9310 |
| Recall | 0.5094 |
| F1 | 0.6585 |
| ROC-AUC | 0.7396 |
| Log Loss | 1.1678 |
| Baseline Accuracy | 0.8413 |
| Decision Threshold | 0.3106 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5843 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 2], [26, 27]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0100 |
| p25 | 0.1061 |
| median | 0.3003 |
| p75 | 0.4915 |
| max | 0.7449 |
| mean | 0.3121 |
| std | 0.1971 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 463 | 992.7581 | 0.1840 |
| 2 | ma_gap_120 | 256 | 855.4091 | 0.1585 |
| 3 | volatility_60 | 347 | 657.4268 | 0.1218 |
| 4 | ma_gap_20 | 119 | 293.4661 | 0.0544 |
| 5 | return_20 | 112 | 255.9573 | 0.0474 |
| 6 | rsi_14 | 126 | 254.9802 | 0.0472 |
| 7 | volume_change_20 | 107 | 223.1984 | 0.0414 |
| 8 | close_to_ema_26 | 109 | 178.8164 | 0.0331 |
| 9 | return_5 | 162 | 172.6534 | 0.0320 |
| 10 | macd_hist_to_close | 90 | 170.1513 | 0.0315 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-06-28 |
| Train End | 2020-06-29 |
| Test Start | 2020-07-15 |
| Test End | 2020-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7302 |
| Balanced Accuracy | 0.7321 |
| Precision | 0.8649 |
| Recall | 0.7273 |
| F1 | 0.7901 |
| ROC-AUC | 0.7392 |
| Log Loss | 0.5682 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.7484 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5809 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 5], [12, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0964 |
| p25 | 0.4904 |
| median | 0.7931 |
| p75 | 0.9006 |
| max | 0.9850 |
| mean | 0.6905 |
| std | 0.2503 |
| count_ge_threshold | 37 |
| count_lt_threshold | 26 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 312 | 797.8912 | 0.1484 |
| 2 | ma_gap_120 | 333 | 644.1843 | 0.1198 |
| 3 | ma_gap_20 | 217 | 521.6749 | 0.0970 |
| 4 | volatility_60 | 248 | 481.6656 | 0.0896 |
| 5 | macd_hist_to_close | 159 | 406.3454 | 0.0756 |
| 6 | macd_signal_to_close | 180 | 326.6442 | 0.0608 |
| 7 | return_60 | 136 | 253.0069 | 0.0471 |
| 8 | macd_to_close | 93 | 187.9915 | 0.0350 |
| 9 | ridge_pred_future_log_return | 150 | 184.0840 | 0.0342 |
| 10 | close_to_ema_12 | 88 | 180.8534 | 0.0336 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-09-27 |
| Train End | 2020-09-28 |
| Test Start | 2020-10-13 |
| Test End | 2021-01-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3810 |
| Balanced Accuracy | 0.4359 |
| Precision | 0.3404 |
| Recall | 0.6667 |
| F1 | 0.4507 |
| ROC-AUC | 0.4754 |
| Log Loss | 1.3199 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.5065 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6103 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 31], [8, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0395 |
| p25 | 0.4926 |
| median | 0.7510 |
| p75 | 0.9414 |
| max | 0.9945 |
| mean | 0.6853 |
| std | 0.2766 |
| count_ge_threshold | 47 |
| count_lt_threshold | 16 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 359 | 850.9678 | 0.1589 |
| 2 | volatility_60 | 373 | 579.5619 | 0.1082 |
| 3 | ridge_pred_margin_to_threshold | 110 | 478.3204 | 0.0893 |
| 4 | volatility_20 | 237 | 423.5838 | 0.0791 |
| 5 | volume_change_20 | 152 | 415.0479 | 0.0775 |
| 6 | return_60 | 174 | 403.6184 | 0.0753 |
| 7 | macd_hist_to_close | 132 | 294.0561 | 0.0549 |
| 8 | ridge_pred_future_log_return | 182 | 229.4215 | 0.0428 |
| 9 | ema_12_to_ema_26 | 46 | 184.6656 | 0.0345 |
| 10 | close_to_ema_26 | 45 | 166.5258 | 0.0311 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-12-28 |
| Train End | 2020-12-28 |
| Test Start | 2021-01-13 |
| Test End | 2021-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6190 |
| Precision | 0.6600 |
| Recall | 0.8919 |
| F1 | 0.7586 |
| ROC-AUC | 0.6798 |
| Log Loss | 1.0162 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.5485 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5196 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 17], [4, 33]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0361 |
| p25 | 0.6416 |
| median | 0.9540 |
| p75 | 0.9858 |
| max | 0.9975 |
| mean | 0.7792 |
| std | 0.2959 |
| count_ge_threshold | 50 |
| count_lt_threshold | 13 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 251 | 596.5205 | 0.1099 |
| 2 | volume_change_20 | 124 | 572.2263 | 0.1055 |
| 3 | macd_signal_to_close | 258 | 558.4201 | 0.1029 |
| 4 | volatility_60 | 258 | 545.5849 | 0.1006 |
| 5 | rsi_14 | 94 | 380.0979 | 0.0701 |
| 6 | ridge_pred_future_log_return | 142 | 260.7642 | 0.0481 |
| 7 | volatility_20 | 217 | 254.7196 | 0.0469 |
| 8 | macd_to_close | 81 | 220.6256 | 0.0407 |
| 9 | return_10 | 83 | 200.8105 | 0.0370 |
| 10 | return_60 | 178 | 190.6271 | 0.0351 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-04-01 |
| Train End | 2021-03-30 |
| Test Start | 2021-04-15 |
| Test End | 2021-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.4090 |
| Precision | 0.5306 |
| Recall | 0.7027 |
| F1 | 0.6047 |
| ROC-AUC | 0.2911 |
| Log Loss | 2.3678 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.0782 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[3, 23], [11, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0129 |
| p25 | 0.1341 |
| median | 0.6631 |
| p75 | 0.9776 |
| max | 0.9998 |
| mean | 0.5801 |
| std | 0.3912 |
| count_ge_threshold | 49 |
| count_lt_threshold | 14 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volume_change_20 | 156 | 794.6537 | 0.1469 |
| 2 | volatility_60 | 299 | 621.0654 | 0.1148 |
| 3 | ridge_pred_margin_to_threshold | 141 | 492.2341 | 0.0910 |
| 4 | rsi_14 | 80 | 442.4743 | 0.0818 |
| 5 | return_60 | 217 | 361.7666 | 0.0669 |
| 6 | close_to_ema_26 | 51 | 300.8386 | 0.0556 |
| 7 | macd_signal_to_close | 244 | 291.4047 | 0.0539 |
| 8 | ma_gap_60 | 47 | 248.4794 | 0.0459 |
| 9 | ridge_pred_future_log_return | 139 | 209.0116 | 0.0386 |
| 10 | return_10 | 85 | 186.9830 | 0.0346 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-07-01 |
| Train End | 2021-06-29 |
| Test Start | 2021-07-15 |
| Test End | 2021-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4858 |
| Precision | 0.5556 |
| Recall | 0.1316 |
| F1 | 0.2128 |
| ROC-AUC | 0.5295 |
| Log Loss | 1.1438 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.9821 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5088 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 4], [33, 5]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0153 |
| p25 | 0.6633 |
| median | 0.9160 |
| p75 | 0.9712 |
| max | 0.9985 |
| mean | 0.7354 |
| std | 0.3233 |
| count_ge_threshold | 9 |
| count_lt_threshold | 54 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 346 | 815.7812 | 0.1548 |
| 2 | volatility_60 | 344 | 787.3366 | 0.1494 |
| 3 | macd_hist_to_close | 149 | 459.9259 | 0.0873 |
| 4 | volume_change_20 | 234 | 371.6930 | 0.0705 |
| 5 | return_10 | 162 | 303.7056 | 0.0576 |
| 6 | ridge_pred_future_log_return | 157 | 234.2002 | 0.0445 |
| 7 | volatility_20 | 121 | 227.9648 | 0.0433 |
| 8 | ma_gap_20 | 118 | 208.1894 | 0.0395 |
| 9 | return_2 | 179 | 179.7455 | 0.0341 |
| 10 | return_20 | 121 | 154.1715 | 0.0293 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-09-30 |
| Train End | 2021-09-28 |
| Test Start | 2021-10-13 |
| Test End | 2022-01-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.4667 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.1273 |
| Log Loss | 2.3618 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.9885 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5068 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 2], [33, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0079 |
| p25 | 0.0524 |
| median | 0.2977 |
| p75 | 0.9003 |
| max | 0.9899 |
| mean | 0.4562 |
| std | 0.3971 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 430 | 1043.2373 | 0.2014 |
| 2 | volatility_60 | 281 | 430.3700 | 0.0831 |
| 3 | volatility_20 | 160 | 333.4608 | 0.0644 |
| 4 | ma_gap_60 | 136 | 312.2327 | 0.0603 |
| 5 | return_10 | 132 | 280.9478 | 0.0542 |
| 6 | volume_z_20 | 164 | 277.6664 | 0.0536 |
| 7 | ma_gap_120 | 66 | 248.9562 | 0.0481 |
| 8 | ridge_pred_future_log_return | 136 | 197.9001 | 0.0382 |
| 9 | return_60 | 114 | 194.2402 | 0.0375 |
| 10 | rsi_14 | 51 | 182.1920 | 0.0352 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-12-30 |
| Train End | 2021-12-28 |
| Test Start | 2022-01-12 |
| Test End | 2022-04-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7143 |
| Balanced Accuracy | 0.7225 |
| Precision | 0.6250 |
| Recall | 0.7692 |
| F1 | 0.6897 |
| ROC-AUC | 0.7869 |
| Log Loss | 0.9810 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.9083 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5690 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 12], [6, 20]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0241 |
| p25 | 0.6724 |
| median | 0.9129 |
| p75 | 0.9590 |
| max | 0.9924 |
| mean | 0.7484 |
| std | 0.2983 |
| count_ge_threshold | 32 |
| count_lt_threshold | 31 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 260 | 563.9222 | 0.1092 |
| 2 | return_60 | 176 | 524.7112 | 0.1016 |
| 3 | ridge_pred_future_log_return | 143 | 456.4781 | 0.0884 |
| 4 | ma_gap_120 | 218 | 375.3771 | 0.0727 |
| 5 | volatility_60 | 223 | 288.9778 | 0.0560 |
| 6 | volume_z_20 | 194 | 264.7581 | 0.0513 |
| 7 | return_20 | 138 | 244.7746 | 0.0474 |
| 8 | close_to_ema_26 | 74 | 231.2581 | 0.0448 |
| 9 | macd_signal_to_close | 170 | 221.1363 | 0.0428 |
| 10 | volume_change_5 | 140 | 178.5887 | 0.0346 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-31 |
| Train End | 2022-03-29 |
| Test Start | 2022-04-13 |
| Test End | 2022-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.3492 |
| Recall | 1.0000 |
| F1 | 0.5176 |
| ROC-AUC | 0.4169 |
| Log Loss | 1.9029 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.2102 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5530 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 41], [0, 22]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.4340 |
| p25 | 0.8741 |
| median | 0.9513 |
| p75 | 0.9707 |
| max | 0.9875 |
| mean | 0.8835 |
| std | 0.1479 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 297 | 672.2006 | 0.1292 |
| 2 | return_60 | 219 | 416.1886 | 0.0800 |
| 3 | macd_signal_to_close | 235 | 400.6153 | 0.0770 |
| 4 | volatility_20 | 199 | 373.3390 | 0.0718 |
| 5 | return_20 | 139 | 365.4435 | 0.0703 |
| 6 | volatility_60 | 248 | 306.7417 | 0.0590 |
| 7 | macd_to_close | 107 | 269.9872 | 0.0519 |
| 8 | ridge_pred_margin_to_threshold | 103 | 207.1983 | 0.0398 |
| 9 | close_to_ema_12 | 86 | 183.1721 | 0.0352 |
| 10 | ma_gap_60 | 144 | 171.7076 | 0.0330 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-30 |
| Train End | 2022-06-29 |
| Test Start | 2022-07-15 |
| Test End | 2022-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.5611 |
| Precision | 0.3250 |
| Recall | 0.7222 |
| F1 | 0.4483 |
| ROC-AUC | 0.5148 |
| Log Loss | 1.6050 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.8283 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6298 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[18, 27], [5, 13]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0073 |
| p25 | 0.3548 |
| median | 0.9031 |
| p75 | 0.9639 |
| max | 0.9963 |
| mean | 0.6863 |
| std | 0.3748 |
| count_ge_threshold | 40 |
| count_lt_threshold | 23 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 236 | 675.7812 | 0.1286 |
| 2 | return_60 | 257 | 572.8672 | 0.1090 |
| 3 | return_20 | 172 | 453.1971 | 0.0862 |
| 4 | ma_gap_120 | 143 | 436.2943 | 0.0830 |
| 5 | volatility_20 | 268 | 433.4462 | 0.0825 |
| 6 | volatility_60 | 206 | 333.3516 | 0.0634 |
| 7 | rsi_14 | 68 | 295.4629 | 0.0562 |
| 8 | macd_to_close | 82 | 265.1479 | 0.0504 |
| 9 | ridge_pred_future_log_return | 164 | 196.3907 | 0.0374 |
| 10 | macd_hist_to_close | 122 | 191.1860 | 0.0364 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-29 |
| Train End | 2022-09-28 |
| Test Start | 2022-10-13 |
| Test End | 2023-01-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7619 |
| Balanced Accuracy | 0.7750 |
| Precision | 0.9231 |
| Recall | 0.7500 |
| F1 | 0.8276 |
| ROC-AUC | 0.8181 |
| Log Loss | 1.5233 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.0687 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5778 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 3], [12, 36]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0004 |
| p25 | 0.0095 |
| median | 0.2168 |
| p75 | 0.5523 |
| max | 0.9356 |
| mean | 0.3193 |
| std | 0.3196 |
| count_ge_threshold | 39 |
| count_lt_threshold | 24 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 247 | 969.0598 | 0.1830 |
| 2 | volatility_60 | 372 | 939.6694 | 0.1774 |
| 3 | ma_gap_120 | 212 | 638.4053 | 0.1205 |
| 4 | return_20 | 234 | 420.6681 | 0.0794 |
| 5 | ma_gap_20 | 86 | 418.7952 | 0.0791 |
| 6 | return_60 | 170 | 252.7670 | 0.0477 |
| 7 | macd_hist_to_close | 178 | 205.1467 | 0.0387 |
| 8 | volatility_20 | 116 | 178.0668 | 0.0336 |
| 9 | return_5 | 160 | 172.6518 | 0.0326 |
| 10 | volume_z_20 | 152 | 154.3246 | 0.0291 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-29 |
| Train End | 2022-12-28 |
| Test Start | 2023-01-13 |
| Test End | 2023-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2381 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6194 |
| Log Loss | 2.8196 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.9900 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4833 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 0], [48, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0006 |
| p25 | 0.0063 |
| median | 0.0216 |
| p75 | 0.0847 |
| max | 0.5475 |
| mean | 0.0709 |
| std | 0.1143 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 518 | 1053.3035 | 0.2020 |
| 2 | volatility_20 | 245 | 869.4484 | 0.1668 |
| 3 | macd_signal_to_close | 234 | 407.7271 | 0.0782 |
| 4 | ma_gap_120 | 156 | 279.4249 | 0.0536 |
| 5 | ma_gap_20 | 127 | 224.5105 | 0.0431 |
| 6 | return_20 | 123 | 213.4985 | 0.0410 |
| 7 | macd_hist_to_close | 151 | 177.7863 | 0.0341 |
| 8 | return_5 | 116 | 175.2095 | 0.0336 |
| 9 | ridge_pred_future_log_return | 106 | 173.3537 | 0.0333 |
| 10 | return_60 | 129 | 166.4093 | 0.0319 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-31 |
| Train End | 2023-03-30 |
| Test Start | 2023-04-17 |
| Test End | 2023-07-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.6565 |
| Precision | 0.9545 |
| Recall | 0.4038 |
| F1 | 0.5676 |
| ROC-AUC | 0.6871 |
| Log Loss | 0.9253 |
| Baseline Accuracy | 0.8254 |
| Decision Threshold | 0.6653 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5470 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[10, 1], [31, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0222 |
| p25 | 0.1309 |
| median | 0.5156 |
| p75 | 0.8495 |
| max | 0.9962 |
| mean | 0.4985 |
| std | 0.3440 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 421 | 1100.1959 | 0.2106 |
| 2 | volatility_60 | 259 | 684.8510 | 0.1311 |
| 3 | return_60 | 180 | 470.3797 | 0.0901 |
| 4 | return_20 | 216 | 380.3430 | 0.0728 |
| 5 | ma_gap_120 | 203 | 360.4870 | 0.0690 |
| 6 | ridge_pred_future_log_return | 195 | 315.5918 | 0.0604 |
| 7 | macd_signal_to_close | 109 | 267.5114 | 0.0512 |
| 8 | return_5 | 179 | 250.2429 | 0.0479 |
| 9 | volume_change_20 | 183 | 215.9781 | 0.0414 |
| 10 | return_10 | 163 | 168.8521 | 0.0323 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-30 |
| Train End | 2023-06-30 |
| Test Start | 2023-07-18 |
| Test End | 2023-10-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5789 |
| Log Loss | 1.3494 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.9121 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5149 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[38, 0], [25, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0011 |
| p25 | 0.0042 |
| median | 0.0146 |
| p75 | 0.1810 |
| max | 0.7823 |
| mean | 0.1319 |
| std | 0.2009 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 264 | 825.3247 | 0.1594 |
| 2 | ridge_pred_future_log_return | 289 | 742.9085 | 0.1435 |
| 3 | ma_gap_120 | 288 | 566.6324 | 0.1094 |
| 4 | volatility_20 | 208 | 487.7708 | 0.0942 |
| 5 | macd_signal_to_close | 148 | 274.7839 | 0.0531 |
| 6 | volume_change_20 | 159 | 239.4344 | 0.0462 |
| 7 | macd_hist_to_close | 168 | 238.5902 | 0.0461 |
| 8 | volume_z_20 | 163 | 225.1685 | 0.0435 |
| 9 | volatility_60 | 164 | 202.5873 | 0.0391 |
| 10 | ridge_pred_future_return | 58 | 189.8178 | 0.0367 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-29 |
| Train End | 2023-09-29 |
| Test Start | 2023-10-16 |
| Test End | 2024-01-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3016 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.3062 |
| Log Loss | 2.1335 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.8845 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5204 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [44, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0033 |
| p25 | 0.0166 |
| median | 0.1044 |
| p75 | 0.3461 |
| max | 0.7560 |
| mean | 0.2041 |
| std | 0.2251 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 251 | 696.6069 | 0.1325 |
| 2 | macd_hist_to_close | 185 | 624.8146 | 0.1189 |
| 3 | ma_gap_120 | 206 | 571.8953 | 0.1088 |
| 4 | volatility_20 | 258 | 515.0569 | 0.0980 |
| 5 | ridge_pred_future_log_return | 203 | 433.7625 | 0.0825 |
| 6 | macd_signal_to_close | 251 | 419.8701 | 0.0799 |
| 7 | volatility_60 | 211 | 386.4060 | 0.0735 |
| 8 | volume_change_20 | 159 | 225.1997 | 0.0428 |
| 9 | volume_z_20 | 154 | 180.0564 | 0.0343 |
| 10 | ma_gap_60 | 109 | 140.5153 | 0.0267 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-29 |
| Train End | 2023-12-29 |
| Test Start | 2024-01-17 |
| Test End | 2024-04-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5207 |
| Precision | 0.6500 |
| Recall | 0.6500 |
| F1 | 0.6500 |
| ROC-AUC | 0.4098 |
| Log Loss | 1.2474 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.3727 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5766 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 14], [14, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0068 |
| p25 | 0.2516 |
| median | 0.5962 |
| p75 | 0.8990 |
| max | 0.9941 |
| mean | 0.5627 |
| std | 0.3354 |
| count_ge_threshold | 40 |
| count_lt_threshold | 23 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 226 | 712.3529 | 0.1341 |
| 2 | ma_gap_120 | 213 | 486.6106 | 0.0916 |
| 3 | rsi_14 | 180 | 425.1134 | 0.0800 |
| 4 | macd_hist_to_close | 260 | 364.6209 | 0.0686 |
| 5 | return_60 | 203 | 361.4645 | 0.0680 |
| 6 | volatility_20 | 293 | 315.1185 | 0.0593 |
| 7 | return_20 | 140 | 314.6758 | 0.0592 |
| 8 | macd_signal_to_close | 128 | 256.8451 | 0.0483 |
| 9 | ma_gap_60 | 99 | 251.1882 | 0.0473 |
| 10 | return_10 | 167 | 231.5371 | 0.0436 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-03-30 |
| Train End | 2024-04-02 |
| Test Start | 2024-04-17 |
| Test End | 2024-07-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.5383 |
| Precision | 0.8000 |
| Recall | 0.1818 |
| F1 | 0.2963 |
| ROC-AUC | 0.7165 |
| Log Loss | 0.6470 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.9849 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5286 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 2], [36, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1312 |
| p25 | 0.7058 |
| median | 0.9026 |
| p75 | 0.9727 |
| max | 0.9985 |
| mean | 0.7952 |
| std | 0.2427 |
| count_ge_threshold | 10 |
| count_lt_threshold | 53 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 351 | 1065.4525 | 0.2023 |
| 2 | ridge_pred_future_log_return | 210 | 660.4352 | 0.1254 |
| 3 | macd_signal_to_close | 211 | 519.0386 | 0.0986 |
| 4 | return_10 | 211 | 242.5870 | 0.0461 |
| 5 | rsi_14 | 99 | 235.0126 | 0.0446 |
| 6 | macd_hist_to_close | 141 | 215.4083 | 0.0409 |
| 7 | return_60 | 62 | 214.2480 | 0.0407 |
| 8 | return_20 | 84 | 207.9758 | 0.0395 |
| 9 | ma_gap_5 | 137 | 201.6580 | 0.0383 |
| 10 | ma_gap_60 | 115 | 176.3519 | 0.0335 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-06-30 |
| Train End | 2024-07-02 |
| Test Start | 2024-07-18 |
| Test End | 2024-10-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3333 |
| Balanced Accuracy | 0.5227 |
| Precision | 1.0000 |
| Recall | 0.0455 |
| F1 | 0.0870 |
| ROC-AUC | 0.4725 |
| Log Loss | 1.0447 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.9884 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5269 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [42, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0110 |
| p25 | 0.5354 |
| median | 0.8162 |
| p75 | 0.9473 |
| max | 0.9984 |
| mean | 0.7021 |
| std | 0.2985 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_hist_to_close | 291 | 695.4667 | 0.1310 |
| 2 | return_60 | 263 | 520.5890 | 0.0980 |
| 3 | volatility_20 | 245 | 505.7603 | 0.0952 |
| 4 | ma_gap_120 | 158 | 426.9874 | 0.0804 |
| 5 | ridge_pred_future_log_return | 188 | 380.2976 | 0.0716 |
| 6 | return_20 | 119 | 331.0950 | 0.0624 |
| 7 | macd_signal_to_close | 158 | 280.5238 | 0.0528 |
| 8 | ridge_pred_margin_to_threshold | 126 | 254.4617 | 0.0479 |
| 9 | ma_gap_60 | 150 | 220.5312 | 0.0415 |
| 10 | close_to_high | 170 | 165.3855 | 0.0311 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-09-29 |
| Train End | 2024-10-01 |
| Test Start | 2024-10-16 |
| Test End | 2025-01-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.5647 |
| Precision | 0.5000 |
| Recall | 0.3043 |
| F1 | 0.3784 |
| ROC-AUC | 0.5978 |
| Log Loss | 1.1162 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.9503 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6005 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[33, 7], [16, 7]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0693 |
| p25 | 0.4303 |
| median | 0.7579 |
| p75 | 0.9283 |
| max | 0.9984 |
| mean | 0.6531 |
| std | 0.3064 |
| count_ge_threshold | 14 |
| count_lt_threshold | 49 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 214 | 569.4351 | 0.1085 |
| 2 | ma_gap_120 | 238 | 483.9648 | 0.0922 |
| 3 | return_60 | 251 | 472.5021 | 0.0900 |
| 4 | ridge_pred_margin_to_threshold | 124 | 423.5470 | 0.0807 |
| 5 | macd_hist_to_close | 264 | 416.3216 | 0.0793 |
| 6 | ridge_pred_future_log_return | 231 | 399.2048 | 0.0761 |
| 7 | volume_change_20 | 158 | 289.7256 | 0.0552 |
| 8 | ema_12_to_ema_26 | 89 | 258.6523 | 0.0493 |
| 9 | macd_to_close | 109 | 205.1977 | 0.0391 |
| 10 | volatility_60 | 95 | 203.5545 | 0.0388 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-12-29 |
| Train End | 2024-12-31 |
| Test Start | 2025-01-17 |
| Test End | 2025-04-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.5750 |
| Precision | 0.4035 |
| Recall | 1.0000 |
| F1 | 0.5750 |
| ROC-AUC | 0.8293 |
| Log Loss | 1.4286 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.7208 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6381 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 34], [0, 23]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0585 |
| p25 | 0.8358 |
| median | 0.9402 |
| p75 | 0.9773 |
| max | 0.9947 |
| mean | 0.8675 |
| std | 0.1841 |
| count_ge_threshold | 57 |
| count_lt_threshold | 6 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 254 | 837.8764 | 0.1579 |
| 2 | return_60 | 334 | 734.6586 | 0.1385 |
| 3 | ridge_pred_future_log_return | 186 | 533.8250 | 0.1006 |
| 4 | macd_to_close | 169 | 414.6840 | 0.0782 |
| 5 | volatility_60 | 197 | 366.2396 | 0.0690 |
| 6 | ridge_pred_margin_to_threshold | 98 | 316.9947 | 0.0597 |
| 7 | ema_12_to_ema_26 | 154 | 314.2310 | 0.0592 |
| 8 | volatility_20 | 154 | 292.8768 | 0.0552 |
| 9 | macd_signal_to_close | 76 | 209.6865 | 0.0395 |
| 10 | volume_z_20 | 115 | 209.1213 | 0.0394 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-03-31 |
| Train End | 2025-04-03 |
| Test Start | 2025-04-21 |
| Test End | 2025-07-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.9206 |
| Balanced Accuracy | 0.4677 |
| Precision | 0.9831 |
| Recall | 0.9355 |
| F1 | 0.9587 |
| ROC-AUC | 0.7742 |
| Log Loss | 0.1304 |
| Baseline Accuracy | 0.9841 |
| Decision Threshold | 0.7203 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5913 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 1], [4, 58]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.4424 |
| p25 | 0.9064 |
| median | 0.9718 |
| p75 | 0.9823 |
| max | 0.9993 |
| mean | 0.9199 |
| std | 0.1189 |
| count_ge_threshold | 59 |
| count_lt_threshold | 4 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 280 | 970.7030 | 0.1794 |
| 2 | ma_gap_120 | 262 | 826.1119 | 0.1527 |
| 3 | volatility_60 | 296 | 521.4251 | 0.0964 |
| 4 | ema_12_to_ema_26 | 141 | 351.5309 | 0.0650 |
| 5 | return_60 | 311 | 314.1587 | 0.0581 |
| 6 | macd_hist_to_close | 197 | 313.3451 | 0.0579 |
| 7 | ridge_pred_future_return | 60 | 257.0780 | 0.0475 |
| 8 | macd_to_close | 87 | 236.8804 | 0.0438 |
| 9 | volatility_20 | 150 | 219.5050 | 0.0406 |
| 10 | ridge_pred_margin_to_threshold | 74 | 190.1907 | 0.0351 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-03 |
| Train End | 2025-07-07 |
| Test Start | 2025-07-22 |
| Test End | 2025-10-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.7758 |
| Log Loss | 1.7398 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.8884 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5881 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 0], [33, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0034 |
| p25 | 0.0082 |
| median | 0.0190 |
| p75 | 0.0530 |
| max | 0.3238 |
| mean | 0.0449 |
| std | 0.0645 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 318 | 1163.0638 | 0.2143 |
| 2 | volatility_20 | 178 | 538.1711 | 0.0991 |
| 3 | ma_gap_60 | 107 | 498.8962 | 0.0919 |
| 4 | macd_signal_to_close | 226 | 403.3087 | 0.0743 |
| 5 | volatility_60 | 323 | 381.5567 | 0.0703 |
| 6 | macd_to_close | 143 | 372.6823 | 0.0687 |
| 7 | ema_12_to_ema_26 | 143 | 310.3934 | 0.0572 |
| 8 | ridge_pred_future_return | 66 | 287.7752 | 0.0530 |
| 9 | high_low_range | 163 | 173.8677 | 0.0320 |
| 10 | ma_gap_120 | 142 | 160.5032 | 0.0296 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-02 |
| Train End | 2025-10-03 |
| Test Start | 2025-10-20 |
| Test End | 2026-01-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.5231 |
| Precision | 0.4474 |
| Recall | 0.6296 |
| F1 | 0.5231 |
| ROC-AUC | 0.4897 |
| Log Loss | 1.2166 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.1431 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7611 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 21], [10, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0062 |
| p25 | 0.0687 |
| median | 0.3658 |
| p75 | 0.7963 |
| max | 0.9482 |
| mean | 0.4139 |
| std | 0.3513 |
| count_ge_threshold | 38 |
| count_lt_threshold | 25 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 328 | 946.1816 | 0.1809 |
| 2 | return_60 | 154 | 373.2374 | 0.0713 |
| 3 | ma_gap_120 | 246 | 356.6422 | 0.0682 |
| 4 | macd_hist_to_close | 157 | 332.2026 | 0.0635 |
| 5 | high_low_range | 140 | 318.0080 | 0.0608 |
| 6 | volatility_60 | 251 | 306.4893 | 0.0586 |
| 7 | ma_gap_20 | 69 | 298.1682 | 0.0570 |
| 8 | ema_12_to_ema_26 | 95 | 270.7211 | 0.0518 |
| 9 | ridge_pred_future_return | 67 | 226.1161 | 0.0432 |
| 10 | volatility_20 | 148 | 216.3407 | 0.0414 |

### Fold 29

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-02 |
| Train End | 2026-01-05 |
| Test Start | 2026-01-21 |
| Test End | 2026-04-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5964 |
| Precision | 0.5556 |
| Recall | 0.5357 |
| F1 | 0.5455 |
| ROC-AUC | 0.5357 |
| Log Loss | 1.0090 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.1202 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6508 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 12], [13, 15]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0100 |
| p25 | 0.0541 |
| median | 0.0901 |
| p75 | 0.2515 |
| max | 0.9066 |
| mean | 0.1957 |
| std | 0.2221 |
| count_ge_threshold | 27 |
| count_lt_threshold | 36 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 247 | 752.1118 | 0.1405 |
| 2 | ma_gap_120 | 445 | 745.1585 | 0.1392 |
| 3 | macd_signal_to_close | 232 | 637.8493 | 0.1192 |
| 4 | volatility_20 | 343 | 561.1009 | 0.1048 |
| 5 | macd_hist_to_close | 195 | 415.9431 | 0.0777 |
| 6 | ma_gap_20 | 87 | 306.4444 | 0.0573 |
| 7 | ridge_pred_future_log_return | 139 | 305.1418 | 0.0570 |
| 8 | return_10 | 77 | 166.5396 | 0.0311 |
| 9 | ma_gap_60 | 114 | 160.2837 | 0.0299 |
| 10 | return_20 | 132 | 153.8261 | 0.0287 |
