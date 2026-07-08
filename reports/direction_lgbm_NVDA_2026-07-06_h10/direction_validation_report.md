# Direction Validation Report: NVDA

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | NVDA |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-07 10:09:21 JST |
| Last Date | 2026-07-06 |
| Last Close | 195.5500 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.0972 |
| Probability Down | 0.9028 |
| Decision Threshold | 0.3710 |
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
| beats_baseline_fold_ratio | 0.2414 | N/A | N/A | N/A |
| confusion_matrix_sum | [[450, 299], [611, 467]] | N/A | N/A | N/A |
| accuracy | 0.5019 | 0.1594 | 0.2381 | 0.8413 |
| balanced_accuracy | 0.5602 | 0.1018 | 0.2905 | 0.8348 |
| precision | 0.5462 | 0.3619 | 0.0000 | 1.0000 |
| recall | 0.4626 | 0.3888 | 0.0000 | 1.0000 |
| f1 | 0.4134 | 0.2980 | 0.0000 | 0.8649 |
| roc_auc | 0.5944 | 0.1717 | 0.1227 | 0.8614 |
| log_loss | 1.3778 | 0.6987 | 0.1496 | 3.5384 |
| baseline_accuracy | 0.6628 | 0.1147 | 0.5079 | 0.9841 |
| decision_threshold | 0.6208 | 0.3539 | 0.0134 | 0.9900 |
| calibration_score | 0.5812 | 0.0625 | 0.4913 | 0.7818 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 321 | 797.2898 | 0.1493 |
| 2 | ema_12_to_ema_26 | 121 | 623.3186 | 0.1167 |
| 3 | ma_gap_60 | 96 | 519.6449 | 0.0973 |
| 4 | ma_gap_120 | 222 | 464.3057 | 0.0870 |
| 5 | volatility_60 | 193 | 388.6951 | 0.0728 |
| 6 | ridge_pred_future_log_return | 139 | 289.1103 | 0.0541 |
| 7 | macd_signal_to_close | 189 | 271.2271 | 0.0508 |
| 8 | return_20 | 153 | 253.1220 | 0.0474 |
| 9 | volume_change_5 | 195 | 246.0992 | 0.0461 |
| 10 | macd_hist_to_close | 190 | 233.6379 | 0.0438 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-01-11 | 2019-04-11 | 0.3968 | 0.5682 | 0.7285 | 0.6984 | 0.9470 | ok |
| 2 | ok | 2019-04-12 | 2019-07-12 | 0.8413 | 0.8348 | 0.8354 | 0.5238 | 0.9152 | ok |
| 3 | ok | 2019-07-15 | 2019-10-10 | 0.6508 | 0.5737 | 0.6667 | 0.6190 | 0.0342 | ok |
| 4 | ok | 2019-10-11 | 2020-01-10 | 0.2540 | 0.5804 | 0.4311 | 0.8889 | 0.0564 | ok |
| 5 | ok | 2020-01-13 | 2020-04-13 | 0.4921 | 0.5000 | 0.6452 | 0.5079 | 0.9086 | ok |
| 6 | ok | 2020-04-14 | 2020-07-13 | 0.6667 | 0.6204 | 0.6564 | 0.8571 | 0.1670 | ok |
| 7 | ok | 2020-07-14 | 2020-10-09 | 0.5556 | 0.6370 | 0.7524 | 0.6984 | 0.8797 | ok |
| 8 | ok | 2020-10-12 | 2021-01-11 | 0.4286 | 0.4744 | 0.4797 | 0.6190 | 0.6127 | ok |
| 9 | ok | 2021-01-12 | 2021-04-13 | 0.6032 | 0.5192 | 0.6455 | 0.5873 | 0.0452 | ok |
| 10 | ok | 2021-04-14 | 2021-07-13 | 0.2540 | 0.2905 | 0.2890 | 0.5873 | 0.9615 | ok |
| 11 | ok | 2021-07-14 | 2021-10-11 | 0.6032 | 0.5192 | 0.5010 | 0.5873 | 0.0134 | ok |
| 12 | ok | 2021-10-12 | 2022-01-10 | 0.4603 | 0.5000 | 0.1227 | 0.5397 | 0.9900 | ok |
| 13 | ok | 2022-01-11 | 2022-04-11 | 0.7302 | 0.7245 | 0.7900 | 0.5873 | 0.9145 | ok |
| 14 | ok | 2022-04-12 | 2022-07-13 | 0.3492 | 0.5119 | 0.5102 | 0.6667 | 0.3410 | ok |
| 15 | ok | 2022-07-14 | 2022-10-11 | 0.5238 | 0.6333 | 0.5519 | 0.7143 | 0.5897 | ok |
| 16 | ok | 2022-10-12 | 2023-01-11 | 0.6667 | 0.7583 | 0.8125 | 0.7619 | 0.2543 | ok |
| 17 | ok | 2023-01-12 | 2023-04-13 | 0.2381 | 0.5000 | 0.6500 | 0.7619 | 0.9865 | ok |
| 18 | ok | 2023-04-14 | 2023-07-14 | 0.4762 | 0.6110 | 0.6661 | 0.8254 | 0.6331 | ok |
| 19 | ok | 2023-07-17 | 2023-10-12 | 0.5873 | 0.5000 | 0.5894 | 0.5873 | 0.8095 | ok |
| 20 | ok | 2023-10-13 | 2024-01-12 | 0.3175 | 0.5000 | 0.3279 | 0.6825 | 0.9164 | ok |
| 21 | ok | 2024-01-16 | 2024-04-15 | 0.5238 | 0.4656 | 0.4290 | 0.6508 | 0.2492 | ok |
| 22 | ok | 2024-04-16 | 2024-07-16 | 0.7778 | 0.6615 | 0.7967 | 0.6984 | 0.4562 | ok |
| 23 | ok | 2024-07-17 | 2024-10-14 | 0.3492 | 0.5233 | 0.5570 | 0.6825 | 0.9880 | ok |
| 24 | ok | 2024-10-15 | 2025-01-15 | 0.6032 | 0.5112 | 0.6218 | 0.6190 | 0.9667 | ok |
| 25 | ok | 2025-01-16 | 2025-04-16 | 0.4603 | 0.5854 | 0.8614 | 0.6508 | 0.7950 | ok |
| 26 | ok | 2025-04-17 | 2025-07-18 | 0.2381 | 0.6129 | 0.5645 | 0.9841 | 0.9871 | ok |
| 27 | ok | 2025-07-21 | 2025-10-16 | 0.4762 | 0.5000 | 0.7596 | 0.5238 | 0.8039 | ok |
| 28 | ok | 2025-10-17 | 2026-01-16 | 0.4444 | 0.4769 | 0.4167 | 0.5714 | 0.0848 | ok |
| 29 | ok | 2026-01-20 | 2026-04-20 | 0.5873 | 0.5517 | 0.5781 | 0.5397 | 0.6960 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2016-12-23 |
| Train End | 2018-12-26 |
| Test Start | 2019-01-11 |
| Test End | 2019-04-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.5682 |
| Precision | 1.0000 |
| Recall | 0.1364 |
| F1 | 0.2400 |
| ROC-AUC | 0.7285 |
| Log Loss | 0.6642 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.9470 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6129 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [38, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0321 |
| p25 | 0.3360 |
| median | 0.5645 |
| p75 | 0.8484 |
| max | 0.9952 |
| mean | 0.5740 |
| std | 0.2953 |
| count_ge_threshold | 6 |
| count_lt_threshold | 57 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 303 | 754.0474 | 0.1430 |
| 2 | volatility_20 | 290 | 747.0257 | 0.1417 |
| 3 | return_60 | 220 | 415.8554 | 0.0789 |
| 4 | macd_signal_to_close | 185 | 304.1042 | 0.0577 |
| 5 | return_10 | 179 | 297.5764 | 0.0564 |
| 6 | ma_gap_60 | 56 | 281.3672 | 0.0534 |
| 7 | macd_hist_to_close | 146 | 238.5024 | 0.0452 |
| 8 | ridge_pred_future_log_return | 208 | 214.1254 | 0.0406 |
| 9 | close_to_ema_12 | 124 | 204.4001 | 0.0388 |
| 10 | close_to_ema_26 | 77 | 166.1208 | 0.0315 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-03-28 |
| Train End | 2019-03-28 |
| Test Start | 2019-04-12 |
| Test End | 2019-07-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.8413 |
| Balanced Accuracy | 0.8348 |
| Precision | 0.7805 |
| Recall | 0.9697 |
| F1 | 0.8649 |
| ROC-AUC | 0.8354 |
| Log Loss | 0.8740 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.9152 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5549 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 9], [1, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0161 |
| p25 | 0.4846 |
| median | 0.9874 |
| p75 | 0.9939 |
| max | 0.9987 |
| mean | 0.7513 |
| std | 0.3725 |
| count_ge_threshold | 41 |
| count_lt_threshold | 22 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 485 | 1047.6267 | 0.1974 |
| 2 | volatility_20 | 244 | 529.0071 | 0.0997 |
| 3 | ma_gap_120 | 234 | 510.9079 | 0.0963 |
| 4 | ma_gap_60 | 181 | 425.2403 | 0.0801 |
| 5 | macd_signal_to_close | 195 | 361.5734 | 0.0681 |
| 6 | return_10 | 126 | 345.7441 | 0.0651 |
| 7 | return_60 | 183 | 319.3974 | 0.0602 |
| 8 | return_20 | 168 | 283.6956 | 0.0535 |
| 9 | volume_change_5 | 101 | 247.2637 | 0.0466 |
| 10 | volume_change_20 | 125 | 126.9950 | 0.0239 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-06-27 |
| Train End | 2019-06-27 |
| Test Start | 2019-07-15 |
| Test End | 2019-10-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.5737 |
| Precision | 0.6604 |
| Recall | 0.8974 |
| F1 | 0.7609 |
| ROC-AUC | 0.6667 |
| Log Loss | 0.9327 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.0342 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5584 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 18], [4, 35]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0051 |
| p25 | 0.1260 |
| median | 0.4735 |
| p75 | 0.7645 |
| max | 0.9913 |
| mean | 0.4618 |
| std | 0.3383 |
| count_ge_threshold | 53 |
| count_lt_threshold | 10 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 221 | 605.3175 | 0.1173 |
| 2 | return_5 | 190 | 564.2379 | 0.1094 |
| 3 | macd_signal_to_close | 295 | 484.7220 | 0.0939 |
| 4 | return_20 | 285 | 436.7728 | 0.0847 |
| 5 | volatility_20 | 173 | 368.1364 | 0.0713 |
| 6 | ridge_pred_future_log_return | 189 | 290.0324 | 0.0562 |
| 7 | volatility_60 | 208 | 254.2827 | 0.0493 |
| 8 | ma_gap_60 | 163 | 250.8995 | 0.0486 |
| 9 | return_10 | 102 | 249.1116 | 0.0483 |
| 10 | volume_z_20 | 88 | 216.9852 | 0.0421 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-09-26 |
| Train End | 2019-09-26 |
| Test Start | 2019-10-11 |
| Test End | 2020-01-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.5804 |
| Precision | 1.0000 |
| Recall | 0.1607 |
| F1 | 0.2769 |
| ROC-AUC | 0.4311 |
| Log Loss | 3.5384 |
| Baseline Accuracy | 0.8889 |
| Decision Threshold | 0.0564 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6455 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 0], [47, 9]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0032 |
| p25 | 0.0091 |
| median | 0.0191 |
| p75 | 0.0322 |
| max | 0.2412 |
| mean | 0.0326 |
| std | 0.0433 |
| count_ge_threshold | 9 |
| count_lt_threshold | 54 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 422 | 1002.5340 | 0.1875 |
| 2 | volatility_60 | 288 | 697.6522 | 0.1305 |
| 3 | macd_signal_to_close | 217 | 679.3328 | 0.1271 |
| 4 | return_20 | 267 | 547.5576 | 0.1024 |
| 5 | volatility_20 | 186 | 367.7167 | 0.0688 |
| 6 | macd_hist_to_close | 152 | 249.0859 | 0.0466 |
| 7 | return_5 | 151 | 222.7726 | 0.0417 |
| 8 | return_60 | 110 | 210.8976 | 0.0394 |
| 9 | volume_z_20 | 137 | 152.2325 | 0.0285 |
| 10 | close_to_high | 93 | 115.5487 | 0.0216 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-12-26 |
| Train End | 2019-12-26 |
| Test Start | 2020-01-13 |
| Test End | 2020-04-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6452 |
| Log Loss | 1.3657 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.9086 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5115 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[31, 0], [32, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0054 |
| p25 | 0.0239 |
| median | 0.0550 |
| p75 | 0.1049 |
| max | 0.6101 |
| mean | 0.1210 |
| std | 0.1647 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 376 | 988.2601 | 0.1855 |
| 2 | volatility_20 | 338 | 643.8538 | 0.1208 |
| 3 | return_20 | 252 | 568.0428 | 0.1066 |
| 4 | volatility_60 | 201 | 528.5231 | 0.0992 |
| 5 | return_5 | 392 | 406.3531 | 0.0763 |
| 6 | macd_signal_to_close | 76 | 286.5440 | 0.0538 |
| 7 | rsi_14 | 109 | 213.2152 | 0.0400 |
| 8 | volume_change_20 | 111 | 194.7692 | 0.0366 |
| 9 | volume_change_5 | 210 | 182.1508 | 0.0342 |
| 10 | ma_gap_20 | 58 | 155.3333 | 0.0292 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-03-28 |
| Train End | 2020-03-27 |
| Test Start | 2020-04-14 |
| Test End | 2020-07-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6204 |
| Precision | 0.9024 |
| Recall | 0.6852 |
| F1 | 0.7789 |
| ROC-AUC | 0.6564 |
| Log Loss | 1.4028 |
| Baseline Accuracy | 0.8571 |
| Decision Threshold | 0.1670 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6157 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 4], [17, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0099 |
| p25 | 0.0940 |
| median | 0.2616 |
| p75 | 0.4007 |
| max | 0.7048 |
| mean | 0.2623 |
| std | 0.1716 |
| count_ge_threshold | 41 |
| count_lt_threshold | 22 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 402 | 890.0508 | 0.1647 |
| 2 | ma_gap_120 | 292 | 879.5863 | 0.1627 |
| 3 | volatility_60 | 359 | 641.0027 | 0.1186 |
| 4 | ma_gap_20 | 155 | 360.5741 | 0.0667 |
| 5 | return_20 | 95 | 240.6169 | 0.0445 |
| 6 | volume_change_20 | 86 | 238.2525 | 0.0441 |
| 7 | macd_signal_to_close | 144 | 233.7035 | 0.0432 |
| 8 | rsi_14 | 135 | 230.2095 | 0.0426 |
| 9 | return_5 | 200 | 205.0254 | 0.0379 |
| 10 | ema_12_to_ema_26 | 97 | 189.0235 | 0.0350 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-06-27 |
| Train End | 2020-06-26 |
| Test Start | 2020-07-14 |
| Test End | 2020-10-09 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.6370 |
| Precision | 0.8636 |
| Recall | 0.4318 |
| F1 | 0.5758 |
| ROC-AUC | 0.7524 |
| Log Loss | 0.5576 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.8797 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5915 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[16, 3], [25, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2014 |
| p25 | 0.6034 |
| median | 0.8327 |
| p75 | 0.9259 |
| max | 0.9898 |
| mean | 0.7550 |
| std | 0.2042 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 332 | 695.2034 | 0.1296 |
| 2 | volatility_20 | 284 | 685.8287 | 0.1278 |
| 3 | volatility_60 | 253 | 487.5699 | 0.0909 |
| 4 | ma_gap_20 | 158 | 471.3680 | 0.0879 |
| 5 | return_60 | 151 | 367.1930 | 0.0684 |
| 6 | macd_hist_to_close | 175 | 356.0725 | 0.0664 |
| 7 | close_to_ema_12 | 121 | 292.8874 | 0.0546 |
| 8 | macd_to_close | 90 | 224.8047 | 0.0419 |
| 9 | rsi_14 | 98 | 192.0431 | 0.0358 |
| 10 | ridge_pred_future_log_return | 161 | 174.8709 | 0.0326 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-09-26 |
| Train End | 2020-09-25 |
| Test Start | 2020-10-12 |
| Test End | 2021-01-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.4744 |
| Precision | 0.3636 |
| Recall | 0.6667 |
| F1 | 0.4706 |
| ROC-AUC | 0.4797 |
| Log Loss | 1.3445 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.6127 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6593 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 28], [8, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0346 |
| p25 | 0.5695 |
| median | 0.7459 |
| p75 | 0.9418 |
| max | 0.9907 |
| mean | 0.7027 |
| std | 0.2738 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 319 | 794.7357 | 0.1482 |
| 2 | ridge_pred_future_log_return | 231 | 592.4766 | 0.1105 |
| 3 | volatility_60 | 369 | 553.4952 | 0.1032 |
| 4 | volume_change_20 | 161 | 464.8011 | 0.0867 |
| 5 | volatility_20 | 212 | 357.3810 | 0.0666 |
| 6 | macd_hist_to_close | 163 | 356.3225 | 0.0664 |
| 7 | ema_12_to_ema_26 | 47 | 323.1339 | 0.0602 |
| 8 | return_60 | 159 | 316.4034 | 0.0590 |
| 9 | close_to_ema_26 | 65 | 175.9775 | 0.0328 |
| 10 | ma_gap_60 | 89 | 141.5076 | 0.0264 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-12-27 |
| Train End | 2020-12-24 |
| Test Start | 2021-01-12 |
| Test End | 2021-04-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5192 |
| Precision | 0.5968 |
| Recall | 1.0000 |
| F1 | 0.7475 |
| ROC-AUC | 0.6455 |
| Log Loss | 1.0689 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.0452 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5132 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 25], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0329 |
| p25 | 0.6819 |
| median | 0.9315 |
| p75 | 0.9854 |
| max | 0.9980 |
| mean | 0.7951 |
| std | 0.2773 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volume_change_20 | 120 | 599.6454 | 0.1106 |
| 2 | macd_signal_to_close | 254 | 564.5702 | 0.1041 |
| 3 | volatility_60 | 246 | 551.0719 | 0.1016 |
| 4 | ma_gap_120 | 268 | 500.1456 | 0.0922 |
| 5 | rsi_14 | 113 | 377.3130 | 0.0696 |
| 6 | ridge_pred_future_log_return | 159 | 328.6567 | 0.0606 |
| 7 | macd_to_close | 107 | 285.2661 | 0.0526 |
| 8 | volatility_20 | 217 | 249.4400 | 0.0460 |
| 9 | return_60 | 197 | 236.8883 | 0.0437 |
| 10 | close_to_ema_12 | 87 | 204.7782 | 0.0378 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-03-29 |
| Train End | 2021-03-29 |
| Test Start | 2021-04-14 |
| Test End | 2021-07-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.2905 |
| Precision | 0.1875 |
| Recall | 0.0811 |
| F1 | 0.1132 |
| ROC-AUC | 0.2890 |
| Log Loss | 2.3111 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.9615 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5240 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 13], [34, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0141 |
| p25 | 0.1080 |
| median | 0.7884 |
| p75 | 0.9626 |
| max | 0.9997 |
| mean | 0.5962 |
| std | 0.3857 |
| count_ge_threshold | 16 |
| count_lt_threshold | 47 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volume_change_20 | 152 | 812.4285 | 0.1504 |
| 2 | volatility_60 | 286 | 597.4726 | 0.1106 |
| 3 | rsi_14 | 68 | 475.1344 | 0.0880 |
| 4 | ridge_pred_future_log_return | 232 | 467.5995 | 0.0866 |
| 5 | return_60 | 210 | 345.7297 | 0.0640 |
| 6 | macd_signal_to_close | 186 | 293.1397 | 0.0543 |
| 7 | close_to_ema_26 | 41 | 244.6429 | 0.0453 |
| 8 | volume_z_20 | 151 | 241.9965 | 0.0448 |
| 9 | return_10 | 109 | 196.9794 | 0.0365 |
| 10 | ma_gap_60 | 54 | 174.0764 | 0.0322 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-06-28 |
| Train End | 2021-06-28 |
| Test Start | 2021-07-14 |
| Test End | 2021-10-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5192 |
| Precision | 0.5968 |
| Recall | 1.0000 |
| F1 | 0.7475 |
| ROC-AUC | 0.5010 |
| Log Loss | 1.2300 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.0134 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4962 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 25], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0110 |
| p25 | 0.6764 |
| median | 0.9298 |
| p75 | 0.9705 |
| max | 0.9974 |
| mean | 0.7495 |
| std | 0.3286 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 350 | 747.9270 | 0.1422 |
| 2 | macd_signal_to_close | 275 | 685.5458 | 0.1303 |
| 3 | macd_hist_to_close | 148 | 388.3083 | 0.0738 |
| 4 | return_10 | 204 | 374.6062 | 0.0712 |
| 5 | volume_change_20 | 226 | 342.4074 | 0.0651 |
| 6 | return_60 | 97 | 246.6627 | 0.0469 |
| 7 | ma_gap_20 | 108 | 232.1258 | 0.0441 |
| 8 | ridge_pred_future_log_return | 136 | 228.7033 | 0.0435 |
| 9 | volatility_20 | 96 | 222.7791 | 0.0424 |
| 10 | return_20 | 193 | 220.2211 | 0.0419 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-09-27 |
| Train End | 2021-09-27 |
| Test Start | 2021-10-12 |
| Test End | 2022-01-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.1227 |
| Log Loss | 2.3502 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9900 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5135 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[29, 0], [34, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0092 |
| p25 | 0.0523 |
| median | 0.3258 |
| p75 | 0.9055 |
| max | 0.9883 |
| mean | 0.4580 |
| std | 0.3997 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 416 | 1025.0144 | 0.1976 |
| 2 | volatility_60 | 283 | 411.4912 | 0.0793 |
| 3 | volatility_20 | 148 | 377.4756 | 0.0728 |
| 4 | ma_gap_120 | 67 | 278.0895 | 0.0536 |
| 5 | close_to_ema_26 | 41 | 250.6014 | 0.0483 |
| 6 | return_10 | 143 | 248.2781 | 0.0479 |
| 7 | volume_z_20 | 162 | 227.3995 | 0.0438 |
| 8 | rsi_14 | 30 | 214.3093 | 0.0413 |
| 9 | ridge_pred_future_log_return | 137 | 195.1179 | 0.0376 |
| 10 | volume_change_20 | 217 | 188.1703 | 0.0363 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-12-27 |
| Train End | 2021-12-27 |
| Test Start | 2022-01-11 |
| Test End | 2022-04-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7302 |
| Balanced Accuracy | 0.7245 |
| Precision | 0.6667 |
| Recall | 0.6923 |
| F1 | 0.6792 |
| ROC-AUC | 0.7900 |
| Log Loss | 0.9632 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.9145 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5629 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 9], [8, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0293 |
| p25 | 0.7179 |
| median | 0.8603 |
| p75 | 0.9470 |
| max | 0.9904 |
| mean | 0.7539 |
| std | 0.2795 |
| count_ge_threshold | 27 |
| count_lt_threshold | 36 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 311 | 585.8614 | 0.1136 |
| 2 | return_60 | 171 | 486.3556 | 0.0943 |
| 3 | ridge_pred_future_log_return | 141 | 485.8228 | 0.0942 |
| 4 | ma_gap_120 | 209 | 416.0010 | 0.0806 |
| 5 | volatility_60 | 231 | 356.1042 | 0.0690 |
| 6 | return_20 | 158 | 311.9718 | 0.0605 |
| 7 | close_to_ema_26 | 61 | 215.4367 | 0.0418 |
| 8 | volume_z_20 | 163 | 209.5955 | 0.0406 |
| 9 | volume_change_5 | 169 | 201.3124 | 0.0390 |
| 10 | macd_signal_to_close | 141 | 181.3264 | 0.0352 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-30 |
| Train End | 2022-03-28 |
| Test Start | 2022-04-12 |
| Test End | 2022-07-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.5119 |
| Precision | 0.3387 |
| Recall | 1.0000 |
| F1 | 0.5060 |
| ROC-AUC | 0.5102 |
| Log Loss | 2.0125 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.3410 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5782 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 41], [0, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2577 |
| p25 | 0.9206 |
| median | 0.9599 |
| p75 | 0.9780 |
| max | 0.9902 |
| mean | 0.9034 |
| std | 0.1404 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 265 | 661.9459 | 0.1267 |
| 2 | return_60 | 222 | 404.6154 | 0.0775 |
| 3 | return_20 | 151 | 385.4427 | 0.0738 |
| 4 | macd_signal_to_close | 197 | 354.0446 | 0.0678 |
| 5 | volatility_20 | 206 | 333.9602 | 0.0639 |
| 6 | macd_to_close | 134 | 328.5987 | 0.0629 |
| 7 | ridge_pred_future_log_return | 166 | 307.7478 | 0.0589 |
| 8 | volatility_60 | 229 | 252.6813 | 0.0484 |
| 9 | close_to_ema_26 | 55 | 226.0734 | 0.0433 |
| 10 | rsi_14 | 70 | 198.3278 | 0.0380 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-29 |
| Train End | 2022-06-28 |
| Test Start | 2022-07-14 |
| Test End | 2022-10-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.6333 |
| Precision | 0.3636 |
| Recall | 0.8889 |
| F1 | 0.5161 |
| ROC-AUC | 0.5519 |
| Log Loss | 1.5053 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.5897 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6388 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 28], [2, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0072 |
| p25 | 0.3168 |
| median | 0.8785 |
| p75 | 0.9662 |
| max | 0.9930 |
| mean | 0.6747 |
| std | 0.3678 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 208 | 661.5444 | 0.1259 |
| 2 | return_60 | 235 | 554.5111 | 0.1055 |
| 3 | return_20 | 175 | 520.1642 | 0.0990 |
| 4 | ma_gap_120 | 145 | 465.5897 | 0.0886 |
| 5 | volatility_20 | 342 | 411.9865 | 0.0784 |
| 6 | ridge_pred_future_log_return | 148 | 294.5267 | 0.0561 |
| 7 | volatility_60 | 246 | 265.1582 | 0.0505 |
| 8 | rsi_14 | 80 | 264.4070 | 0.0503 |
| 9 | macd_hist_to_close | 150 | 205.0559 | 0.0390 |
| 10 | macd_to_close | 55 | 177.2927 | 0.0337 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-28 |
| Train End | 2022-09-27 |
| Test Start | 2022-10-12 |
| Test End | 2023-01-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.7583 |
| Precision | 0.9655 |
| Recall | 0.5833 |
| F1 | 0.7273 |
| ROC-AUC | 0.8125 |
| Log Loss | 1.5044 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.2543 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5917 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 1], [20, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0007 |
| p25 | 0.0098 |
| median | 0.2085 |
| p75 | 0.5914 |
| max | 0.9320 |
| mean | 0.3125 |
| std | 0.3211 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 402 | 927.8447 | 0.1753 |
| 2 | macd_signal_to_close | 226 | 877.1299 | 0.1657 |
| 3 | ma_gap_120 | 229 | 609.9963 | 0.1152 |
| 4 | return_20 | 225 | 447.9532 | 0.0846 |
| 5 | ma_gap_20 | 80 | 358.3876 | 0.0677 |
| 6 | volatility_20 | 148 | 274.9505 | 0.0519 |
| 7 | return_60 | 140 | 235.3726 | 0.0445 |
| 8 | macd_hist_to_close | 172 | 229.4731 | 0.0433 |
| 9 | return_5 | 190 | 203.8419 | 0.0385 |
| 10 | volume_z_20 | 148 | 137.9780 | 0.0261 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-28 |
| Train End | 2022-12-27 |
| Test Start | 2023-01-12 |
| Test End | 2023-04-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2381 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6500 |
| Log Loss | 2.7593 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.9865 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4913 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 0], [48, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0005 |
| p25 | 0.0050 |
| median | 0.0246 |
| p75 | 0.1080 |
| max | 0.5156 |
| mean | 0.0807 |
| std | 0.1259 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 482 | 1064.2254 | 0.2052 |
| 2 | volatility_20 | 245 | 869.3553 | 0.1676 |
| 3 | macd_signal_to_close | 189 | 355.0526 | 0.0685 |
| 4 | ma_gap_120 | 166 | 329.1631 | 0.0635 |
| 5 | ridge_pred_future_log_return | 186 | 296.2472 | 0.0571 |
| 6 | return_20 | 153 | 253.1749 | 0.0488 |
| 7 | macd_hist_to_close | 192 | 209.1800 | 0.0403 |
| 8 | ma_gap_20 | 102 | 207.8451 | 0.0401 |
| 9 | return_60 | 121 | 165.4200 | 0.0319 |
| 10 | return_5 | 114 | 162.1460 | 0.0313 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-30 |
| Train End | 2023-03-29 |
| Test Start | 2023-04-14 |
| Test End | 2023-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.6110 |
| Precision | 0.9130 |
| Recall | 0.4038 |
| F1 | 0.5600 |
| ROC-AUC | 0.6661 |
| Log Loss | 0.8453 |
| Baseline Accuracy | 0.8254 |
| Decision Threshold | 0.6331 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5684 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 2], [31, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0308 |
| p25 | 0.2383 |
| median | 0.4866 |
| p75 | 0.7499 |
| max | 0.9945 |
| mean | 0.5000 |
| std | 0.3057 |
| count_ge_threshold | 23 |
| count_lt_threshold | 40 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 415 | 1094.9476 | 0.2092 |
| 2 | volatility_60 | 296 | 754.8015 | 0.1442 |
| 3 | return_60 | 201 | 466.0989 | 0.0890 |
| 4 | return_20 | 244 | 426.7282 | 0.0815 |
| 5 | ma_gap_120 | 222 | 307.4589 | 0.0587 |
| 6 | ridge_pred_future_log_return | 152 | 279.4400 | 0.0534 |
| 7 | volume_change_20 | 194 | 246.0499 | 0.0470 |
| 8 | macd_signal_to_close | 128 | 232.4906 | 0.0444 |
| 9 | return_5 | 145 | 222.5299 | 0.0425 |
| 10 | return_10 | 164 | 182.9062 | 0.0349 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-29 |
| Train End | 2023-06-29 |
| Test Start | 2023-07-17 |
| Test End | 2023-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5873 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5894 |
| Log Loss | 1.3837 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.8095 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5143 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[37, 0], [26, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0012 |
| p25 | 0.0048 |
| median | 0.0172 |
| p75 | 0.1989 |
| max | 0.7494 |
| mean | 0.1334 |
| std | 0.2011 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 263 | 822.9042 | 0.1584 |
| 2 | ridge_pred_future_log_return | 255 | 703.2187 | 0.1354 |
| 3 | ma_gap_120 | 296 | 662.6478 | 0.1275 |
| 4 | volatility_20 | 237 | 460.6549 | 0.0887 |
| 5 | macd_signal_to_close | 153 | 309.5761 | 0.0596 |
| 6 | macd_hist_to_close | 178 | 237.3726 | 0.0457 |
| 7 | volume_change_20 | 186 | 207.1655 | 0.0399 |
| 8 | volatility_60 | 179 | 205.5604 | 0.0396 |
| 9 | ridge_pred_future_return | 47 | 177.2019 | 0.0341 |
| 10 | volume_z_20 | 150 | 173.2579 | 0.0333 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-28 |
| Train End | 2023-09-28 |
| Test Start | 2023-10-13 |
| Test End | 2024-01-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3175 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.3279 |
| Log Loss | 1.8772 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.9164 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5506 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [43, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0042 |
| p25 | 0.0238 |
| median | 0.1357 |
| p75 | 0.2913 |
| max | 0.6738 |
| mean | 0.2057 |
| std | 0.2090 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 237 | 793.5223 | 0.1514 |
| 2 | macd_hist_to_close | 221 | 606.2106 | 0.1156 |
| 3 | volatility_20 | 280 | 503.7692 | 0.0961 |
| 4 | ridge_pred_future_log_return | 226 | 489.2031 | 0.0933 |
| 5 | ma_gap_120 | 202 | 448.5612 | 0.0856 |
| 6 | volatility_60 | 238 | 347.4924 | 0.0663 |
| 7 | macd_signal_to_close | 197 | 332.6979 | 0.0635 |
| 8 | ma_gap_60 | 168 | 261.7317 | 0.0499 |
| 9 | volume_change_20 | 179 | 218.1209 | 0.0416 |
| 10 | volume_z_20 | 145 | 172.8476 | 0.0330 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-28 |
| Train End | 2023-12-28 |
| Test Start | 2024-01-16 |
| Test End | 2024-04-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.4656 |
| Precision | 0.6279 |
| Recall | 0.6585 |
| F1 | 0.6429 |
| ROC-AUC | 0.4290 |
| Log Loss | 1.1779 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.2492 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6047 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 16], [14, 27]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0118 |
| p25 | 0.1944 |
| median | 0.5908 |
| p75 | 0.8864 |
| max | 0.9923 |
| mean | 0.5453 |
| std | 0.3395 |
| count_ge_threshold | 43 |
| count_lt_threshold | 20 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 186 | 587.8321 | 0.1109 |
| 2 | ma_gap_120 | 179 | 458.9930 | 0.0866 |
| 3 | rsi_14 | 177 | 441.8324 | 0.0834 |
| 4 | return_60 | 216 | 376.3271 | 0.0710 |
| 5 | macd_hist_to_close | 241 | 366.9691 | 0.0692 |
| 6 | volatility_20 | 284 | 334.6738 | 0.0631 |
| 7 | return_20 | 110 | 323.7612 | 0.0611 |
| 8 | volatility_60 | 141 | 262.7605 | 0.0496 |
| 9 | macd_signal_to_close | 117 | 251.2801 | 0.0474 |
| 10 | return_10 | 148 | 210.7956 | 0.0398 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-03-29 |
| Train End | 2024-04-01 |
| Test Start | 2024-04-16 |
| Test End | 2024-07-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7778 |
| Balanced Accuracy | 0.6615 |
| Precision | 0.7778 |
| Recall | 0.9545 |
| F1 | 0.8571 |
| ROC-AUC | 0.7967 |
| Log Loss | 0.5241 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.4562 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5427 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 12], [2, 42]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0982 |
| p25 | 0.6929 |
| median | 0.9103 |
| p75 | 0.9712 |
| max | 0.9978 |
| mean | 0.7886 |
| std | 0.2473 |
| count_ge_threshold | 54 |
| count_lt_threshold | 9 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 299 | 1109.3593 | 0.2124 |
| 2 | ridge_pred_future_log_return | 199 | 591.7838 | 0.1133 |
| 3 | macd_signal_to_close | 208 | 448.6351 | 0.0859 |
| 4 | return_60 | 53 | 277.5011 | 0.0531 |
| 5 | macd_hist_to_close | 208 | 266.3466 | 0.0510 |
| 6 | rsi_14 | 110 | 257.4772 | 0.0493 |
| 7 | return_10 | 186 | 232.8133 | 0.0446 |
| 8 | ma_gap_60 | 144 | 202.8876 | 0.0388 |
| 9 | close_to_ema_26 | 85 | 184.6700 | 0.0354 |
| 10 | return_20 | 90 | 159.7118 | 0.0306 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-06-29 |
| Train End | 2024-07-01 |
| Test Start | 2024-07-17 |
| Test End | 2024-10-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.5233 |
| Precision | 1.0000 |
| Recall | 0.0465 |
| F1 | 0.0889 |
| ROC-AUC | 0.5570 |
| Log Loss | 0.9381 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.9880 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5207 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [41, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0255 |
| p25 | 0.5488 |
| median | 0.8122 |
| p75 | 0.9360 |
| max | 0.9978 |
| mean | 0.6916 |
| std | 0.3041 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 256 | 597.7714 | 0.1126 |
| 2 | macd_hist_to_close | 235 | 563.5950 | 0.1061 |
| 3 | volatility_20 | 266 | 465.6523 | 0.0877 |
| 4 | ma_gap_120 | 172 | 425.6096 | 0.0801 |
| 5 | ridge_pred_future_log_return | 223 | 420.5386 | 0.0792 |
| 6 | return_20 | 176 | 414.9779 | 0.0781 |
| 7 | macd_signal_to_close | 161 | 293.9185 | 0.0554 |
| 8 | ridge_pred_margin_to_threshold | 115 | 281.4051 | 0.0530 |
| 9 | return_10 | 192 | 234.2669 | 0.0441 |
| 10 | ma_gap_60 | 99 | 160.6253 | 0.0302 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-09-28 |
| Train End | 2024-09-30 |
| Test Start | 2024-10-15 |
| Test End | 2025-01-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5112 |
| Precision | 0.4286 |
| Recall | 0.1250 |
| F1 | 0.1935 |
| ROC-AUC | 0.6218 |
| Log Loss | 1.0307 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.9667 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6202 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[35, 4], [21, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0624 |
| p25 | 0.2574 |
| median | 0.5853 |
| p75 | 0.8691 |
| max | 0.9948 |
| mean | 0.5830 |
| std | 0.3205 |
| count_ge_threshold | 7 |
| count_lt_threshold | 56 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 262 | 583.5001 | 0.1129 |
| 2 | return_60 | 223 | 572.3014 | 0.1108 |
| 3 | return_20 | 184 | 565.0175 | 0.1093 |
| 4 | macd_hist_to_close | 293 | 443.8348 | 0.0859 |
| 5 | ridge_pred_future_log_return | 210 | 438.6138 | 0.0849 |
| 6 | ridge_pred_margin_to_threshold | 71 | 299.3867 | 0.0579 |
| 7 | volume_change_20 | 113 | 274.3278 | 0.0531 |
| 8 | volatility_60 | 121 | 208.1370 | 0.0403 |
| 9 | ema_12_to_ema_26 | 76 | 202.6319 | 0.0392 |
| 10 | close_to_ema_12 | 101 | 190.8236 | 0.0369 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-12-28 |
| Train End | 2024-12-30 |
| Test Start | 2025-01-16 |
| Test End | 2025-04-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.5854 |
| Precision | 0.3929 |
| Recall | 1.0000 |
| F1 | 0.5641 |
| ROC-AUC | 0.8614 |
| Log Loss | 1.7185 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.7950 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6381 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 34], [0, 22]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0631 |
| p25 | 0.9106 |
| median | 0.9615 |
| p75 | 0.9834 |
| max | 0.9963 |
| mean | 0.8976 |
| std | 0.1789 |
| count_ge_threshold | 56 |
| count_lt_threshold | 7 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 243 | 778.6542 | 0.1466 |
| 2 | return_60 | 317 | 763.0945 | 0.1437 |
| 3 | ridge_pred_future_log_return | 208 | 674.1173 | 0.1269 |
| 4 | macd_to_close | 209 | 464.7761 | 0.0875 |
| 5 | volatility_60 | 193 | 387.7762 | 0.0730 |
| 6 | ema_12_to_ema_26 | 133 | 298.7239 | 0.0563 |
| 7 | volatility_20 | 169 | 274.6124 | 0.0517 |
| 8 | macd_signal_to_close | 92 | 203.0647 | 0.0382 |
| 9 | return_20 | 137 | 189.1604 | 0.0356 |
| 10 | volume_z_20 | 113 | 180.5593 | 0.0340 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-03-30 |
| Train End | 2025-04-02 |
| Test Start | 2025-04-17 |
| Test End | 2025-07-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2381 |
| Balanced Accuracy | 0.6129 |
| Precision | 1.0000 |
| Recall | 0.2258 |
| F1 | 0.3684 |
| ROC-AUC | 0.5645 |
| Log Loss | 0.1496 |
| Baseline Accuracy | 0.9841 |
| Decision Threshold | 0.9871 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5953 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 0], [48, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2858 |
| p25 | 0.8961 |
| median | 0.9537 |
| p75 | 0.9797 |
| max | 0.9992 |
| mean | 0.9109 |
| std | 0.1236 |
| count_ge_threshold | 14 |
| count_lt_threshold | 49 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_margin_to_threshold | 293 | 883.0886 | 0.1636 |
| 2 | ma_gap_120 | 310 | 867.1720 | 0.1606 |
| 3 | volatility_60 | 257 | 562.7439 | 0.1042 |
| 4 | return_60 | 292 | 386.6133 | 0.0716 |
| 5 | ridge_pred_future_log_return | 86 | 329.9947 | 0.0611 |
| 6 | ema_12_to_ema_26 | 116 | 258.8358 | 0.0479 |
| 7 | macd_to_close | 83 | 229.0885 | 0.0424 |
| 8 | macd_hist_to_close | 183 | 217.3021 | 0.0402 |
| 9 | return_20 | 187 | 212.6439 | 0.0394 |
| 10 | high_low_range | 163 | 170.5437 | 0.0316 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-06-30 |
| Train End | 2025-07-03 |
| Test Start | 2025-07-21 |
| Test End | 2025-10-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.7596 |
| Log Loss | 1.4939 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.8039 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5950 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 0], [33, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0053 |
| p25 | 0.0161 |
| median | 0.0365 |
| p75 | 0.0762 |
| max | 0.4432 |
| mean | 0.0699 |
| std | 0.0889 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 323 | 1156.7451 | 0.2126 |
| 2 | ma_gap_60 | 104 | 560.8175 | 0.1031 |
| 3 | ema_12_to_ema_26 | 157 | 510.8415 | 0.0939 |
| 4 | volatility_60 | 398 | 438.3292 | 0.0805 |
| 5 | volatility_20 | 160 | 366.7079 | 0.0674 |
| 6 | macd_to_close | 179 | 353.1512 | 0.0649 |
| 7 | macd_signal_to_close | 201 | 312.8181 | 0.0575 |
| 8 | ridge_pred_future_return | 58 | 285.4118 | 0.0524 |
| 9 | ma_gap_120 | 132 | 180.9303 | 0.0332 |
| 10 | high_low_range | 159 | 142.2159 | 0.0261 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-09-29 |
| Train End | 2025-10-02 |
| Test Start | 2025-10-17 |
| Test End | 2026-01-16 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.4769 |
| Precision | 0.4130 |
| Recall | 0.7037 |
| F1 | 0.5205 |
| ROC-AUC | 0.4167 |
| Log Loss | 1.4396 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.0848 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7818 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 27], [8, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0032 |
| p25 | 0.0715 |
| median | 0.3363 |
| p75 | 0.8399 |
| max | 0.9748 |
| mean | 0.4300 |
| std | 0.3649 |
| count_ge_threshold | 46 |
| count_lt_threshold | 17 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 317 | 921.2528 | 0.1746 |
| 2 | return_60 | 156 | 411.7139 | 0.0780 |
| 3 | volatility_60 | 322 | 361.5360 | 0.0685 |
| 4 | macd_hist_to_close | 160 | 358.2297 | 0.0679 |
| 5 | ma_gap_120 | 285 | 350.9989 | 0.0665 |
| 6 | ema_12_to_ema_26 | 130 | 313.6248 | 0.0594 |
| 7 | volatility_20 | 163 | 291.0404 | 0.0552 |
| 8 | return_10 | 118 | 269.2756 | 0.0510 |
| 9 | high_low_range | 122 | 231.7804 | 0.0439 |
| 10 | return_20 | 241 | 219.8456 | 0.0417 |

### Fold 29

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-12-29 |
| Train End | 2026-01-02 |
| Test Start | 2026-01-20 |
| Test End | 2026-04-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5873 |
| Balanced Accuracy | 0.5517 |
| Precision | 1.0000 |
| Recall | 0.1034 |
| F1 | 0.1875 |
| ROC-AUC | 0.5781 |
| Log Loss | 0.9931 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.6960 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6623 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[34, 0], [26, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0202 |
| p25 | 0.0512 |
| median | 0.0970 |
| p75 | 0.2739 |
| max | 0.8376 |
| mean | 0.1939 |
| std | 0.2020 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 235 | 797.6436 | 0.1494 |
| 2 | ma_gap_120 | 396 | 694.9895 | 0.1302 |
| 3 | macd_signal_to_close | 223 | 552.2445 | 0.1034 |
| 4 | volatility_20 | 343 | 511.8668 | 0.0959 |
| 5 | ridge_pred_future_log_return | 189 | 402.2214 | 0.0753 |
| 6 | ma_gap_20 | 101 | 352.5188 | 0.0660 |
| 7 | macd_hist_to_close | 148 | 294.8212 | 0.0552 |
| 8 | return_20 | 149 | 262.7183 | 0.0492 |
| 9 | return_10 | 108 | 179.7331 | 0.0337 |
| 10 | ma_gap_60 | 133 | 152.5558 | 0.0286 |
