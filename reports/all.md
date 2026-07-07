# Direction Validation Report: ^N225

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^N225 |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-07 08:09:16 JST |
| Last Date | 2026-07-03 |
| Last Close | 69744.0703 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.4343 |
| Probability Down | 0.5657 |
| Decision Threshold | 0.3919 |
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
| beats_baseline_fold_ratio | 0.2500 | N/A | N/A | N/A |
| confusion_matrix_sum | [[538, 297], [575, 354]] | N/A | N/A | N/A |
| accuracy | 0.5057 | 0.1128 | 0.3016 | 0.6984 |
| balanced_accuracy | 0.5388 | 0.0641 | 0.4158 | 0.6658 |
| precision | 0.6267 | 0.2951 | 0.0000 | 1.0000 |
| recall | 0.4068 | 0.3251 | 0.0000 | 1.0000 |
| f1 | 0.3812 | 0.2401 | 0.0000 | 0.8224 |
| roc_auc | 0.6196 | 0.1102 | 0.3721 | 0.7929 |
| log_loss | 1.1039 | 0.3829 | 0.6209 | 2.2152 |
| baseline_accuracy | 0.6094 | 0.0725 | 0.5079 | 0.7302 |
| decision_threshold | 0.6163 | 0.3176 | 0.0100 | 0.9815 |
| calibration_score | 0.5933 | 0.0544 | 0.5033 | 0.7012 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 340 | 906.8253 | 0.1719 |
| 2 | volatility_10 | 306 | 615.6245 | 0.1167 |
| 3 | volatility_60 | 197 | 507.0157 | 0.0961 |
| 4 | ma_gap_60 | 95 | 467.4636 | 0.0886 |
| 5 | volatility_20 | 218 | 428.2226 | 0.0812 |
| 6 | macd_hist_to_close | 138 | 243.6954 | 0.0462 |
| 7 | close_to_ema_12 | 139 | 243.6759 | 0.0462 |
| 8 | ma_gap_5 | 93 | 221.4601 | 0.0420 |
| 9 | ma_gap_120 | 67 | 214.0295 | 0.0406 |
| 10 | ema_12_to_ema_26 | 85 | 195.2916 | 0.0370 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-02-27 | 2019-06-06 | 0.4921 | 0.4676 | 0.4455 | 0.5714 | 0.8871 | ok |
| 2 | ok | 2019-06-07 | 2019-09-05 | 0.4127 | 0.4158 | 0.5272 | 0.5079 | 0.7291 | ok |
| 3 | ok | 2019-09-06 | 2019-12-12 | 0.3810 | 0.5125 | 0.5315 | 0.6349 | 0.9000 | ok |
| 4 | ok | 2019-12-13 | 2020-03-19 | 0.3492 | 0.4286 | 0.6383 | 0.6667 | 0.4006 | ok |
| 5 | ok | 2020-03-23 | 2020-06-23 | 0.6984 | 0.5000 | 0.7428 | 0.6984 | 0.0100 | ok |
| 6 | ok | 2020-06-24 | 2020-09-29 | 0.5079 | 0.5179 | 0.5327 | 0.5556 | 0.7304 | ok |
| 7 | ok | 2020-09-30 | 2020-12-30 | 0.6032 | 0.5889 | 0.6235 | 0.7143 | 0.3381 | ok |
| 8 | ok | 2021-01-04 | 2021-04-05 | 0.4603 | 0.5143 | 0.6520 | 0.5556 | 0.6602 | ok |
| 9 | ok | 2021-04-06 | 2021-07-07 | 0.6190 | 0.5723 | 0.5217 | 0.7302 | 0.7968 | ok |
| 10 | ok | 2021-07-08 | 2021-10-11 | 0.5079 | 0.5036 | 0.5184 | 0.5556 | 0.5028 | ok |
| 11 | ok | 2021-10-12 | 2022-01-13 | 0.5714 | 0.5951 | 0.6653 | 0.5873 | 0.8815 | ok |
| 12 | ok | 2022-01-14 | 2022-04-15 | 0.3651 | 0.5455 | 0.7500 | 0.6984 | 0.1758 | ok |
| 13 | ok | 2022-04-18 | 2022-07-20 | 0.6190 | 0.6571 | 0.7929 | 0.5556 | 0.9635 | ok |
| 14 | ok | 2022-07-21 | 2022-10-21 | 0.5873 | 0.5536 | 0.4878 | 0.5556 | 0.9140 | ok |
| 15 | ok | 2022-10-24 | 2023-01-25 | 0.6508 | 0.6507 | 0.6754 | 0.5079 | 0.2281 | ok |
| 16 | ok | 2023-01-26 | 2023-04-26 | 0.3968 | 0.5366 | 0.6608 | 0.6508 | 0.8976 | ok |
| 17 | ok | 2023-04-27 | 2023-07-28 | 0.5397 | 0.5774 | 0.6874 | 0.6032 | 0.2869 | ok |
| 18 | ok | 2023-07-31 | 2023-10-30 | 0.5714 | 0.5000 | 0.5288 | 0.5714 | 0.9369 | ok |
| 19 | ok | 2023-10-31 | 2024-02-02 | 0.3175 | 0.5000 | 0.7860 | 0.6825 | 0.9815 | ok |
| 20 | ok | 2024-02-05 | 2024-05-09 | 0.5397 | 0.5227 | 0.6909 | 0.5238 | 0.0701 | ok |
| 21 | ok | 2024-05-10 | 2024-08-07 | 0.5238 | 0.5946 | 0.4844 | 0.5873 | 0.1674 | ok |
| 22 | ok | 2024-08-08 | 2024-11-11 | 0.6667 | 0.6658 | 0.7550 | 0.5079 | 0.5957 | ok |
| 23 | ok | 2024-11-12 | 2025-02-14 | 0.6349 | 0.5208 | 0.6506 | 0.6190 | 0.8221 | ok |
| 24 | ok | 2025-02-17 | 2025-05-21 | 0.6190 | 0.6174 | 0.7127 | 0.5079 | 0.8877 | ok |
| 25 | ok | 2025-05-22 | 2025-08-20 | 0.3810 | 0.5568 | 0.7356 | 0.6984 | 0.8881 | ok |
| 26 | ok | 2025-08-21 | 2025-11-21 | 0.3016 | 0.5217 | 0.5205 | 0.7302 | 0.9681 | ok |
| 27 | ok | 2025-11-25 | 2026-02-27 | 0.4286 | 0.5263 | 0.6579 | 0.6032 | 0.5000 | ok |
| 28 | ok | 2026-03-02 | 2026-06-03 | 0.4127 | 0.4227 | 0.3721 | 0.6825 | 0.1355 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2016-12-30 |
| Train End | 2019-02-08 |
| Test Start | 2019-02-27 |
| Test End | 2019-06-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.4676 |
| Precision | 0.3810 |
| Recall | 0.2963 |
| F1 | 0.3333 |
| ROC-AUC | 0.4455 |
| Log Loss | 1.4376 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.8871 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6173 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 13], [19, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0044 |
| p25 | 0.1215 |
| median | 0.7329 |
| p75 | 0.9194 |
| max | 0.9842 |
| mean | 0.5646 |
| std | 0.3714 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 269 | 1071.7424 | 0.1997 |
| 2 | ma_gap_120 | 410 | 930.1376 | 0.1733 |
| 3 | volatility_20 | 222 | 438.7993 | 0.0818 |
| 4 | macd_hist_to_close | 238 | 351.2234 | 0.0654 |
| 5 | return_10 | 161 | 266.8008 | 0.0497 |
| 6 | macd_signal_to_close | 157 | 232.7986 | 0.0434 |
| 7 | ema_12_to_ema_26 | 81 | 207.9779 | 0.0388 |
| 8 | return_60 | 65 | 166.3566 | 0.0310 |
| 9 | ma_gap_5 | 97 | 151.5208 | 0.0282 |
| 10 | volume_z_20 | 106 | 147.2891 | 0.0274 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-04-04 |
| Train End | 2019-05-22 |
| Test Start | 2019-06-07 |
| Test End | 2019-09-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4158 |
| Precision | 0.3684 |
| Recall | 0.2188 |
| F1 | 0.2745 |
| ROC-AUC | 0.5272 |
| Log Loss | 0.8505 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.7291 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6190 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 12], [25, 7]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0340 |
| p25 | 0.3164 |
| median | 0.4658 |
| p75 | 0.7954 |
| max | 0.9333 |
| mean | 0.5253 |
| std | 0.2533 |
| count_ge_threshold | 19 |
| count_lt_threshold | 44 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 260 | 1036.1716 | 0.1980 |
| 2 | ma_gap_120 | 238 | 681.0146 | 0.1301 |
| 3 | macd_hist_to_close | 143 | 352.1781 | 0.0673 |
| 4 | volatility_10 | 185 | 346.1171 | 0.0661 |
| 5 | volatility_20 | 154 | 310.4632 | 0.0593 |
| 6 | ridge_pred_future_log_return | 156 | 286.5771 | 0.0548 |
| 7 | macd_signal_to_close | 136 | 219.4910 | 0.0419 |
| 8 | return_60 | 123 | 194.4839 | 0.0372 |
| 9 | volume_change_5 | 112 | 191.6057 | 0.0366 |
| 10 | volatility_5 | 104 | 184.8291 | 0.0353 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-07-11 |
| Train End | 2019-08-22 |
| Test Start | 2019-09-06 |
| Test End | 2019-12-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3810 |
| Balanced Accuracy | 0.5125 |
| Precision | 1.0000 |
| Recall | 0.0250 |
| F1 | 0.0488 |
| ROC-AUC | 0.5315 |
| Log Loss | 1.6120 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.9000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5446 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 0], [39, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0027 |
| p25 | 0.0338 |
| median | 0.0955 |
| p75 | 0.2322 |
| max | 0.9106 |
| mean | 0.1843 |
| std | 0.2253 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 293 | 1393.0045 | 0.2612 |
| 2 | volatility_20 | 329 | 799.9348 | 0.1500 |
| 3 | ma_gap_120 | 198 | 541.3951 | 0.1015 |
| 4 | return_60 | 209 | 344.8299 | 0.0647 |
| 5 | return_20 | 238 | 324.9925 | 0.0609 |
| 6 | return_5 | 100 | 273.9551 | 0.0514 |
| 7 | macd_signal_to_close | 147 | 203.8282 | 0.0382 |
| 8 | volatility_5 | 167 | 170.7048 | 0.0320 |
| 9 | volume_change_5 | 130 | 157.6084 | 0.0296 |
| 10 | volatility_10 | 87 | 105.2108 | 0.0197 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-10-12 |
| Train End | 2019-11-28 |
| Test Start | 2019-12-13 |
| Test End | 2020-03-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.4286 |
| Precision | 0.2917 |
| Recall | 0.6667 |
| F1 | 0.4058 |
| ROC-AUC | 0.6383 |
| Log Loss | 1.1302 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.4006 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5751 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 34], [7, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0021 |
| p25 | 0.4051 |
| median | 0.7293 |
| p75 | 0.8526 |
| max | 0.9848 |
| mean | 0.5986 |
| std | 0.3373 |
| count_ge_threshold | 48 |
| count_lt_threshold | 15 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 277 | 806.5294 | 0.1575 |
| 2 | volatility_60 | 235 | 619.5486 | 0.1210 |
| 3 | volatility_20 | 291 | 509.4001 | 0.0995 |
| 4 | ridge_pred_future_log_return | 152 | 379.9800 | 0.0742 |
| 5 | macd_signal_to_close | 180 | 333.7993 | 0.0652 |
| 6 | volume_change_5 | 157 | 187.6457 | 0.0366 |
| 7 | volatility_10 | 108 | 186.1800 | 0.0364 |
| 8 | return_20 | 120 | 157.2840 | 0.0307 |
| 9 | ma_gap_60 | 92 | 140.2456 | 0.0274 |
| 10 | volatility_5 | 110 | 138.5148 | 0.0270 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-01-18 |
| Train End | 2020-03-05 |
| Test Start | 2020-03-23 |
| Test End | 2020-06-23 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6984 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6984 |
| Recall | 1.0000 |
| F1 | 0.8224 |
| ROC-AUC | 0.7428 |
| Log Loss | 1.1038 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.0100 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5409 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 19], [0, 44]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0100 |
| p25 | 0.0297 |
| median | 0.6596 |
| p75 | 0.9522 |
| max | 0.9951 |
| mean | 0.4924 |
| std | 0.4356 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 302 | 938.6913 | 0.1836 |
| 2 | volatility_60 | 255 | 828.6012 | 0.1620 |
| 3 | ma_gap_60 | 168 | 328.6912 | 0.0643 |
| 4 | volatility_20 | 172 | 264.7525 | 0.0518 |
| 5 | ema_12_to_ema_26 | 167 | 246.4956 | 0.0482 |
| 6 | volume_change_20 | 158 | 224.0493 | 0.0438 |
| 7 | return_60 | 149 | 211.5609 | 0.0414 |
| 8 | return_5 | 102 | 207.9691 | 0.0407 |
| 9 | ma_gap_20 | 129 | 207.1367 | 0.0405 |
| 10 | ma_gap_120 | 144 | 196.0576 | 0.0383 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-04-20 |
| Train End | 2020-06-09 |
| Test Start | 2020-06-24 |
| Test End | 2020-09-29 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.5179 |
| Precision | 0.5769 |
| Recall | 0.4286 |
| F1 | 0.4918 |
| ROC-AUC | 0.5327 |
| Log Loss | 0.8136 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.7304 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5810 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 11], [20, 15]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0810 |
| p25 | 0.4683 |
| median | 0.6649 |
| p75 | 0.8052 |
| max | 0.9769 |
| mean | 0.6191 |
| std | 0.2366 |
| count_ge_threshold | 26 |
| count_lt_threshold | 37 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 387 | 906.4742 | 0.1792 |
| 2 | ema_12_to_ema_26 | 174 | 357.0250 | 0.0706 |
| 3 | volatility_20 | 131 | 264.4167 | 0.0523 |
| 4 | macd_signal_to_close | 166 | 251.5346 | 0.0497 |
| 5 | ma_gap_120 | 162 | 248.9157 | 0.0492 |
| 6 | return_5 | 159 | 244.9354 | 0.0484 |
| 7 | return_20 | 107 | 228.8523 | 0.0452 |
| 8 | macd_hist_to_close | 136 | 209.4700 | 0.0414 |
| 9 | volatility_10 | 117 | 206.1902 | 0.0408 |
| 10 | ma_gap_60 | 117 | 183.6134 | 0.0363 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-07-24 |
| Train End | 2020-09-11 |
| Test Start | 2020-09-30 |
| Test End | 2020-12-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5889 |
| Precision | 0.7778 |
| Recall | 0.6222 |
| F1 | 0.6914 |
| ROC-AUC | 0.6235 |
| Log Loss | 0.7828 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.3381 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6178 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[10, 8], [17, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0465 |
| p25 | 0.2349 |
| median | 0.3789 |
| p75 | 0.6372 |
| max | 0.9291 |
| mean | 0.4514 |
| std | 0.2462 |
| count_ge_threshold | 36 |
| count_lt_threshold | 27 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 283 | 489.6385 | 0.0969 |
| 2 | ma_gap_120 | 191 | 477.1559 | 0.0944 |
| 3 | ridge_pred_future_log_return | 192 | 366.3371 | 0.0725 |
| 4 | volatility_10 | 162 | 317.3149 | 0.0628 |
| 5 | return_20 | 145 | 293.9562 | 0.0582 |
| 6 | return_60 | 111 | 290.8538 | 0.0576 |
| 7 | macd_hist_to_close | 163 | 261.4689 | 0.0517 |
| 8 | volatility_20 | 158 | 210.3492 | 0.0416 |
| 9 | volatility_5 | 164 | 189.2663 | 0.0375 |
| 10 | macd_signal_to_close | 120 | 183.3231 | 0.0363 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-10-25 |
| Train End | 2020-12-16 |
| Test Start | 2021-01-04 |
| Test End | 2021-04-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.5143 |
| Precision | 1.0000 |
| Recall | 0.0286 |
| F1 | 0.0556 |
| ROC-AUC | 0.6520 |
| Log Loss | 1.0686 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.6602 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5983 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 0], [34, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0118 |
| p25 | 0.0658 |
| median | 0.1417 |
| p75 | 0.2800 |
| max | 0.7476 |
| mean | 0.1978 |
| std | 0.1703 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 187 | 505.3163 | 0.0970 |
| 2 | macd_signal_to_close | 140 | 368.0758 | 0.0706 |
| 3 | return_20 | 94 | 366.3234 | 0.0703 |
| 4 | volume_change_20 | 207 | 349.8233 | 0.0671 |
| 5 | volatility_60 | 223 | 345.5748 | 0.0663 |
| 6 | volatility_10 | 144 | 336.3018 | 0.0645 |
| 7 | ma_gap_60 | 81 | 322.4838 | 0.0619 |
| 8 | ridge_pred_future_log_return | 187 | 297.2844 | 0.0571 |
| 9 | macd_hist_to_close | 89 | 261.7008 | 0.0502 |
| 10 | high_low_range | 203 | 182.4182 | 0.0350 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-02-12 |
| Train End | 2021-03-22 |
| Test Start | 2021-04-06 |
| Test End | 2021-07-07 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5723 |
| Precision | 0.3478 |
| Recall | 0.4706 |
| F1 | 0.4000 |
| ROC-AUC | 0.5217 |
| Log Loss | 1.2901 |
| Baseline Accuracy | 0.7302 |
| Decision Threshold | 0.7968 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6118 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[31, 15], [9, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0654 |
| p25 | 0.4025 |
| median | 0.6439 |
| p75 | 0.9257 |
| max | 0.9930 |
| mean | 0.6282 |
| std | 0.2888 |
| count_ge_threshold | 23 |
| count_lt_threshold | 40 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 163 | 663.8091 | 0.1250 |
| 2 | ridge_pred_future_log_return | 291 | 647.6579 | 0.1219 |
| 3 | close_to_ema_12 | 206 | 511.4662 | 0.0963 |
| 4 | volume_change_20 | 204 | 327.1375 | 0.0616 |
| 5 | macd_signal_to_close | 133 | 288.4921 | 0.0543 |
| 6 | volatility_10 | 203 | 278.7727 | 0.0525 |
| 7 | volatility_60 | 153 | 274.6843 | 0.0517 |
| 8 | return_10 | 151 | 224.6525 | 0.0423 |
| 9 | ridge_pred_future_return | 59 | 221.8928 | 0.0418 |
| 10 | volume_change_5 | 179 | 200.9509 | 0.0378 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-05-23 |
| Train End | 2021-06-23 |
| Test Start | 2021-07-08 |
| Test End | 2021-10-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.5036 |
| Precision | 0.4483 |
| Recall | 0.4643 |
| F1 | 0.4561 |
| ROC-AUC | 0.5184 |
| Log Loss | 0.9485 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.5028 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6499 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 16], [15, 13]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0212 |
| p25 | 0.2092 |
| median | 0.4450 |
| p75 | 0.6681 |
| max | 0.9772 |
| mean | 0.4531 |
| std | 0.2891 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 171 | 463.8377 | 0.0887 |
| 2 | volatility_60 | 215 | 434.5054 | 0.0831 |
| 3 | macd_hist_to_close | 163 | 391.5123 | 0.0749 |
| 4 | volatility_10 | 139 | 341.5932 | 0.0653 |
| 5 | ma_gap_120 | 96 | 322.4725 | 0.0617 |
| 6 | close_to_ema_12 | 136 | 308.4320 | 0.0590 |
| 7 | volatility_20 | 129 | 280.3447 | 0.0536 |
| 8 | return_60 | 183 | 228.1050 | 0.0436 |
| 9 | ridge_pred_future_log_return | 153 | 204.6178 | 0.0391 |
| 10 | volume_z_20 | 140 | 191.9297 | 0.0367 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-08-23 |
| Train End | 2021-09-27 |
| Test Start | 2021-10-12 |
| Test End | 2022-01-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5714 |
| Balanced Accuracy | 0.5951 |
| Precision | 0.4872 |
| Recall | 0.7308 |
| F1 | 0.5846 |
| ROC-AUC | 0.6653 |
| Log Loss | 1.4903 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.8815 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6108 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 20], [7, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1169 |
| p25 | 0.7471 |
| median | 0.9474 |
| p75 | 0.9859 |
| max | 0.9976 |
| mean | 0.8339 |
| std | 0.2181 |
| count_ge_threshold | 39 |
| count_lt_threshold | 24 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 207 | 779.8760 | 0.1506 |
| 2 | volatility_5 | 190 | 373.7688 | 0.0722 |
| 3 | ma_gap_60 | 97 | 320.0052 | 0.0618 |
| 4 | rsi_14 | 30 | 313.9064 | 0.0606 |
| 5 | macd_signal_to_close | 128 | 276.7600 | 0.0534 |
| 6 | return_60 | 85 | 236.6101 | 0.0457 |
| 7 | ma_gap_120 | 110 | 224.5019 | 0.0433 |
| 8 | ma_gap_20 | 103 | 213.0563 | 0.0411 |
| 9 | close_to_ema_12 | 76 | 179.0714 | 0.0346 |
| 10 | volatility_10 | 152 | 173.8046 | 0.0336 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-11-29 |
| Train End | 2021-12-27 |
| Test Start | 2022-01-14 |
| Test End | 2022-04-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.5455 |
| Precision | 0.3220 |
| Recall | 1.0000 |
| F1 | 0.4872 |
| ROC-AUC | 0.7500 |
| Log Loss | 1.4459 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.1758 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6157 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 40], [0, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0295 |
| p25 | 0.5156 |
| median | 0.9006 |
| p75 | 0.9810 |
| max | 0.9953 |
| mean | 0.7490 |
| std | 0.3027 |
| count_ge_threshold | 59 |
| count_lt_threshold | 4 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 391 | 1027.9460 | 0.1945 |
| 2 | ma_gap_20 | 89 | 449.3242 | 0.0850 |
| 3 | macd_signal_to_close | 243 | 364.2245 | 0.0689 |
| 4 | volatility_5 | 190 | 333.2444 | 0.0631 |
| 5 | volume_change_5 | 197 | 274.9825 | 0.0520 |
| 6 | macd_hist_to_close | 103 | 268.3349 | 0.0508 |
| 7 | ma_gap_60 | 121 | 255.9686 | 0.0484 |
| 8 | volume_change_20 | 161 | 222.8923 | 0.0422 |
| 9 | ridge_pred_future_log_return | 130 | 220.4108 | 0.0417 |
| 10 | volatility_20 | 174 | 204.4624 | 0.0387 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-06 |
| Train End | 2022-04-01 |
| Test Start | 2022-04-18 |
| Test End | 2022-07-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6571 |
| Precision | 1.0000 |
| Recall | 0.3143 |
| F1 | 0.4783 |
| ROC-AUC | 0.7929 |
| Log Loss | 0.6311 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9635 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6206 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 0], [24, 11]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0159 |
| p25 | 0.2858 |
| median | 0.7706 |
| p75 | 0.9228 |
| max | 0.9930 |
| mean | 0.6300 |
| std | 0.3329 |
| count_ge_threshold | 11 |
| count_lt_threshold | 52 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 351 | 1009.6765 | 0.1897 |
| 2 | volatility_20 | 242 | 445.9724 | 0.0838 |
| 3 | macd_signal_to_close | 191 | 389.5309 | 0.0732 |
| 4 | return_10 | 196 | 362.3320 | 0.0681 |
| 5 | high_low_range | 101 | 327.7288 | 0.0616 |
| 6 | volume_change_5 | 221 | 325.4052 | 0.0611 |
| 7 | volatility_5 | 184 | 298.8209 | 0.0561 |
| 8 | ridge_pred_future_log_return | 175 | 273.4327 | 0.0514 |
| 9 | ma_gap_60 | 124 | 208.0635 | 0.0391 |
| 10 | ridge_pred_margin_to_threshold | 98 | 191.6616 | 0.0360 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-10 |
| Train End | 2022-07-05 |
| Test Start | 2022-07-21 |
| Test End | 2022-10-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5873 |
| Balanced Accuracy | 0.5536 |
| Precision | 0.5833 |
| Recall | 0.2500 |
| F1 | 0.3500 |
| ROC-AUC | 0.4878 |
| Log Loss | 1.2760 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9140 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7012 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 5], [21, 7]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0025 |
| p25 | 0.1407 |
| median | 0.4720 |
| p75 | 0.8646 |
| max | 0.9927 |
| mean | 0.4987 |
| std | 0.3607 |
| count_ge_threshold | 12 |
| count_lt_threshold | 51 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 351 | 787.6624 | 0.1509 |
| 2 | macd_hist_to_close | 153 | 446.7174 | 0.0856 |
| 3 | macd_to_close | 90 | 389.3094 | 0.0746 |
| 4 | return_20 | 123 | 334.4936 | 0.0641 |
| 5 | volatility_20 | 172 | 316.4757 | 0.0606 |
| 6 | macd_signal_to_close | 173 | 307.3490 | 0.0589 |
| 7 | volatility_5 | 191 | 293.9579 | 0.0563 |
| 8 | return_5 | 215 | 285.0752 | 0.0546 |
| 9 | volume_change_5 | 181 | 234.3251 | 0.0449 |
| 10 | close_to_high | 115 | 203.5991 | 0.0390 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-14 |
| Train End | 2022-10-06 |
| Test Start | 2022-10-24 |
| Test End | 2023-01-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.6507 |
| Precision | 0.6452 |
| Recall | 0.6452 |
| F1 | 0.6452 |
| ROC-AUC | 0.6754 |
| Log Loss | 1.0084 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.2281 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5645 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 11], [11, 20]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0023 |
| p25 | 0.0534 |
| median | 0.1784 |
| p75 | 0.8981 |
| max | 0.9931 |
| mean | 0.4181 |
| std | 0.3964 |
| count_ge_threshold | 31 |
| count_lt_threshold | 32 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 198 | 892.0244 | 0.1724 |
| 2 | volatility_60 | 233 | 545.8975 | 0.1055 |
| 3 | macd_signal_to_close | 190 | 398.5031 | 0.0770 |
| 4 | return_5 | 156 | 314.5760 | 0.0608 |
| 5 | close_to_high | 196 | 285.0285 | 0.0551 |
| 6 | return_10 | 99 | 269.9834 | 0.0522 |
| 7 | volatility_10 | 132 | 236.0093 | 0.0456 |
| 8 | volatility_5 | 204 | 231.4825 | 0.0447 |
| 9 | volatility_20 | 151 | 219.9361 | 0.0425 |
| 10 | return_60 | 154 | 217.1119 | 0.0420 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-17 |
| Train End | 2023-01-11 |
| Test Start | 2023-01-26 |
| Test End | 2023-04-26 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.5366 |
| Precision | 1.0000 |
| Recall | 0.0732 |
| F1 | 0.1364 |
| ROC-AUC | 0.6608 |
| Log Loss | 1.0740 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.8976 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5335 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 0], [38, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0097 |
| p25 | 0.0586 |
| median | 0.2216 |
| p75 | 0.4145 |
| max | 0.9264 |
| mean | 0.2993 |
| std | 0.2664 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 428 | 962.6529 | 0.1846 |
| 2 | macd_signal_to_close | 250 | 723.0125 | 0.1387 |
| 3 | return_20 | 257 | 684.7336 | 0.1313 |
| 4 | ema_12_to_ema_26 | 116 | 296.5741 | 0.0569 |
| 5 | close_to_high | 157 | 279.2838 | 0.0536 |
| 6 | ma_gap_20 | 34 | 246.9283 | 0.0474 |
| 7 | volatility_5 | 177 | 179.1884 | 0.0344 |
| 8 | return_5 | 175 | 171.9214 | 0.0330 |
| 9 | volatility_20 | 130 | 148.0113 | 0.0284 |
| 10 | close_to_low | 160 | 143.4739 | 0.0275 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-23 |
| Train End | 2023-04-12 |
| Test Start | 2023-04-27 |
| Test End | 2023-07-28 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5774 |
| Precision | 0.7143 |
| Recall | 0.3947 |
| F1 | 0.5085 |
| ROC-AUC | 0.6874 |
| Log Loss | 0.9705 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.2869 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5786 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 6], [23, 15]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0198 |
| p25 | 0.1052 |
| median | 0.1978 |
| p75 | 0.3297 |
| max | 0.9210 |
| mean | 0.2560 |
| std | 0.2041 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 185 | 636.6199 | 0.1286 |
| 2 | macd_signal_to_close | 181 | 422.9226 | 0.0854 |
| 3 | volatility_60 | 180 | 392.5297 | 0.0793 |
| 4 | ema_12_to_ema_26 | 180 | 301.2215 | 0.0608 |
| 5 | close_to_high | 141 | 281.2190 | 0.0568 |
| 6 | return_60 | 157 | 267.4088 | 0.0540 |
| 7 | volatility_10 | 162 | 266.7199 | 0.0539 |
| 8 | ma_gap_120 | 86 | 246.5058 | 0.0498 |
| 9 | volatility_20 | 138 | 179.8686 | 0.0363 |
| 10 | ma_gap_60 | 85 | 173.1484 | 0.0350 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-24 |
| Train End | 2023-07-13 |
| Test Start | 2023-07-31 |
| Test End | 2023-10-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5714 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5288 |
| Log Loss | 0.9227 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.9369 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5057 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[36, 0], [27, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0035 |
| p25 | 0.0983 |
| median | 0.3190 |
| p75 | 0.5655 |
| max | 0.8648 |
| mean | 0.3489 |
| std | 0.2692 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 212 | 684.7670 | 0.1373 |
| 2 | macd_signal_to_close | 131 | 484.8387 | 0.0972 |
| 3 | ma_gap_60 | 144 | 349.4517 | 0.0701 |
| 4 | ema_12_to_ema_26 | 144 | 287.6605 | 0.0577 |
| 5 | macd_to_close | 92 | 283.4888 | 0.0569 |
| 6 | volatility_10 | 188 | 217.9976 | 0.0437 |
| 7 | ma_gap_20 | 89 | 208.8116 | 0.0419 |
| 8 | volume_z_20 | 122 | 200.8834 | 0.0403 |
| 9 | volume_change_5 | 153 | 192.3994 | 0.0386 |
| 10 | ma_gap_120 | 120 | 163.4019 | 0.0328 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-28 |
| Train End | 2023-10-16 |
| Test Start | 2023-10-31 |
| Test End | 2024-02-02 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3175 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.7860 |
| Log Loss | 0.9360 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.9815 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5075 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [43, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0073 |
| p25 | 0.0593 |
| median | 0.2443 |
| p75 | 0.5727 |
| max | 0.9677 |
| mean | 0.3387 |
| std | 0.2983 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 146 | 614.2944 | 0.1180 |
| 2 | volatility_60 | 260 | 602.9138 | 0.1158 |
| 3 | ridge_pred_future_log_return | 227 | 443.6107 | 0.0852 |
| 4 | ma_gap_60 | 108 | 298.0043 | 0.0572 |
| 5 | volume_change_20 | 224 | 290.5989 | 0.0558 |
| 6 | volatility_20 | 124 | 231.8086 | 0.0445 |
| 7 | macd_hist_to_close | 128 | 220.6608 | 0.0424 |
| 8 | volatility_5 | 189 | 202.3941 | 0.0389 |
| 9 | ma_gap_120 | 113 | 166.1688 | 0.0319 |
| 10 | ema_12_to_ema_26 | 91 | 156.6709 | 0.0301 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-28 |
| Train End | 2024-01-19 |
| Test Start | 2024-02-05 |
| Test End | 2024-05-09 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5227 |
| Precision | 0.5370 |
| Recall | 0.8788 |
| F1 | 0.6667 |
| ROC-AUC | 0.6909 |
| Log Loss | 0.8717 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.0701 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5425 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 25], [4, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0274 |
| p25 | 0.0961 |
| median | 0.1940 |
| p75 | 0.3371 |
| max | 0.7410 |
| mean | 0.2466 |
| std | 0.1797 |
| count_ge_threshold | 54 |
| count_lt_threshold | 9 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 310 | 759.9747 | 0.1453 |
| 2 | return_60 | 207 | 522.9248 | 0.1000 |
| 3 | ridge_pred_future_log_return | 161 | 521.7459 | 0.0997 |
| 4 | macd_hist_to_close | 166 | 385.2164 | 0.0736 |
| 5 | macd_signal_to_close | 217 | 265.2217 | 0.0507 |
| 6 | volatility_20 | 156 | 250.3515 | 0.0479 |
| 7 | close_to_high | 187 | 248.7002 | 0.0475 |
| 8 | ma_gap_60 | 114 | 236.1906 | 0.0452 |
| 9 | ridge_pred_future_return | 43 | 215.4602 | 0.0412 |
| 10 | volatility_10 | 138 | 207.7208 | 0.0397 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-04-04 |
| Train End | 2024-04-22 |
| Test Start | 2024-05-10 |
| Test End | 2024-08-07 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5946 |
| Precision | 0.4643 |
| Recall | 1.0000 |
| F1 | 0.6341 |
| ROC-AUC | 0.4844 |
| Log Loss | 0.9087 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.1674 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5913 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 30], [0, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0431 |
| p25 | 0.3083 |
| median | 0.5170 |
| p75 | 0.7161 |
| max | 0.9361 |
| mean | 0.5097 |
| std | 0.2557 |
| count_ge_threshold | 56 |
| count_lt_threshold | 7 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 306 | 1033.5636 | 0.1934 |
| 2 | return_60 | 223 | 520.1961 | 0.0974 |
| 3 | volatility_60 | 245 | 450.7653 | 0.0844 |
| 4 | ridge_pred_future_log_return | 217 | 380.9727 | 0.0713 |
| 5 | ma_gap_120 | 158 | 281.2040 | 0.0526 |
| 6 | macd_hist_to_close | 228 | 276.2940 | 0.0517 |
| 7 | ridge_pred_margin_to_threshold | 96 | 262.9012 | 0.0492 |
| 8 | ma_gap_60 | 123 | 241.8545 | 0.0453 |
| 9 | volatility_10 | 182 | 238.3466 | 0.0446 |
| 10 | volatility_20 | 185 | 198.0411 | 0.0371 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-07-06 |
| Train End | 2024-07-24 |
| Test Start | 2024-08-08 |
| Test End | 2024-11-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6658 |
| Precision | 0.6571 |
| Recall | 0.7188 |
| F1 | 0.6866 |
| ROC-AUC | 0.7550 |
| Log Loss | 0.6985 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.5957 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5726 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 12], [9, 23]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0023 |
| p25 | 0.1388 |
| median | 0.6519 |
| p75 | 0.9289 |
| max | 0.9845 |
| mean | 0.5681 |
| std | 0.3839 |
| count_ge_threshold | 35 |
| count_lt_threshold | 28 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 165 | 595.4125 | 0.1121 |
| 2 | volatility_60 | 305 | 558.0662 | 0.1051 |
| 3 | ma_gap_120 | 163 | 418.4213 | 0.0788 |
| 4 | rsi_14 | 96 | 345.3072 | 0.0650 |
| 5 | macd_signal_to_close | 124 | 331.7171 | 0.0625 |
| 6 | return_60 | 177 | 317.1340 | 0.0597 |
| 7 | ridge_pred_margin_to_threshold | 137 | 304.9932 | 0.0574 |
| 8 | ridge_pred_future_return | 43 | 240.7035 | 0.0453 |
| 9 | volatility_5 | 167 | 233.6787 | 0.0440 |
| 10 | macd_hist_to_close | 139 | 196.1282 | 0.0369 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-10-07 |
| Train End | 2024-10-25 |
| Test Start | 2024-11-12 |
| Test End | 2025-02-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.5208 |
| Precision | 1.0000 |
| Recall | 0.0417 |
| F1 | 0.0800 |
| ROC-AUC | 0.6506 |
| Log Loss | 0.6743 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.8221 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6792 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[39, 0], [23, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0132 |
| p25 | 0.1390 |
| median | 0.3128 |
| p75 | 0.5401 |
| max | 0.8456 |
| mean | 0.3531 |
| std | 0.2342 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 254 | 885.3625 | 0.1662 |
| 2 | macd_signal_to_close | 205 | 596.2148 | 0.1119 |
| 3 | volatility_60 | 249 | 552.9842 | 0.1038 |
| 4 | ridge_pred_future_log_return | 204 | 535.2800 | 0.1005 |
| 5 | return_60 | 150 | 281.7707 | 0.0529 |
| 6 | ma_gap_60 | 119 | 277.0768 | 0.0520 |
| 7 | macd_to_close | 112 | 205.7091 | 0.0386 |
| 8 | ma_gap_120 | 97 | 198.4706 | 0.0373 |
| 9 | ridge_pred_future_return | 42 | 185.6329 | 0.0348 |
| 10 | ema_12_to_ema_26 | 78 | 181.0828 | 0.0340 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-01-12 |
| Train End | 2025-01-30 |
| Test Start | 2025-02-17 |
| Test End | 2025-05-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6174 |
| Precision | 0.6400 |
| Recall | 0.5161 |
| F1 | 0.5714 |
| ROC-AUC | 0.7127 |
| Log Loss | 0.9233 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.8877 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6681 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 9], [15, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1007 |
| p25 | 0.6836 |
| median | 0.8361 |
| p75 | 0.9389 |
| max | 0.9963 |
| mean | 0.7744 |
| std | 0.2273 |
| count_ge_threshold | 25 |
| count_lt_threshold | 38 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_60 | 257 | 703.1226 | 0.1356 |
| 2 | ma_gap_120 | 246 | 581.4796 | 0.1121 |
| 3 | volatility_60 | 249 | 578.1284 | 0.1115 |
| 4 | macd_signal_to_close | 142 | 554.9798 | 0.1070 |
| 5 | volatility_20 | 218 | 325.0198 | 0.0627 |
| 6 | volume_change_5 | 178 | 308.1367 | 0.0594 |
| 7 | ema_12_to_ema_26 | 108 | 191.0762 | 0.0368 |
| 8 | volume_change_20 | 150 | 189.3571 | 0.0365 |
| 9 | macd_to_close | 56 | 159.1003 | 0.0307 |
| 10 | volatility_10 | 169 | 158.5130 | 0.0306 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-04-13 |
| Train End | 2025-05-07 |
| Test Start | 2025-05-22 |
| Test End | 2025-08-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3810 |
| Balanced Accuracy | 0.5568 |
| Precision | 1.0000 |
| Recall | 0.1136 |
| F1 | 0.2041 |
| ROC-AUC | 0.7356 |
| Log Loss | 0.6209 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.8881 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6856 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [39, 5]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0648 |
| p25 | 0.3523 |
| median | 0.5190 |
| p75 | 0.7615 |
| max | 0.9594 |
| mean | 0.5445 |
| std | 0.2565 |
| count_ge_threshold | 5 |
| count_lt_threshold | 58 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 231 | 569.7699 | 0.1089 |
| 2 | return_60 | 224 | 559.7551 | 0.1070 |
| 3 | volatility_20 | 113 | 391.0434 | 0.0747 |
| 4 | macd_signal_to_close | 168 | 368.0429 | 0.0703 |
| 5 | return_20 | 152 | 362.4932 | 0.0693 |
| 6 | ema_12_to_ema_26 | 174 | 353.7117 | 0.0676 |
| 7 | ma_gap_60 | 130 | 257.5048 | 0.0492 |
| 8 | ma_gap_20 | 88 | 246.6193 | 0.0471 |
| 9 | volume_z_20 | 123 | 242.4271 | 0.0463 |
| 10 | ma_gap_120 | 101 | 239.8496 | 0.0458 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-14 |
| Train End | 2025-08-05 |
| Test Start | 2025-08-21 |
| Test End | 2025-11-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3016 |
| Balanced Accuracy | 0.5217 |
| Precision | 1.0000 |
| Recall | 0.0435 |
| F1 | 0.0833 |
| ROC-AUC | 0.5205 |
| Log Loss | 1.1683 |
| Baseline Accuracy | 0.7302 |
| Decision Threshold | 0.9681 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5312 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 0], [44, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0040 |
| p25 | 0.1339 |
| median | 0.4210 |
| p75 | 0.7834 |
| max | 0.9792 |
| mean | 0.4466 |
| std | 0.3218 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 281 | 686.9680 | 0.1289 |
| 2 | ridge_pred_future_log_return | 226 | 628.8331 | 0.1180 |
| 3 | volatility_60 | 247 | 438.1525 | 0.0822 |
| 4 | macd_signal_to_close | 190 | 387.8834 | 0.0728 |
| 5 | ma_gap_60 | 77 | 348.7212 | 0.0654 |
| 6 | ridge_pred_future_return | 67 | 271.4248 | 0.0509 |
| 7 | ma_gap_20 | 89 | 265.1210 | 0.0497 |
| 8 | volatility_20 | 189 | 253.4035 | 0.0475 |
| 9 | high_low_range | 179 | 194.8637 | 0.0366 |
| 10 | volatility_10 | 155 | 179.8283 | 0.0337 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-17 |
| Train End | 2025-11-07 |
| Test Start | 2025-11-25 |
| Test End | 2026-02-27 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.5263 |
| Precision | 1.0000 |
| Recall | 0.0526 |
| F1 | 0.1000 |
| ROC-AUC | 0.6579 |
| Log Loss | 2.2152 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.5000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5033 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 0], [36, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0008 |
| p25 | 0.0073 |
| median | 0.0162 |
| p75 | 0.0307 |
| max | 0.9702 |
| mean | 0.0690 |
| std | 0.1725 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 251 | 962.3659 | 0.1803 |
| 2 | volatility_60 | 418 | 884.1727 | 0.1657 |
| 3 | ma_gap_60 | 173 | 443.2566 | 0.0831 |
| 4 | ridge_pred_future_log_return | 213 | 344.8651 | 0.0646 |
| 5 | volatility_20 | 211 | 332.6609 | 0.0623 |
| 6 | high_low_range | 170 | 223.8412 | 0.0419 |
| 7 | volatility_10 | 230 | 217.4078 | 0.0407 |
| 8 | ma_gap_120 | 67 | 212.5427 | 0.0398 |
| 9 | return_2 | 149 | 187.6986 | 0.0352 |
| 10 | volatility_5 | 147 | 186.0159 | 0.0349 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-22 |
| Train End | 2026-02-12 |
| Test Start | 2026-03-02 |
| Test End | 2026-06-03 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4227 |
| Precision | 0.6071 |
| Recall | 0.3953 |
| F1 | 0.4789 |
| ROC-AUC | 0.3721 |
| Log Loss | 2.0364 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.1355 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6451 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 11], [26, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0067 |
| p25 | 0.0207 |
| median | 0.0419 |
| p75 | 0.4474 |
| max | 0.9295 |
| mean | 0.2450 |
| std | 0.2994 |
| count_ge_threshold | 28 |
| count_lt_threshold | 35 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 385 | 710.3601 | 0.1331 |
| 2 | macd_signal_to_close | 182 | 604.4146 | 0.1132 |
| 3 | ma_gap_120 | 235 | 580.6640 | 0.1088 |
| 4 | return_20 | 207 | 498.4532 | 0.0934 |
| 5 | macd_hist_to_close | 177 | 455.4055 | 0.0853 |
| 6 | return_60 | 152 | 384.8372 | 0.0721 |
| 7 | volatility_10 | 266 | 303.8104 | 0.0569 |
| 8 | macd_to_close | 111 | 199.1442 | 0.0373 |
| 9 | ema_12_to_ema_26 | 104 | 178.9829 | 0.0335 |
| 10 | volatility_5 | 172 | 169.0679 | 0.0317 |
