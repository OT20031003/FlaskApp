# Direction Validation Report: ^GSPC

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^GSPC |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-15 15:36:03 NDT |
| Last Date | 2026-07-15 |
| Last Close | 7570.1099 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.2753 |
| Probability Down | 0.7247 |
| Decision Threshold | 0.9735 |
| Model Valid | False |
| Validation Method | purged_rolling_walk_forward |
| Target Return Threshold | 0.0000 |
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
| confusion_matrix_sum | [[229, 413], [460, 725]] | N/A | N/A | N/A |
| accuracy | 0.5222 | 0.1715 | 0.1111 | 0.8254 |
| balanced_accuracy | 0.5573 | 0.0832 | 0.4324 | 0.7889 |
| precision | 0.6719 | 0.2282 | 0.0000 | 1.0000 |
| recall | 0.6575 | 0.3333 | 0.0000 | 1.0000 |
| f1 | 0.5809 | 0.2288 | 0.0000 | 0.9043 |
| roc_auc | 0.6097 | 0.1332 | 0.3794 | 0.8631 |
| log_loss | 1.2960 | 0.6671 | 0.3331 | 3.1333 |
| baseline_accuracy | 0.6836 | 0.1237 | 0.5238 | 0.9048 |
| decision_threshold | 0.4722 | 0.3490 | 0.0100 | 0.9876 |
| calibration_score | 0.5791 | 0.0710 | 0.4972 | 0.7781 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 304 | 961.8454 | 0.1811 |
| 2 | return_60 | 249 | 647.0857 | 0.1218 |
| 3 | volatility_20 | 301 | 447.5288 | 0.0842 |
| 4 | macd_hist_to_close | 184 | 419.0654 | 0.0789 |
| 5 | log_return_1 | 156 | 357.4685 | 0.0673 |
| 6 | ma_gap_120 | 146 | 322.1903 | 0.0607 |
| 7 | macd_signal_to_close | 142 | 227.0953 | 0.0428 |
| 8 | ridge_pred_future_log_return | 164 | 221.6196 | 0.0417 |
| 9 | return_20 | 188 | 197.1985 | 0.0371 |
| 10 | return_5 | 158 | 195.7225 | 0.0368 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-01-22 | 2019-04-22 | 0.1111 | 0.5088 | 0.8567 | 0.9048 | 0.7571 | ok |
| 2 | ok | 2019-04-23 | 2019-07-22 | 0.5556 | 0.5000 | 0.6429 | 0.5556 | 0.0250 | ok |
| 3 | ok | 2019-07-23 | 2019-10-18 | 0.5556 | 0.5000 | 0.5612 | 0.5556 | 0.0100 | ok |
| 4 | ok | 2019-10-21 | 2020-01-21 | 0.8254 | 0.5000 | 0.7203 | 0.8254 | 0.0241 | ok |
| 5 | ok | 2020-01-22 | 2020-04-21 | 0.5079 | 0.4324 | 0.5483 | 0.5873 | 0.1751 | ok |
| 6 | ok | 2020-04-22 | 2020-07-21 | 0.3810 | 0.6176 | 0.7990 | 0.8095 | 0.9876 | ok |
| 7 | ok | 2020-07-22 | 2020-10-19 | 0.4921 | 0.5657 | 0.6688 | 0.6190 | 0.9378 | ok |
| 8 | ok | 2020-10-20 | 2021-01-20 | 0.1905 | 0.5446 | 0.4133 | 0.8889 | 0.5834 | ok |
| 9 | ok | 2021-01-21 | 2021-04-21 | 0.7937 | 0.7889 | 0.8432 | 0.7143 | 0.2661 | ok |
| 10 | ok | 2021-04-22 | 2021-07-21 | 0.8095 | 0.7377 | 0.8631 | 0.7937 | 0.7696 | ok |
| 11 | ok | 2021-07-22 | 2021-10-19 | 0.6032 | 0.5192 | 0.3794 | 0.5873 | 0.1351 | ok |
| 12 | ok | 2021-10-20 | 2022-01-19 | 0.5397 | 0.5197 | 0.5626 | 0.5238 | 0.6839 | ok |
| 13 | ok | 2022-01-20 | 2022-04-20 | 0.3333 | 0.5000 | 0.5986 | 0.6667 | 0.4536 | ok |
| 14 | ok | 2022-04-21 | 2022-07-21 | 0.5397 | 0.5242 | 0.5263 | 0.5238 | 0.9342 | ok |
| 15 | ok | 2022-07-22 | 2022-10-19 | 0.5238 | 0.5880 | 0.5967 | 0.6349 | 0.5484 | ok |
| 16 | ok | 2022-10-20 | 2023-01-20 | 0.6349 | 0.6071 | 0.7392 | 0.6667 | 0.4215 | ok |
| 17 | ok | 2023-01-23 | 2023-04-21 | 0.5556 | 0.5036 | 0.4929 | 0.5556 | 0.0424 | ok |
| 18 | ok | 2023-04-24 | 2023-07-26 | 0.6825 | 0.5927 | 0.6329 | 0.8254 | 0.0827 | ok |
| 19 | ok | 2023-07-27 | 2023-10-24 | 0.6190 | 0.6441 | 0.7616 | 0.6508 | 0.0453 | ok |
| 20 | ok | 2023-10-25 | 2024-01-25 | 0.4127 | 0.6009 | 0.6199 | 0.9048 | 0.1498 | ok |
| 21 | ok | 2024-01-26 | 2024-04-25 | 0.6825 | 0.5000 | 0.4930 | 0.6825 | 0.5000 | ok |
| 22 | ok | 2024-04-26 | 2024-07-26 | 0.4444 | 0.5128 | 0.5616 | 0.6825 | 0.1363 | ok |
| 23 | ok | 2024-07-29 | 2024-10-24 | 0.3968 | 0.4512 | 0.4291 | 0.6825 | 0.9313 | ok |
| 24 | ok | 2024-10-25 | 2025-01-28 | 0.7302 | 0.6489 | 0.6304 | 0.6349 | 0.4425 | ok |
| 25 | ok | 2025-01-29 | 2025-04-29 | 0.5238 | 0.5571 | 0.5102 | 0.5556 | 0.9777 | ok |
| 26 | ok | 2025-04-30 | 2025-07-30 | 0.4444 | 0.6875 | 0.7066 | 0.8889 | 0.9561 | ok |
| 27 | ok | 2025-07-31 | 2025-10-28 | 0.2063 | 0.5000 | 0.5400 | 0.7937 | 0.9867 | ok |
| 28 | ok | 2025-10-29 | 2026-01-29 | 0.4921 | 0.4418 | 0.3929 | 0.5873 | 0.4575 | ok |
| 29 | ok | 2026-01-30 | 2026-04-30 | 0.5556 | 0.5682 | 0.5909 | 0.5238 | 0.2717 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-01-04 |
| Train End | 2019-01-04 |
| Test Start | 2019-01-22 |
| Test End | 2019-04-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.1111 |
| Balanced Accuracy | 0.5088 |
| Precision | 1.0000 |
| Recall | 0.0175 |
| F1 | 0.0345 |
| ROC-AUC | 0.8567 |
| Log Loss | 2.1948 |
| Baseline Accuracy | 0.9048 |
| Decision Threshold | 0.7571 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5406 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 0], [56, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0051 |
| p25 | 0.0252 |
| median | 0.0865 |
| p75 | 0.2515 |
| max | 0.8102 |
| mean | 0.1621 |
| std | 0.1835 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 402 | 1395.6885 | 0.2622 |
| 2 | ma_gap_120 | 226 | 630.4034 | 0.1184 |
| 3 | ma_gap_60 | 178 | 473.2821 | 0.0889 |
| 4 | ridge_pred_future_log_return | 200 | 326.6305 | 0.0614 |
| 5 | macd_hist_to_close | 143 | 272.0321 | 0.0511 |
| 6 | volume_change_5 | 301 | 264.1401 | 0.0496 |
| 7 | return_20 | 168 | 209.6166 | 0.0394 |
| 8 | volume_change_20 | 125 | 198.2536 | 0.0372 |
| 9 | return_60 | 82 | 195.1587 | 0.0367 |
| 10 | return_10 | 90 | 142.6444 | 0.0268 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-04-05 |
| Train End | 2019-04-05 |
| Test Start | 2019-04-23 |
| Test End | 2019-07-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.5556 |
| Recall | 1.0000 |
| F1 | 0.7143 |
| ROC-AUC | 0.6429 |
| Log Loss | 1.0285 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.0250 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5202 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 28], [0, 35]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0450 |
| p25 | 0.5879 |
| median | 0.8884 |
| p75 | 0.9848 |
| max | 0.9944 |
| mean | 0.7603 |
| std | 0.2695 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 482 | 1455.8233 | 0.2731 |
| 2 | ma_gap_120 | 227 | 633.1629 | 0.1188 |
| 3 | ma_gap_60 | 191 | 537.9110 | 0.1009 |
| 4 | return_60 | 139 | 440.3195 | 0.0826 |
| 5 | volume_change_20 | 176 | 298.2715 | 0.0560 |
| 6 | volume_change_5 | 284 | 222.5719 | 0.0418 |
| 7 | volume_z_20 | 135 | 194.6606 | 0.0365 |
| 8 | macd_hist_to_close | 109 | 164.2445 | 0.0308 |
| 9 | ridge_pred_future_log_return | 107 | 132.5549 | 0.0249 |
| 10 | return_20 | 119 | 125.7893 | 0.0236 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-07-06 |
| Train End | 2019-07-08 |
| Test Start | 2019-07-23 |
| Test End | 2019-10-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.5556 |
| Recall | 1.0000 |
| F1 | 0.7143 |
| ROC-AUC | 0.5612 |
| Log Loss | 1.0282 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.0100 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5062 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 28], [0, 35]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0112 |
| p25 | 0.2784 |
| median | 0.6801 |
| p75 | 0.9025 |
| max | 0.9830 |
| mean | 0.5905 |
| std | 0.3396 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 469 | 1183.4426 | 0.2371 |
| 2 | volatility_20 | 185 | 587.9218 | 0.1178 |
| 3 | ridge_pred_future_log_return | 230 | 420.0897 | 0.0842 |
| 4 | macd_hist_to_close | 172 | 363.5933 | 0.0729 |
| 5 | ma_gap_60 | 206 | 330.0597 | 0.0661 |
| 6 | ma_gap_120 | 123 | 310.7733 | 0.0623 |
| 7 | macd_signal_to_close | 119 | 250.7688 | 0.0502 |
| 8 | volume_change_20 | 150 | 227.3040 | 0.0455 |
| 9 | ma_gap_20 | 86 | 113.6788 | 0.0228 |
| 10 | close_to_ema_12 | 57 | 95.4451 | 0.0191 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-10-04 |
| Train End | 2019-10-04 |
| Test Start | 2019-10-21 |
| Test End | 2020-01-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.8254 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.8254 |
| Recall | 1.0000 |
| F1 | 0.9043 |
| ROC-AUC | 0.7203 |
| Log Loss | 0.4897 |
| Baseline Accuracy | 0.8254 |
| Decision Threshold | 0.0241 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 11], [0, 52]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1820 |
| p25 | 0.4863 |
| median | 0.8856 |
| p75 | 0.9584 |
| max | 0.9961 |
| mean | 0.7353 |
| std | 0.2700 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 446 | 1216.7872 | 0.2377 |
| 2 | macd_signal_to_close | 154 | 490.3291 | 0.0958 |
| 3 | ma_gap_120 | 167 | 400.6885 | 0.0783 |
| 4 | volume_change_20 | 134 | 303.2289 | 0.0592 |
| 5 | ridge_pred_future_log_return | 133 | 269.6245 | 0.0527 |
| 6 | macd_hist_to_close | 138 | 255.5617 | 0.0499 |
| 7 | return_10 | 86 | 244.9569 | 0.0479 |
| 8 | ema_12_to_ema_26 | 107 | 228.1019 | 0.0446 |
| 9 | volatility_20 | 125 | 213.3792 | 0.0417 |
| 10 | close_to_ema_26 | 59 | 184.6444 | 0.0361 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-01-04 |
| Train End | 2020-01-06 |
| Test Start | 2020-01-22 |
| Test End | 2020-04-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.4324 |
| Precision | 0.5517 |
| Recall | 0.8649 |
| F1 | 0.6737 |
| ROC-AUC | 0.5483 |
| Log Loss | 1.2152 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.1751 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5389 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 26], [5, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0209 |
| p25 | 0.7490 |
| median | 0.9010 |
| p75 | 0.9566 |
| max | 0.9937 |
| mean | 0.7781 |
| std | 0.2657 |
| count_ge_threshold | 58 |
| count_lt_threshold | 5 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 313 | 1021.7700 | 0.1945 |
| 2 | volatility_20 | 221 | 653.0804 | 0.1243 |
| 3 | rsi_14 | 83 | 326.0203 | 0.0621 |
| 4 | ma_gap_60 | 144 | 311.0358 | 0.0592 |
| 5 | volume_change_20 | 165 | 305.0716 | 0.0581 |
| 6 | ma_gap_120 | 189 | 296.7578 | 0.0565 |
| 7 | return_60 | 181 | 252.4919 | 0.0481 |
| 8 | ema_12_to_ema_26 | 132 | 234.6782 | 0.0447 |
| 9 | macd_signal_to_close | 119 | 194.1495 | 0.0370 |
| 10 | return_5 | 164 | 178.4254 | 0.0340 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-04-06 |
| Train End | 2020-04-06 |
| Test Start | 2020-04-22 |
| Test End | 2020-07-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3810 |
| Balanced Accuracy | 0.6176 |
| Precision | 1.0000 |
| Recall | 0.2353 |
| F1 | 0.3810 |
| ROC-AUC | 0.7990 |
| Log Loss | 0.5051 |
| Baseline Accuracy | 0.8095 |
| Decision Threshold | 0.9876 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4972 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 0], [39, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.7102 |
| p25 | 0.9303 |
| median | 0.9715 |
| p75 | 0.9844 |
| max | 0.9965 |
| mean | 0.9448 |
| std | 0.0604 |
| count_ge_threshold | 12 |
| count_lt_threshold | 51 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 468 | 1242.4996 | 0.2375 |
| 2 | volume_change_20 | 271 | 452.8989 | 0.0866 |
| 3 | return_60 | 160 | 427.1304 | 0.0817 |
| 4 | ridge_pred_future_log_return | 206 | 412.8000 | 0.0789 |
| 5 | volatility_20 | 209 | 379.0153 | 0.0725 |
| 6 | return_5 | 163 | 267.4552 | 0.0511 |
| 7 | ma_gap_60 | 118 | 210.3693 | 0.0402 |
| 8 | macd_hist_to_close | 135 | 170.9609 | 0.0327 |
| 9 | high_low_range | 131 | 140.8141 | 0.0269 |
| 10 | close_to_ema_26 | 71 | 139.6669 | 0.0267 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-07-06 |
| Train End | 2020-07-07 |
| Test Start | 2020-07-22 |
| Test End | 2020-10-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.5657 |
| Precision | 0.7692 |
| Recall | 0.2564 |
| F1 | 0.3846 |
| ROC-AUC | 0.6688 |
| Log Loss | 0.7382 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.9378 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6185 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 3], [29, 10]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0492 |
| p25 | 0.4361 |
| median | 0.7346 |
| p75 | 0.9167 |
| max | 0.9955 |
| mean | 0.6743 |
| std | 0.2822 |
| count_ge_threshold | 13 |
| count_lt_threshold | 50 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 347 | 851.1125 | 0.1579 |
| 2 | ma_gap_120 | 277 | 660.2563 | 0.1225 |
| 3 | volume_change_20 | 243 | 512.4651 | 0.0951 |
| 4 | volatility_60 | 251 | 495.1868 | 0.0919 |
| 5 | ridge_pred_future_log_return | 209 | 388.6490 | 0.0721 |
| 6 | macd_hist_to_close | 171 | 311.0591 | 0.0577 |
| 7 | rsi_14 | 146 | 281.0073 | 0.0521 |
| 8 | macd_signal_to_close | 134 | 254.5235 | 0.0472 |
| 9 | close_to_ema_12 | 53 | 215.7118 | 0.0400 |
| 10 | ma_gap_60 | 181 | 207.5572 | 0.0385 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-10-04 |
| Train End | 2020-10-05 |
| Test Start | 2020-10-20 |
| Test End | 2021-01-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.1905 |
| Balanced Accuracy | 0.5446 |
| Precision | 1.0000 |
| Recall | 0.0893 |
| F1 | 0.1639 |
| ROC-AUC | 0.4133 |
| Log Loss | 1.6381 |
| Baseline Accuracy | 0.8889 |
| Decision Threshold | 0.5834 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6448 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 0], [51, 5]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0148 |
| p25 | 0.0922 |
| median | 0.1986 |
| p75 | 0.3347 |
| max | 0.9400 |
| mean | 0.2421 |
| std | 0.1906 |
| count_ge_threshold | 5 |
| count_lt_threshold | 58 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_hist_to_close | 220 | 731.2265 | 0.1414 |
| 2 | ma_gap_120 | 245 | 635.3023 | 0.1228 |
| 3 | volatility_60 | 156 | 405.8491 | 0.0785 |
| 4 | return_60 | 261 | 400.0273 | 0.0773 |
| 5 | volatility_20 | 182 | 389.5087 | 0.0753 |
| 6 | volume_change_20 | 216 | 313.7470 | 0.0607 |
| 7 | ma_gap_60 | 170 | 284.3329 | 0.0550 |
| 8 | macd_signal_to_close | 144 | 247.3381 | 0.0478 |
| 9 | volume_change_5 | 151 | 187.6610 | 0.0363 |
| 10 | return_10 | 93 | 180.1554 | 0.0348 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-01-07 |
| Train End | 2021-01-05 |
| Test Start | 2021-01-21 |
| Test End | 2021-04-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7937 |
| Balanced Accuracy | 0.7889 |
| Precision | 0.9000 |
| Recall | 0.8000 |
| F1 | 0.8471 |
| ROC-AUC | 0.8432 |
| Log Loss | 0.6730 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.2661 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7340 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 4], [9, 36]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0221 |
| p25 | 0.1497 |
| median | 0.4055 |
| p75 | 0.6983 |
| max | 0.9886 |
| mean | 0.4492 |
| std | 0.3050 |
| count_ge_threshold | 40 |
| count_lt_threshold | 23 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 285 | 869.1836 | 0.1685 |
| 2 | volatility_60 | 311 | 731.4946 | 0.1418 |
| 3 | ma_gap_120 | 259 | 591.5179 | 0.1147 |
| 4 | volume_change_20 | 175 | 509.8010 | 0.0988 |
| 5 | volume_z_20 | 170 | 389.4873 | 0.0755 |
| 6 | macd_hist_to_close | 250 | 345.9893 | 0.0671 |
| 7 | ema_12_to_ema_26 | 126 | 312.0137 | 0.0605 |
| 8 | close_to_low | 200 | 246.7448 | 0.0478 |
| 9 | return_10 | 67 | 147.9825 | 0.0287 |
| 10 | high_low_range | 80 | 119.0101 | 0.0231 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-04-08 |
| Train End | 2021-04-07 |
| Test Start | 2021-04-22 |
| Test End | 2021-07-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.8095 |
| Balanced Accuracy | 0.7377 |
| Precision | 0.8958 |
| Recall | 0.8600 |
| F1 | 0.8776 |
| ROC-AUC | 0.8631 |
| Log Loss | 0.3972 |
| Baseline Accuracy | 0.7937 |
| Decision Threshold | 0.7696 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7781 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 5], [7, 43]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1406 |
| p25 | 0.8036 |
| median | 0.9352 |
| p75 | 0.9755 |
| max | 0.9959 |
| mean | 0.8568 |
| std | 0.1802 |
| count_ge_threshold | 48 |
| count_lt_threshold | 15 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 345 | 737.1347 | 0.1373 |
| 2 | volatility_60 | 240 | 646.0688 | 0.1204 |
| 3 | ema_12_to_ema_26 | 137 | 447.7737 | 0.0834 |
| 4 | volume_z_20 | 159 | 444.7204 | 0.0828 |
| 5 | return_60 | 174 | 381.6974 | 0.0711 |
| 6 | macd_signal_to_close | 200 | 378.5862 | 0.0705 |
| 7 | rsi_14 | 89 | 365.8072 | 0.0681 |
| 8 | volume_change_20 | 203 | 311.4713 | 0.0580 |
| 9 | return_10 | 114 | 252.2282 | 0.0470 |
| 10 | intraday_return | 208 | 161.4346 | 0.0301 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-07-09 |
| Train End | 2021-07-07 |
| Test Start | 2021-07-22 |
| Test End | 2021-10-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5192 |
| Precision | 0.5968 |
| Recall | 1.0000 |
| F1 | 0.7475 |
| ROC-AUC | 0.3794 |
| Log Loss | 1.2513 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.1351 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6549 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 25], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1081 |
| p25 | 0.5915 |
| median | 0.8313 |
| p75 | 0.9390 |
| max | 0.9968 |
| mean | 0.7245 |
| std | 0.2591 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 391 | 970.9156 | 0.1799 |
| 2 | ema_12_to_ema_26 | 177 | 676.4608 | 0.1253 |
| 3 | macd_hist_to_close | 188 | 543.1625 | 0.1006 |
| 4 | macd_signal_to_close | 198 | 441.7534 | 0.0818 |
| 5 | ma_gap_120 | 234 | 372.6209 | 0.0690 |
| 6 | volume_z_20 | 95 | 331.7869 | 0.0615 |
| 7 | return_60 | 192 | 281.2315 | 0.0521 |
| 8 | volatility_20 | 115 | 223.9564 | 0.0415 |
| 9 | rsi_14 | 85 | 216.2735 | 0.0401 |
| 10 | close_to_low | 178 | 211.9709 | 0.0393 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-10-07 |
| Train End | 2021-10-05 |
| Test Start | 2021-10-20 |
| Test End | 2022-01-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5197 |
| Precision | 0.5345 |
| Recall | 0.9394 |
| F1 | 0.6813 |
| ROC-AUC | 0.5626 |
| Log Loss | 1.2222 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.6839 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6350 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[3, 27], [2, 31]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2952 |
| p25 | 0.8298 |
| median | 0.9178 |
| p75 | 0.9699 |
| max | 0.9888 |
| mean | 0.8739 |
| std | 0.1286 |
| count_ge_threshold | 58 |
| count_lt_threshold | 5 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 312 | 912.0786 | 0.1684 |
| 2 | ma_gap_120 | 296 | 733.7918 | 0.1355 |
| 3 | ema_12_to_ema_26 | 170 | 600.5526 | 0.1109 |
| 4 | volatility_20 | 179 | 425.0960 | 0.0785 |
| 5 | volume_z_20 | 193 | 352.2625 | 0.0650 |
| 6 | return_60 | 155 | 330.4216 | 0.0610 |
| 7 | ridge_pred_future_log_return | 197 | 250.9704 | 0.0463 |
| 8 | close_to_ema_26 | 78 | 196.4078 | 0.0363 |
| 9 | ma_gap_60 | 90 | 147.0912 | 0.0272 |
| 10 | macd_signal_to_close | 78 | 145.3874 | 0.0268 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-01-07 |
| Train End | 2022-01-04 |
| Test Start | 2022-01-20 |
| Test End | 2022-04-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3333 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.3333 |
| Recall | 1.0000 |
| F1 | 0.5000 |
| ROC-AUC | 0.5986 |
| Log Loss | 3.1333 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.4536 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5618 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 42], [0, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.8502 |
| p25 | 0.9847 |
| median | 0.9926 |
| p75 | 0.9965 |
| max | 0.9998 |
| mean | 0.9821 |
| std | 0.0275 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 315 | 1080.7649 | 0.1994 |
| 2 | volatility_60 | 363 | 941.8140 | 0.1738 |
| 3 | volatility_20 | 329 | 764.0792 | 0.1410 |
| 4 | volume_z_20 | 139 | 292.6473 | 0.0540 |
| 5 | return_60 | 127 | 272.0342 | 0.0502 |
| 6 | return_10 | 155 | 236.8205 | 0.0437 |
| 7 | ema_12_to_ema_26 | 193 | 228.3539 | 0.0421 |
| 8 | ridge_pred_future_log_return | 131 | 188.7638 | 0.0348 |
| 9 | macd_to_close | 86 | 165.4426 | 0.0305 |
| 10 | volume_change_20 | 78 | 121.7967 | 0.0225 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-04-07 |
| Train End | 2022-04-05 |
| Test Start | 2022-04-21 |
| Test End | 2022-07-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5242 |
| Precision | 0.5385 |
| Recall | 0.8485 |
| F1 | 0.6588 |
| ROC-AUC | 0.5263 |
| Log Loss | 1.6355 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.9342 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5318 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 24], [5, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.7626 |
| p25 | 0.9467 |
| median | 0.9674 |
| p75 | 0.9835 |
| max | 0.9935 |
| mean | 0.9534 |
| std | 0.0465 |
| count_ge_threshold | 52 |
| count_lt_threshold | 11 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 381 | 1282.0409 | 0.2424 |
| 2 | volatility_20 | 270 | 456.5318 | 0.0863 |
| 3 | return_10 | 129 | 349.1888 | 0.0660 |
| 4 | ma_gap_120 | 161 | 335.9373 | 0.0635 |
| 5 | ema_12_to_ema_26 | 202 | 315.1025 | 0.0596 |
| 6 | volume_change_20 | 237 | 299.3183 | 0.0566 |
| 7 | ridge_pred_future_log_return | 131 | 298.6424 | 0.0565 |
| 8 | return_60 | 163 | 248.0344 | 0.0469 |
| 9 | volume_z_20 | 148 | 208.6941 | 0.0395 |
| 10 | macd_signal_to_close | 155 | 194.2351 | 0.0367 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-07-08 |
| Train End | 2022-07-07 |
| Test Start | 2022-07-22 |
| Test End | 2022-10-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5880 |
| Precision | 0.4222 |
| Recall | 0.8261 |
| F1 | 0.5588 |
| ROC-AUC | 0.5967 |
| Log Loss | 1.3359 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.5484 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5383 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 26], [4, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0991 |
| p25 | 0.4787 |
| median | 0.8760 |
| p75 | 0.9657 |
| max | 0.9941 |
| mean | 0.7212 |
| std | 0.2861 |
| count_ge_threshold | 45 |
| count_lt_threshold | 18 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 376 | 1065.7562 | 0.2035 |
| 2 | volatility_20 | 347 | 648.6392 | 0.1239 |
| 3 | ma_gap_60 | 118 | 512.7133 | 0.0979 |
| 4 | ema_12_to_ema_26 | 213 | 408.5445 | 0.0780 |
| 5 | return_60 | 158 | 302.4028 | 0.0578 |
| 6 | macd_signal_to_close | 161 | 223.1467 | 0.0426 |
| 7 | volume_change_20 | 141 | 197.4708 | 0.0377 |
| 8 | volume_z_20 | 162 | 197.3126 | 0.0377 |
| 9 | return_2 | 179 | 162.0793 | 0.0310 |
| 10 | volume_change_5 | 123 | 141.4319 | 0.0270 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-10-06 |
| Train End | 2022-10-05 |
| Test Start | 2022-10-20 |
| Test End | 2023-01-20 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.6071 |
| Precision | 0.7436 |
| Recall | 0.6905 |
| F1 | 0.7160 |
| ROC-AUC | 0.7392 |
| Log Loss | 0.6084 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.4215 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5812 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 10], [13, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0777 |
| p25 | 0.3372 |
| median | 0.4947 |
| p75 | 0.7184 |
| max | 0.9011 |
| mean | 0.5130 |
| std | 0.2314 |
| count_ge_threshold | 39 |
| count_lt_threshold | 24 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 433 | 1163.3656 | 0.2283 |
| 2 | volatility_20 | 213 | 642.7910 | 0.1262 |
| 3 | volume_change_20 | 217 | 447.8510 | 0.0879 |
| 4 | macd_signal_to_close | 159 | 353.4109 | 0.0694 |
| 5 | ridge_pred_future_log_return | 161 | 223.9163 | 0.0439 |
| 6 | volume_change_5 | 178 | 223.7921 | 0.0439 |
| 7 | return_60 | 106 | 206.8217 | 0.0406 |
| 8 | ma_gap_60 | 93 | 171.3737 | 0.0336 |
| 9 | ema_12_to_ema_26 | 95 | 160.9156 | 0.0316 |
| 10 | macd_hist_to_close | 124 | 148.0281 | 0.0291 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-01-06 |
| Train End | 2023-01-05 |
| Test Start | 2023-01-23 |
| Test End | 2023-04-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5036 |
| Precision | 0.5574 |
| Recall | 0.9714 |
| F1 | 0.7083 |
| ROC-AUC | 0.4929 |
| Log Loss | 0.9845 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.0424 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5987 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 27], [1, 34]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0065 |
| p25 | 0.1945 |
| median | 0.3939 |
| p75 | 0.6111 |
| max | 0.8811 |
| mean | 0.4039 |
| std | 0.2555 |
| count_ge_threshold | 61 |
| count_lt_threshold | 2 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 579 | 1388.0915 | 0.2689 |
| 2 | volume_change_20 | 213 | 580.8420 | 0.1125 |
| 3 | ma_gap_60 | 148 | 464.8117 | 0.0900 |
| 4 | volatility_20 | 182 | 379.4810 | 0.0735 |
| 5 | return_60 | 193 | 282.2708 | 0.0547 |
| 6 | ridge_pred_future_log_return | 180 | 232.3892 | 0.0450 |
| 7 | macd_signal_to_close | 98 | 197.7993 | 0.0383 |
| 8 | macd_hist_to_close | 148 | 194.5865 | 0.0377 |
| 9 | high_low_range | 147 | 194.4783 | 0.0377 |
| 10 | log_return_1 | 145 | 131.6903 | 0.0255 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-04-08 |
| Train End | 2023-04-06 |
| Test Start | 2023-04-24 |
| Test End | 2023-07-26 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6825 |
| Balanced Accuracy | 0.5927 |
| Precision | 0.8636 |
| Recall | 0.7308 |
| F1 | 0.7917 |
| ROC-AUC | 0.6329 |
| Log Loss | 1.3704 |
| Baseline Accuracy | 0.8254 |
| Decision Threshold | 0.0827 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6095 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 6], [14, 38]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0156 |
| p25 | 0.0632 |
| median | 0.2248 |
| p75 | 0.5323 |
| max | 0.9631 |
| mean | 0.3217 |
| std | 0.2831 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 352 | 1274.0799 | 0.2442 |
| 2 | return_60 | 297 | 858.9405 | 0.1646 |
| 3 | ridge_pred_future_log_return | 225 | 415.9647 | 0.0797 |
| 4 | volume_change_20 | 278 | 406.7506 | 0.0780 |
| 5 | macd_signal_to_close | 176 | 222.2865 | 0.0426 |
| 6 | volatility_20 | 110 | 218.2188 | 0.0418 |
| 7 | ma_gap_60 | 120 | 212.0693 | 0.0406 |
| 8 | ema_12_to_ema_26 | 147 | 193.6378 | 0.0371 |
| 9 | macd_hist_to_close | 88 | 154.2286 | 0.0296 |
| 10 | log_return_1 | 166 | 149.9998 | 0.0287 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-07-08 |
| Train End | 2023-07-12 |
| Test Start | 2023-07-27 |
| Test End | 2023-10-24 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6441 |
| Precision | 0.4706 |
| Recall | 0.7273 |
| F1 | 0.5714 |
| ROC-AUC | 0.7616 |
| Log Loss | 0.7434 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.0453 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6667 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 18], [6, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0101 |
| p25 | 0.0206 |
| median | 0.0608 |
| p75 | 0.2252 |
| max | 0.9702 |
| mean | 0.1639 |
| std | 0.2150 |
| count_ge_threshold | 34 |
| count_lt_threshold | 29 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 261 | 1040.0344 | 0.2017 |
| 2 | volatility_60 | 310 | 776.0988 | 0.1505 |
| 3 | ma_gap_120 | 190 | 390.5451 | 0.0757 |
| 4 | macd_signal_to_close | 185 | 384.8854 | 0.0746 |
| 5 | volatility_20 | 231 | 339.0372 | 0.0657 |
| 6 | ma_gap_60 | 162 | 260.7156 | 0.0506 |
| 7 | macd_hist_to_close | 175 | 246.6292 | 0.0478 |
| 8 | volume_z_20 | 146 | 239.9950 | 0.0465 |
| 9 | volume_change_20 | 239 | 217.3167 | 0.0421 |
| 10 | return_2 | 112 | 153.6585 | 0.0298 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-10-06 |
| Train End | 2023-10-10 |
| Test Start | 2023-10-25 |
| Test End | 2024-01-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.6009 |
| Precision | 0.9545 |
| Recall | 0.3684 |
| F1 | 0.5316 |
| ROC-AUC | 0.6199 |
| Log Loss | 2.2980 |
| Baseline Accuracy | 0.9048 |
| Decision Threshold | 0.1498 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5244 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 1], [36, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0033 |
| p25 | 0.0135 |
| median | 0.0551 |
| p75 | 0.7546 |
| max | 0.9970 |
| mean | 0.3022 |
| std | 0.3989 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 319 | 920.4801 | 0.1744 |
| 2 | macd_hist_to_close | 219 | 759.5711 | 0.1439 |
| 3 | volatility_60 | 296 | 564.1819 | 0.1069 |
| 4 | ma_gap_120 | 212 | 356.6947 | 0.0676 |
| 5 | volatility_20 | 140 | 312.6925 | 0.0592 |
| 6 | return_60 | 113 | 279.0707 | 0.0529 |
| 7 | ridge_pred_future_log_return | 224 | 266.2726 | 0.0504 |
| 8 | ma_gap_60 | 180 | 207.3873 | 0.0393 |
| 9 | return_10 | 125 | 185.5247 | 0.0351 |
| 10 | volume_z_20 | 197 | 179.0255 | 0.0339 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-01-05 |
| Train End | 2024-01-10 |
| Test Start | 2024-01-26 |
| Test End | 2024-04-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6825 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6825 |
| Recall | 1.0000 |
| F1 | 0.8113 |
| ROC-AUC | 0.4930 |
| Log Loss | 1.2276 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.5000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 20], [0, 43]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.8054 |
| p25 | 0.9603 |
| median | 0.9781 |
| p75 | 0.9871 |
| max | 0.9968 |
| mean | 0.9649 |
| std | 0.0370 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 327 | 954.0608 | 0.1807 |
| 2 | macd_signal_to_close | 256 | 920.6361 | 0.1743 |
| 3 | macd_hist_to_close | 270 | 785.4562 | 0.1487 |
| 4 | volatility_20 | 199 | 384.8122 | 0.0729 |
| 5 | ma_gap_120 | 177 | 311.8747 | 0.0591 |
| 6 | return_60 | 189 | 228.4654 | 0.0433 |
| 7 | ridge_pred_future_log_return | 140 | 159.4079 | 0.0302 |
| 8 | volume_z_20 | 128 | 155.6646 | 0.0295 |
| 9 | ema_12_to_ema_26 | 69 | 138.7236 | 0.0263 |
| 10 | volume_change_20 | 137 | 137.7143 | 0.0261 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-04-06 |
| Train End | 2024-04-11 |
| Test Start | 2024-04-26 |
| Test End | 2024-07-26 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.5128 |
| Precision | 0.7000 |
| Recall | 0.3256 |
| F1 | 0.4444 |
| ROC-AUC | 0.5616 |
| Log Loss | 2.1846 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.1363 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5521 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 6], [29, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0012 |
| p25 | 0.0097 |
| median | 0.0259 |
| p75 | 0.5052 |
| max | 0.9800 |
| mean | 0.2296 |
| std | 0.3297 |
| count_ge_threshold | 20 |
| count_lt_threshold | 43 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 270 | 1036.5177 | 0.1923 |
| 2 | macd_signal_to_close | 211 | 562.0626 | 0.1043 |
| 3 | return_20 | 227 | 553.0830 | 0.1026 |
| 4 | return_60 | 274 | 466.7451 | 0.0866 |
| 5 | ridge_pred_future_log_return | 320 | 432.0170 | 0.0801 |
| 6 | ema_12_to_ema_26 | 161 | 381.4237 | 0.0708 |
| 7 | ma_gap_120 | 251 | 367.3423 | 0.0681 |
| 8 | volatility_20 | 167 | 322.0674 | 0.0597 |
| 9 | macd_hist_to_close | 171 | 292.7364 | 0.0543 |
| 10 | volume_change_20 | 177 | 152.6516 | 0.0283 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-07-08 |
| Train End | 2024-07-12 |
| Test Start | 2024-07-29 |
| Test End | 2024-10-24 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.4512 |
| Precision | 0.6190 |
| Recall | 0.3023 |
| F1 | 0.4062 |
| ROC-AUC | 0.4291 |
| Log Loss | 1.5270 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.9313 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5552 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 8], [30, 13]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0191 |
| p25 | 0.1182 |
| median | 0.4845 |
| p75 | 0.9786 |
| max | 0.9978 |
| mean | 0.5115 |
| std | 0.3891 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 466 | 1578.8528 | 0.2922 |
| 2 | return_20 | 297 | 677.9357 | 0.1254 |
| 3 | macd_signal_to_close | 152 | 396.9473 | 0.0735 |
| 4 | return_60 | 238 | 320.8900 | 0.0594 |
| 5 | ma_gap_120 | 195 | 279.9014 | 0.0518 |
| 6 | volatility_20 | 164 | 251.0355 | 0.0465 |
| 7 | ridge_pred_future_log_return | 195 | 232.5146 | 0.0430 |
| 8 | macd_hist_to_close | 128 | 231.9176 | 0.0429 |
| 9 | volume_change_20 | 214 | 225.6172 | 0.0417 |
| 10 | close_to_ema_12 | 111 | 149.9934 | 0.0278 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-10-06 |
| Train End | 2024-10-10 |
| Test Start | 2024-10-25 |
| Test End | 2025-01-28 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7302 |
| Balanced Accuracy | 0.6489 |
| Precision | 0.7170 |
| Recall | 0.9500 |
| F1 | 0.8172 |
| ROC-AUC | 0.6304 |
| Log Loss | 1.2819 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.4425 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5667 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[8, 15], [2, 38]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0013 |
| p25 | 0.8998 |
| median | 0.9837 |
| p75 | 0.9929 |
| max | 0.9994 |
| mean | 0.8021 |
| std | 0.3585 |
| count_ge_threshold | 53 |
| count_lt_threshold | 10 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 467 | 1213.7359 | 0.2236 |
| 2 | volatility_20 | 288 | 807.3719 | 0.1487 |
| 3 | ma_gap_60 | 200 | 724.5259 | 0.1335 |
| 4 | return_20 | 165 | 404.4338 | 0.0745 |
| 5 | ridge_pred_future_log_return | 295 | 316.0275 | 0.0582 |
| 6 | macd_signal_to_close | 135 | 286.1687 | 0.0527 |
| 7 | return_60 | 170 | 260.3733 | 0.0480 |
| 8 | ma_gap_20 | 153 | 222.7939 | 0.0410 |
| 9 | macd_hist_to_close | 174 | 202.4655 | 0.0373 |
| 10 | close_to_ema_12 | 105 | 163.3359 | 0.0301 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-01-06 |
| Train End | 2025-01-13 |
| Test Start | 2025-01-29 |
| Test End | 2025-04-29 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5571 |
| Precision | 0.4800 |
| Recall | 0.8571 |
| F1 | 0.6154 |
| ROC-AUC | 0.5102 |
| Log Loss | 2.4275 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9777 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6731 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 26], [4, 24]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.9064 |
| p25 | 0.9812 |
| median | 0.9898 |
| p75 | 0.9925 |
| max | 0.9983 |
| mean | 0.9825 |
| std | 0.0180 |
| count_ge_threshold | 50 |
| count_lt_threshold | 13 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 420 | 946.6031 | 0.1753 |
| 2 | volatility_60 | 314 | 889.9217 | 0.1648 |
| 3 | volatility_20 | 349 | 828.7727 | 0.1535 |
| 4 | return_20 | 239 | 438.9901 | 0.0813 |
| 5 | ridge_pred_future_log_return | 186 | 426.0196 | 0.0789 |
| 6 | rsi_14 | 95 | 275.5216 | 0.0510 |
| 7 | macd_signal_to_close | 74 | 194.4865 | 0.0360 |
| 8 | ma_gap_60 | 128 | 189.9471 | 0.0352 |
| 9 | macd_hist_to_close | 133 | 142.3982 | 0.0264 |
| 10 | volume_change_20 | 190 | 136.1316 | 0.0252 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-04-10 |
| Train End | 2025-04-14 |
| Test Start | 2025-04-30 |
| Test End | 2025-07-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.6875 |
| Precision | 1.0000 |
| Recall | 0.3750 |
| F1 | 0.5455 |
| ROC-AUC | 0.7066 |
| Log Loss | 0.3331 |
| Baseline Accuracy | 0.8889 |
| Decision Threshold | 0.9561 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5284 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 0], [35, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.5811 |
| p25 | 0.8380 |
| median | 0.9295 |
| p75 | 0.9767 |
| max | 0.9995 |
| mean | 0.8919 |
| std | 0.1073 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 378 | 1499.5860 | 0.2757 |
| 2 | volatility_20 | 257 | 687.0937 | 0.1263 |
| 3 | ma_gap_60 | 188 | 498.1126 | 0.0916 |
| 4 | volatility_60 | 293 | 458.1539 | 0.0842 |
| 5 | ridge_pred_future_log_return | 233 | 316.1819 | 0.0581 |
| 6 | macd_hist_to_close | 152 | 236.7886 | 0.0435 |
| 7 | macd_signal_to_close | 127 | 179.7418 | 0.0330 |
| 8 | ma_gap_120 | 119 | 163.8152 | 0.0301 |
| 9 | volume_change_20 | 158 | 155.1617 | 0.0285 |
| 10 | return_10 | 126 | 115.7494 | 0.0213 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-13 |
| Train End | 2025-07-16 |
| Test Start | 2025-07-31 |
| Test End | 2025-10-28 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2063 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5400 |
| Log Loss | 1.9856 |
| Baseline Accuracy | 0.7937 |
| Decision Threshold | 0.9867 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5119 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 0], [50, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0078 |
| p25 | 0.0317 |
| median | 0.0721 |
| p75 | 0.2650 |
| max | 0.7958 |
| mean | 0.1708 |
| std | 0.2025 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 326 | 1482.3359 | 0.2774 |
| 2 | volatility_20 | 211 | 489.6918 | 0.0916 |
| 3 | volatility_60 | 288 | 487.9778 | 0.0913 |
| 4 | macd_signal_to_close | 242 | 439.1377 | 0.0822 |
| 5 | ridge_pred_future_log_return | 230 | 307.7269 | 0.0576 |
| 6 | volume_change_20 | 112 | 264.8036 | 0.0496 |
| 7 | macd_hist_to_close | 167 | 238.7221 | 0.0447 |
| 8 | ma_gap_120 | 130 | 200.0845 | 0.0374 |
| 9 | volume_z_20 | 115 | 192.0647 | 0.0359 |
| 10 | close_to_ema_26 | 119 | 175.1713 | 0.0328 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-11 |
| Train End | 2025-10-14 |
| Test Start | 2025-10-29 |
| Test End | 2026-01-29 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.4418 |
| Precision | 0.5510 |
| Recall | 0.7297 |
| F1 | 0.6279 |
| ROC-AUC | 0.3929 |
| Log Loss | 0.9560 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.4575 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6011 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 22], [10, 27]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1742 |
| p25 | 0.5605 |
| median | 0.7229 |
| p75 | 0.8603 |
| max | 0.9461 |
| mean | 0.6633 |
| std | 0.2354 |
| count_ge_threshold | 49 |
| count_lt_threshold | 14 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 310 | 942.6692 | 0.1830 |
| 2 | macd_signal_to_close | 308 | 839.6244 | 0.1630 |
| 3 | volatility_60 | 307 | 552.9428 | 0.1073 |
| 4 | ridge_pred_future_log_return | 232 | 457.2185 | 0.0887 |
| 5 | volume_z_20 | 112 | 251.2700 | 0.0488 |
| 6 | ma_gap_120 | 96 | 224.8738 | 0.0436 |
| 7 | volatility_20 | 144 | 210.8456 | 0.0409 |
| 8 | close_to_ema_26 | 96 | 169.3205 | 0.0329 |
| 9 | ma_gap_60 | 71 | 150.3042 | 0.0292 |
| 10 | macd_hist_to_close | 140 | 150.2630 | 0.0292 |

### Fold 29

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-11 |
| Train End | 2026-01-14 |
| Test Start | 2026-01-30 |
| Test End | 2026-04-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5682 |
| Precision | 0.6667 |
| Recall | 0.3030 |
| F1 | 0.4167 |
| ROC-AUC | 0.5909 |
| Log Loss | 1.1709 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.2717 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5256 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 5], [23, 10]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0062 |
| p25 | 0.0356 |
| median | 0.1262 |
| p75 | 0.2493 |
| max | 0.9494 |
| mean | 0.2177 |
| std | 0.2537 |
| count_ge_threshold | 15 |
| count_lt_threshold | 48 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 397 | 1395.5325 | 0.2643 |
| 2 | macd_signal_to_close | 257 | 573.3806 | 0.1086 |
| 3 | volatility_20 | 270 | 510.9028 | 0.0967 |
| 4 | return_60 | 288 | 489.6510 | 0.0927 |
| 5 | ridge_pred_future_log_return | 172 | 272.8168 | 0.0517 |
| 6 | ma_gap_120 | 91 | 180.2792 | 0.0341 |
| 7 | volume_change_20 | 122 | 166.1030 | 0.0315 |
| 8 | volume_z_20 | 102 | 163.2508 | 0.0309 |
| 9 | ma_gap_60 | 82 | 154.9893 | 0.0294 |
| 10 | close_to_ema_12 | 44 | 143.1332 | 0.0271 |
