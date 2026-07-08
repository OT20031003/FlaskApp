# Direction Validation Report: MU

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | MU |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-07 10:05:01 JST |
| Last Date | 2026-07-06 |
| Last Close | 984.7500 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.6042 |
| Probability Down | 0.3958 |
| Decision Threshold | 0.8638 |
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
| beats_baseline_fold_ratio | 0.2759 | N/A | N/A | N/A |
| confusion_matrix_sum | [[507, 335], [567, 418]] | N/A | N/A | N/A |
| accuracy | 0.5063 | 0.1101 | 0.2540 | 0.6825 |
| balanced_accuracy | 0.5251 | 0.0871 | 0.3915 | 0.7688 |
| precision | 0.5366 | 0.2619 | 0.0000 | 1.0000 |
| recall | 0.4325 | 0.2970 | 0.0000 | 1.0000 |
| f1 | 0.4233 | 0.2261 | 0.0000 | 0.7723 |
| roc_auc | 0.5618 | 0.1257 | 0.3245 | 0.8333 |
| log_loss | 1.5269 | 0.5605 | 0.4406 | 3.0175 |
| baseline_accuracy | 0.6163 | 0.0888 | 0.5079 | 0.7937 |
| decision_threshold | 0.5064 | 0.3898 | 0.0103 | 0.9900 |
| calibration_score | 0.5966 | 0.0469 | 0.5213 | 0.6825 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 239 | 874.4878 | 0.1667 |
| 2 | ridge_pred_future_log_return | 195 | 684.9005 | 0.1306 |
| 3 | volatility_20 | 245 | 414.6625 | 0.0790 |
| 4 | return_60 | 217 | 392.1880 | 0.0748 |
| 5 | macd_signal_to_close | 247 | 381.7739 | 0.0728 |
| 6 | return_10 | 94 | 317.4307 | 0.0605 |
| 7 | macd_hist_to_close | 183 | 211.8636 | 0.0404 |
| 8 | volume_z_20 | 178 | 181.6330 | 0.0346 |
| 9 | ridge_pred_future_return | 42 | 168.7093 | 0.0322 |
| 10 | ma_gap_60 | 71 | 161.5332 | 0.0308 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-01-11 | 2019-04-11 | 0.4286 | 0.5714 | 0.6791 | 0.6667 | 0.9885 | ok |
| 2 | ok | 2019-04-12 | 2019-07-12 | 0.4444 | 0.4071 | 0.3245 | 0.5556 | 0.9900 | ok |
| 3 | ok | 2019-07-15 | 2019-10-10 | 0.6349 | 0.6187 | 0.7333 | 0.5397 | 0.0103 | ok |
| 4 | ok | 2019-10-11 | 2020-01-10 | 0.4444 | 0.5444 | 0.5630 | 0.7143 | 0.0702 | ok |
| 5 | ok | 2020-01-13 | 2020-04-13 | 0.6508 | 0.6626 | 0.7684 | 0.6032 | 0.1497 | ok |
| 6 | ok | 2020-04-14 | 2020-07-13 | 0.6349 | 0.5208 | 0.6325 | 0.6190 | 0.6434 | ok |
| 7 | ok | 2020-07-14 | 2020-10-09 | 0.3651 | 0.3915 | 0.4412 | 0.5397 | 0.9686 | ok |
| 8 | ok | 2020-10-12 | 2021-01-11 | 0.2540 | 0.5000 | 0.6383 | 0.7460 | 0.8713 | ok |
| 9 | ok | 2021-01-12 | 2021-04-13 | 0.4762 | 0.5000 | 0.4172 | 0.5238 | 0.9565 | ok |
| 10 | ok | 2021-04-14 | 2021-07-13 | 0.4286 | 0.4000 | 0.4889 | 0.7143 | 0.9218 | ok |
| 11 | ok | 2021-07-14 | 2021-10-11 | 0.3492 | 0.5000 | 0.5122 | 0.6508 | 0.2912 | ok |
| 12 | ok | 2021-10-12 | 2022-01-10 | 0.6825 | 0.7688 | 0.8333 | 0.7619 | 0.9196 | ok |
| 13 | ok | 2022-01-11 | 2022-04-11 | 0.5556 | 0.4889 | 0.6037 | 0.7143 | 0.9101 | ok |
| 14 | ok | 2022-04-12 | 2022-07-13 | 0.5238 | 0.5288 | 0.5879 | 0.5238 | 0.8270 | ok |
| 15 | ok | 2022-07-14 | 2022-10-11 | 0.5397 | 0.5240 | 0.5726 | 0.6190 | 0.3855 | ok |
| 16 | ok | 2022-10-12 | 2023-01-11 | 0.6190 | 0.6242 | 0.6643 | 0.5397 | 0.0570 | ok |
| 17 | ok | 2023-01-12 | 2023-04-13 | 0.5556 | 0.5324 | 0.5072 | 0.5714 | 0.2765 | ok |
| 18 | ok | 2023-04-14 | 2023-07-14 | 0.6190 | 0.6076 | 0.6141 | 0.5238 | 0.2285 | ok |
| 19 | ok | 2023-07-17 | 2023-10-12 | 0.4921 | 0.4818 | 0.4384 | 0.5238 | 0.0572 | ok |
| 20 | ok | 2023-10-13 | 2024-01-12 | 0.5397 | 0.4163 | 0.4593 | 0.6984 | 0.0771 | ok |
| 21 | ok | 2024-01-16 | 2024-04-15 | 0.6190 | 0.6528 | 0.7058 | 0.5714 | 0.1618 | ok |
| 22 | ok | 2024-04-16 | 2024-07-16 | 0.3968 | 0.4179 | 0.4337 | 0.5556 | 0.0373 | ok |
| 23 | ok | 2024-07-17 | 2024-10-14 | 0.5079 | 0.5000 | 0.5433 | 0.5079 | 0.1479 | ok |
| 24 | ok | 2024-10-15 | 2025-01-15 | 0.6032 | 0.5249 | 0.7173 | 0.5873 | 0.9763 | ok |
| 25 | ok | 2025-01-16 | 2025-04-16 | 0.5397 | 0.5607 | 0.5867 | 0.5556 | 0.9818 | ok |
| 26 | ok | 2025-04-17 | 2025-07-18 | 0.4762 | 0.4582 | 0.4163 | 0.6349 | 0.2051 | ok |
| 27 | ok | 2025-07-21 | 2025-10-16 | 0.4762 | 0.4138 | 0.3569 | 0.7937 | 0.5385 | ok |
| 28 | ok | 2025-10-17 | 2026-01-16 | 0.2540 | 0.5300 | 0.4369 | 0.7937 | 0.9726 | ok |
| 29 | ok | 2026-01-20 | 2026-04-20 | 0.5714 | 0.5803 | 0.6162 | 0.5238 | 0.0639 | ok |

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
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.5714 |
| Precision | 1.0000 |
| Recall | 0.1429 |
| F1 | 0.2500 |
| ROC-AUC | 0.6791 |
| Log Loss | 1.0643 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.9885 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6611 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 0], [36, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0178 |
| p25 | 0.0630 |
| median | 0.1700 |
| p75 | 0.8547 |
| max | 0.9959 |
| mean | 0.3758 |
| std | 0.3832 |
| count_ge_threshold | 6 |
| count_lt_threshold | 57 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 319 | 1135.8252 | 0.2135 |
| 2 | volatility_60 | 347 | 683.3852 | 0.1284 |
| 3 | ma_gap_120 | 279 | 577.3504 | 0.1085 |
| 4 | ridge_pred_future_log_return | 279 | 547.4160 | 0.1029 |
| 5 | macd_hist_to_close | 267 | 482.0178 | 0.0906 |
| 6 | return_10 | 157 | 261.7772 | 0.0492 |
| 7 | return_60 | 211 | 179.2891 | 0.0337 |
| 8 | volatility_20 | 106 | 167.5389 | 0.0315 |
| 9 | ma_gap_20 | 92 | 124.7649 | 0.0234 |
| 10 | ridge_pred_future_return | 39 | 108.9910 | 0.0205 |

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
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.4071 |
| Precision | 0.1818 |
| Recall | 0.0714 |
| F1 | 0.1026 |
| ROC-AUC | 0.3245 |
| Log Loss | 1.9506 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9900 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5825 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[26, 9], [26, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0111 |
| p25 | 0.2243 |
| median | 0.6106 |
| p75 | 0.9800 |
| max | 0.9976 |
| mean | 0.5579 |
| std | 0.3741 |
| count_ge_threshold | 11 |
| count_lt_threshold | 52 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_margin_to_threshold | 191 | 822.6520 | 0.1574 |
| 2 | volatility_20 | 216 | 575.2453 | 0.1101 |
| 3 | volatility_60 | 293 | 543.7232 | 0.1040 |
| 4 | close_to_ema_26 | 68 | 475.6652 | 0.0910 |
| 5 | return_10 | 169 | 372.1778 | 0.0712 |
| 6 | ma_gap_120 | 173 | 350.9012 | 0.0671 |
| 7 | return_60 | 156 | 322.5107 | 0.0617 |
| 8 | ridge_pred_future_log_return | 157 | 258.2663 | 0.0494 |
| 9 | ma_gap_60 | 158 | 239.6140 | 0.0459 |
| 10 | macd_hist_to_close | 151 | 180.8796 | 0.0346 |

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
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.6187 |
| Precision | 0.6222 |
| Recall | 0.8235 |
| F1 | 0.7089 |
| ROC-AUC | 0.7333 |
| Log Loss | 1.3487 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.0103 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6262 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 17], [6, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0017 |
| p25 | 0.0087 |
| median | 0.0422 |
| p75 | 0.1309 |
| max | 0.9987 |
| mean | 0.2081 |
| std | 0.3475 |
| count_ge_threshold | 45 |
| count_lt_threshold | 18 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_20 | 245 | 1059.4028 | 0.1968 |
| 2 | volatility_20 | 221 | 575.2445 | 0.1069 |
| 3 | return_60 | 211 | 397.6675 | 0.0739 |
| 4 | ridge_pred_future_log_return | 216 | 396.2949 | 0.0736 |
| 5 | volatility_60 | 141 | 302.3061 | 0.0562 |
| 6 | macd_hist_to_close | 149 | 297.0943 | 0.0552 |
| 7 | return_10 | 147 | 291.5235 | 0.0542 |
| 8 | macd_signal_to_close | 144 | 243.1120 | 0.0452 |
| 9 | ma_gap_120 | 183 | 238.1956 | 0.0443 |
| 10 | ma_gap_5 | 49 | 176.0077 | 0.0327 |

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
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.5444 |
| Precision | 0.7778 |
| Recall | 0.3111 |
| F1 | 0.4444 |
| ROC-AUC | 0.5630 |
| Log Loss | 2.1909 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.0702 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6389 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 4], [31, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0042 |
| p25 | 0.0247 |
| median | 0.0381 |
| p75 | 0.0830 |
| max | 0.6824 |
| mean | 0.0692 |
| std | 0.0935 |
| count_ge_threshold | 18 |
| count_lt_threshold | 45 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_hist_to_close | 145 | 932.9513 | 0.1702 |
| 2 | volatility_60 | 197 | 921.0217 | 0.1680 |
| 3 | ma_gap_120 | 252 | 515.5991 | 0.0941 |
| 4 | ma_gap_60 | 157 | 510.0897 | 0.0930 |
| 5 | ridge_pred_future_log_return | 217 | 402.8809 | 0.0735 |
| 6 | ridge_pred_margin_to_threshold | 211 | 381.7548 | 0.0696 |
| 7 | return_10 | 219 | 380.2719 | 0.0694 |
| 8 | volume_change_5 | 159 | 157.7348 | 0.0288 |
| 9 | return_60 | 87 | 151.1697 | 0.0276 |
| 10 | volatility_20 | 125 | 119.5994 | 0.0218 |

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
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.6626 |
| Precision | 0.5455 |
| Recall | 0.7200 |
| F1 | 0.6207 |
| ROC-AUC | 0.7684 |
| Log Loss | 0.8065 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.1497 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5976 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 15], [7, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0098 |
| p25 | 0.0424 |
| median | 0.3372 |
| p75 | 0.9618 |
| max | 0.9975 |
| mean | 0.4469 |
| std | 0.4214 |
| count_ge_threshold | 33 |
| count_lt_threshold | 30 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_60 | 259 | 1026.9478 | 0.1934 |
| 2 | volatility_60 | 235 | 884.1921 | 0.1665 |
| 3 | ma_gap_120 | 265 | 556.0755 | 0.1047 |
| 4 | ridge_pred_future_log_return | 186 | 395.2013 | 0.0744 |
| 5 | volatility_20 | 175 | 258.9158 | 0.0487 |
| 6 | macd_hist_to_close | 111 | 196.0840 | 0.0369 |
| 7 | volume_change_5 | 127 | 183.8254 | 0.0346 |
| 8 | macd_signal_to_close | 112 | 140.5931 | 0.0265 |
| 9 | return_60 | 111 | 132.1107 | 0.0249 |
| 10 | return_2 | 148 | 123.9788 | 0.0233 |

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
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.5208 |
| Precision | 0.6290 |
| Recall | 1.0000 |
| F1 | 0.7723 |
| ROC-AUC | 0.6325 |
| Log Loss | 1.2438 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.6434 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5902 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 23], [0, 39]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.4141 |
| p25 | 0.9431 |
| median | 0.9800 |
| p75 | 0.9891 |
| max | 0.9965 |
| mean | 0.9404 |
| std | 0.0992 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 403 | 1105.5831 | 0.2080 |
| 2 | ma_gap_60 | 321 | 1003.6871 | 0.1888 |
| 3 | volatility_60 | 204 | 546.8514 | 0.1029 |
| 4 | ma_gap_120 | 148 | 301.5477 | 0.0567 |
| 5 | macd_hist_to_close | 165 | 265.2405 | 0.0499 |
| 6 | macd_signal_to_close | 121 | 233.9844 | 0.0440 |
| 7 | return_2 | 202 | 223.2342 | 0.0420 |
| 8 | ema_12_to_ema_26 | 83 | 220.5376 | 0.0415 |
| 9 | return_60 | 113 | 148.3764 | 0.0279 |
| 10 | return_20 | 94 | 135.7001 | 0.0255 |

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
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.3915 |
| Precision | 0.2000 |
| Recall | 0.0588 |
| F1 | 0.0909 |
| ROC-AUC | 0.4412 |
| Log Loss | 1.2797 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9686 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6477 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 8], [32, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0093 |
| p25 | 0.6486 |
| median | 0.8865 |
| p75 | 0.9514 |
| max | 0.9972 |
| mean | 0.7741 |
| std | 0.2443 |
| count_ge_threshold | 10 |
| count_lt_threshold | 53 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 195 | 671.8495 | 0.1262 |
| 2 | volatility_20 | 217 | 648.5957 | 0.1219 |
| 3 | volatility_60 | 243 | 588.6403 | 0.1106 |
| 4 | ma_gap_60 | 88 | 489.7304 | 0.0920 |
| 5 | return_20 | 141 | 339.8512 | 0.0639 |
| 6 | return_60 | 183 | 318.3195 | 0.0598 |
| 7 | ema_12_to_ema_26 | 135 | 245.4780 | 0.0461 |
| 8 | ridge_pred_future_log_return | 192 | 226.5756 | 0.0426 |
| 9 | macd_hist_to_close | 140 | 200.1173 | 0.0376 |
| 10 | return_5 | 58 | 198.4317 | 0.0373 |

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
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6383 |
| Log Loss | 1.1656 |
| Baseline Accuracy | 0.7460 |
| Decision Threshold | 0.8713 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5837 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[16, 0], [47, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0080 |
| p25 | 0.1143 |
| median | 0.2610 |
| p75 | 0.4882 |
| max | 0.8485 |
| mean | 0.3193 |
| std | 0.2472 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 288 | 782.2323 | 0.1502 |
| 2 | volatility_60 | 290 | 678.3013 | 0.1302 |
| 3 | return_60 | 295 | 626.5421 | 0.1203 |
| 4 | ma_gap_120 | 201 | 420.2237 | 0.0807 |
| 5 | ridge_pred_future_log_return | 163 | 407.2712 | 0.0782 |
| 6 | high_low_range | 173 | 250.8196 | 0.0482 |
| 7 | volume_change_20 | 135 | 239.3534 | 0.0460 |
| 8 | volume_change_5 | 233 | 210.1575 | 0.0404 |
| 9 | macd_hist_to_close | 103 | 195.2111 | 0.0375 |
| 10 | ridge_pred_margin_to_threshold | 55 | 125.4636 | 0.0241 |

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
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.4172 |
| Log Loss | 1.6187 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.9565 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5913 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[30, 0], [33, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0021 |
| p25 | 0.0234 |
| median | 0.0475 |
| p75 | 0.1845 |
| max | 0.8415 |
| mean | 0.1539 |
| std | 0.2091 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 360 | 904.1959 | 0.1721 |
| 2 | volatility_60 | 301 | 633.0838 | 0.1205 |
| 3 | close_to_ema_26 | 146 | 471.1245 | 0.0897 |
| 4 | rsi_14 | 87 | 338.0768 | 0.0644 |
| 5 | high_low_range | 144 | 298.6288 | 0.0569 |
| 6 | volume_change_5 | 212 | 291.6937 | 0.0555 |
| 7 | ma_gap_120 | 119 | 208.8948 | 0.0398 |
| 8 | return_20 | 149 | 208.8704 | 0.0398 |
| 9 | ma_gap_60 | 134 | 196.6392 | 0.0374 |
| 10 | volume_z_20 | 128 | 181.2233 | 0.0345 |

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
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.4000 |
| Precision | 0.2000 |
| Recall | 0.3333 |
| F1 | 0.2500 |
| ROC-AUC | 0.4889 |
| Log Loss | 1.7573 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.9218 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5251 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 24], [12, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0179 |
| p25 | 0.7408 |
| median | 0.9009 |
| p75 | 0.9693 |
| max | 0.9957 |
| mean | 0.7817 |
| std | 0.2791 |
| count_ge_threshold | 30 |
| count_lt_threshold | 33 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 346 | 1100.1800 | 0.2105 |
| 2 | volatility_60 | 289 | 470.2206 | 0.0900 |
| 3 | ma_gap_20 | 148 | 449.4810 | 0.0860 |
| 4 | rsi_14 | 162 | 317.3080 | 0.0607 |
| 5 | volume_change_20 | 182 | 253.9701 | 0.0486 |
| 6 | volume_z_20 | 168 | 233.0769 | 0.0446 |
| 7 | macd_signal_to_close | 155 | 208.7570 | 0.0399 |
| 8 | return_20 | 166 | 194.0068 | 0.0371 |
| 9 | return_5 | 173 | 177.6362 | 0.0340 |
| 10 | high_low_range | 141 | 171.0897 | 0.0327 |

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
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.3492 |
| Recall | 1.0000 |
| F1 | 0.5176 |
| ROC-AUC | 0.5122 |
| Log Loss | 3.0175 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.2912 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6540 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 41], [0, 22]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.8121 |
| p25 | 0.9805 |
| median | 0.9935 |
| p75 | 0.9949 |
| max | 0.9987 |
| mean | 0.9805 |
| std | 0.0312 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 385 | 666.3273 | 0.1280 |
| 2 | volatility_20 | 194 | 603.8943 | 0.1160 |
| 3 | return_60 | 161 | 599.2446 | 0.1151 |
| 4 | macd_signal_to_close | 168 | 412.7615 | 0.0793 |
| 5 | ma_gap_20 | 55 | 313.0261 | 0.0601 |
| 6 | macd_hist_to_close | 144 | 289.8512 | 0.0557 |
| 7 | ma_gap_120 | 163 | 237.1844 | 0.0456 |
| 8 | rsi_14 | 106 | 207.8812 | 0.0399 |
| 9 | close_to_ema_26 | 92 | 186.7254 | 0.0359 |
| 10 | return_5 | 180 | 163.9158 | 0.0315 |

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
| Accuracy | 0.6825 |
| Balanced Accuracy | 0.7688 |
| Precision | 0.9667 |
| Recall | 0.6042 |
| F1 | 0.7436 |
| ROC-AUC | 0.8333 |
| Log Loss | 0.4406 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.9196 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6488 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 1], [19, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1084 |
| p25 | 0.5738 |
| median | 0.8887 |
| p75 | 0.9888 |
| max | 0.9981 |
| mean | 0.7524 |
| std | 0.2882 |
| count_ge_threshold | 30 |
| count_lt_threshold | 33 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 443 | 1209.5044 | 0.2305 |
| 2 | ma_gap_120 | 348 | 683.5077 | 0.1303 |
| 3 | ma_gap_20 | 167 | 499.6071 | 0.0952 |
| 4 | return_5 | 199 | 289.2133 | 0.0551 |
| 5 | volatility_20 | 184 | 277.0169 | 0.0528 |
| 6 | return_10 | 179 | 253.8525 | 0.0484 |
| 7 | ridge_pred_future_log_return | 187 | 247.5688 | 0.0472 |
| 8 | return_20 | 147 | 188.3406 | 0.0359 |
| 9 | volume_change_5 | 147 | 168.5204 | 0.0321 |
| 10 | volume_change_20 | 155 | 145.8762 | 0.0278 |

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
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.4889 |
| Precision | 0.2727 |
| Recall | 0.3333 |
| F1 | 0.3000 |
| ROC-AUC | 0.6037 |
| Log Loss | 1.2790 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.9101 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6785 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[29, 16], [12, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0018 |
| p25 | 0.2709 |
| median | 0.8311 |
| p75 | 0.9360 |
| max | 0.9874 |
| mean | 0.6384 |
| std | 0.3730 |
| count_ge_threshold | 22 |
| count_lt_threshold | 41 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 402 | 1021.1174 | 0.1940 |
| 2 | ma_gap_120 | 290 | 562.2204 | 0.1068 |
| 3 | volatility_20 | 279 | 390.7786 | 0.0742 |
| 4 | ridge_pred_future_log_return | 223 | 365.0619 | 0.0693 |
| 5 | macd_hist_to_close | 120 | 339.9106 | 0.0646 |
| 6 | macd_signal_to_close | 152 | 330.8856 | 0.0629 |
| 7 | return_10 | 72 | 276.3459 | 0.0525 |
| 8 | return_20 | 211 | 270.9589 | 0.0515 |
| 9 | return_60 | 151 | 223.8064 | 0.0425 |
| 10 | volume_change_20 | 229 | 197.6065 | 0.0375 |

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
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5288 |
| Precision | 0.5600 |
| Recall | 0.4242 |
| F1 | 0.4828 |
| ROC-AUC | 0.5879 |
| Log Loss | 1.0838 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.8270 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5871 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 11], [19, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0018 |
| p25 | 0.2603 |
| median | 0.7239 |
| p75 | 0.9615 |
| max | 0.9935 |
| mean | 0.5968 |
| std | 0.3656 |
| count_ge_threshold | 25 |
| count_lt_threshold | 38 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_hist_to_close | 276 | 845.9788 | 0.1593 |
| 2 | ridge_pred_future_log_return | 270 | 816.3858 | 0.1538 |
| 3 | volatility_60 | 219 | 441.8920 | 0.0832 |
| 4 | macd_signal_to_close | 111 | 344.1675 | 0.0648 |
| 5 | volatility_20 | 327 | 321.6417 | 0.0606 |
| 6 | return_20 | 243 | 233.1099 | 0.0439 |
| 7 | ridge_pred_future_return | 48 | 209.6980 | 0.0395 |
| 8 | ema_12_to_ema_26 | 106 | 207.8994 | 0.0392 |
| 9 | ma_gap_60 | 78 | 203.2911 | 0.0383 |
| 10 | ma_gap_120 | 195 | 195.6146 | 0.0368 |

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
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5240 |
| Precision | 0.4074 |
| Recall | 0.4583 |
| F1 | 0.4314 |
| ROC-AUC | 0.5726 |
| Log Loss | 1.5136 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.3855 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6400 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 16], [13, 11]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0031 |
| p25 | 0.0128 |
| median | 0.1524 |
| p75 | 0.9664 |
| max | 0.9933 |
| mean | 0.3983 |
| std | 0.4289 |
| count_ge_threshold | 27 |
| count_lt_threshold | 36 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 274 | 721.8898 | 0.1361 |
| 2 | macd_hist_to_close | 224 | 614.2734 | 0.1158 |
| 3 | ema_12_to_ema_26 | 177 | 572.8274 | 0.1080 |
| 4 | ma_gap_60 | 99 | 528.0558 | 0.0995 |
| 5 | macd_signal_to_close | 173 | 456.5650 | 0.0861 |
| 6 | volatility_20 | 200 | 280.4625 | 0.0529 |
| 7 | volume_change_20 | 195 | 256.9608 | 0.0484 |
| 8 | return_60 | 130 | 202.1582 | 0.0381 |
| 9 | ridge_pred_future_log_return | 117 | 174.3520 | 0.0329 |
| 10 | return_20 | 171 | 173.6164 | 0.0327 |

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
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6242 |
| Precision | 0.6786 |
| Recall | 0.5588 |
| F1 | 0.6129 |
| ROC-AUC | 0.6643 |
| Log Loss | 1.4776 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.0570 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6071 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 9], [15, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0019 |
| p25 | 0.0129 |
| median | 0.0374 |
| p75 | 0.2257 |
| max | 0.9972 |
| mean | 0.2306 |
| std | 0.3597 |
| count_ge_threshold | 28 |
| count_lt_threshold | 35 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 365 | 1038.6613 | 0.1931 |
| 2 | rsi_14 | 227 | 732.0304 | 0.1361 |
| 3 | macd_to_close | 184 | 610.1305 | 0.1134 |
| 4 | volume_change_20 | 257 | 377.7962 | 0.0702 |
| 5 | macd_signal_to_close | 230 | 356.2761 | 0.0662 |
| 6 | ema_12_to_ema_26 | 85 | 250.3437 | 0.0465 |
| 7 | macd_hist_to_close | 174 | 228.8134 | 0.0425 |
| 8 | ma_gap_60 | 100 | 168.1019 | 0.0313 |
| 9 | ridge_pred_future_log_return | 132 | 163.3057 | 0.0304 |
| 10 | return_60 | 140 | 146.3595 | 0.0272 |

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
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5324 |
| Precision | 0.4762 |
| Recall | 0.3704 |
| F1 | 0.4167 |
| ROC-AUC | 0.5072 |
| Log Loss | 1.1500 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.2765 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6215 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 11], [17, 10]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0034 |
| p25 | 0.0215 |
| median | 0.0999 |
| p75 | 0.3451 |
| max | 0.9712 |
| mean | 0.2422 |
| std | 0.2780 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 289 | 754.2949 | 0.1463 |
| 2 | macd_hist_to_close | 156 | 421.1870 | 0.0817 |
| 3 | volatility_20 | 197 | 367.1486 | 0.0712 |
| 4 | ema_12_to_ema_26 | 129 | 355.4741 | 0.0690 |
| 5 | volume_change_20 | 196 | 340.8949 | 0.0661 |
| 6 | macd_signal_to_close | 201 | 320.7368 | 0.0622 |
| 7 | close_to_low | 203 | 268.3658 | 0.0521 |
| 8 | ridge_pred_future_log_return | 162 | 256.4818 | 0.0498 |
| 9 | ma_gap_120 | 112 | 218.0766 | 0.0423 |
| 10 | return_60 | 139 | 190.9769 | 0.0370 |

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
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6076 |
| Precision | 0.6875 |
| Recall | 0.3667 |
| F1 | 0.4783 |
| ROC-AUC | 0.6141 |
| Log Loss | 1.1493 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.2285 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6825 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 5], [19, 11]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0035 |
| p25 | 0.0298 |
| median | 0.0695 |
| p75 | 0.2225 |
| max | 0.9677 |
| mean | 0.1803 |
| std | 0.2511 |
| count_ge_threshold | 16 |
| count_lt_threshold | 47 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 293 | 769.3930 | 0.1485 |
| 2 | volatility_60 | 311 | 675.2867 | 0.1303 |
| 3 | volatility_20 | 206 | 486.8243 | 0.0940 |
| 4 | ma_gap_120 | 200 | 341.5642 | 0.0659 |
| 5 | volume_change_20 | 206 | 272.7634 | 0.0527 |
| 6 | return_5 | 147 | 248.4069 | 0.0479 |
| 7 | macd_hist_to_close | 169 | 242.9018 | 0.0469 |
| 8 | macd_to_close | 100 | 232.0894 | 0.0448 |
| 9 | ema_12_to_ema_26 | 106 | 220.5117 | 0.0426 |
| 10 | volume_change_5 | 209 | 218.4808 | 0.0422 |

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
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.4818 |
| Precision | 0.4444 |
| Recall | 0.2667 |
| F1 | 0.3333 |
| ROC-AUC | 0.4384 |
| Log Loss | 1.7863 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.0572 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5557 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 10], [22, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0035 |
| p25 | 0.0138 |
| median | 0.0267 |
| p75 | 0.0626 |
| max | 0.2051 |
| mean | 0.0417 |
| std | 0.0398 |
| count_ge_threshold | 18 |
| count_lt_threshold | 45 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 306 | 983.3126 | 0.1854 |
| 2 | volatility_60 | 281 | 466.3217 | 0.0879 |
| 3 | volatility_20 | 211 | 452.4633 | 0.0853 |
| 4 | return_60 | 234 | 401.5998 | 0.0757 |
| 5 | ma_gap_60 | 110 | 355.0105 | 0.0669 |
| 6 | macd_hist_to_close | 216 | 276.9589 | 0.0522 |
| 7 | ridge_pred_future_log_return | 174 | 244.7375 | 0.0461 |
| 8 | high_low_range | 180 | 237.6573 | 0.0448 |
| 9 | ema_12_to_ema_26 | 107 | 206.4665 | 0.0389 |
| 10 | volume_change_5 | 155 | 193.1044 | 0.0364 |

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
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.4163 |
| Precision | 0.6531 |
| Recall | 0.7273 |
| F1 | 0.6882 |
| ROC-AUC | 0.4593 |
| Log Loss | 1.4595 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.0771 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6073 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[2, 17], [12, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0030 |
| p25 | 0.0943 |
| median | 0.1824 |
| p75 | 0.4080 |
| max | 0.8649 |
| mean | 0.2686 |
| std | 0.2345 |
| count_ge_threshold | 49 |
| count_lt_threshold | 14 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 335 | 1246.1002 | 0.2375 |
| 2 | volatility_20 | 317 | 661.9874 | 0.1262 |
| 3 | macd_hist_to_close | 206 | 467.3942 | 0.0891 |
| 4 | volatility_60 | 248 | 428.7248 | 0.0817 |
| 5 | volume_change_20 | 259 | 358.2938 | 0.0683 |
| 6 | ema_12_to_ema_26 | 124 | 255.9724 | 0.0488 |
| 7 | intraday_return | 166 | 214.4211 | 0.0409 |
| 8 | log_return_1 | 127 | 174.4991 | 0.0333 |
| 9 | ma_gap_60 | 118 | 164.0628 | 0.0313 |
| 10 | return_10 | 55 | 155.3533 | 0.0296 |

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
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.6528 |
| Precision | 0.8333 |
| Recall | 0.4167 |
| F1 | 0.5556 |
| ROC-AUC | 0.7058 |
| Log Loss | 1.4224 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.1618 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5310 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 3], [21, 15]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0011 |
| p25 | 0.0207 |
| median | 0.0430 |
| p75 | 0.2301 |
| max | 0.9613 |
| mean | 0.1690 |
| std | 0.2443 |
| count_ge_threshold | 18 |
| count_lt_threshold | 45 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 344 | 1181.1202 | 0.2244 |
| 2 | macd_hist_to_close | 196 | 557.5476 | 0.1059 |
| 3 | volatility_20 | 232 | 400.0284 | 0.0760 |
| 4 | volatility_60 | 166 | 370.2025 | 0.0703 |
| 5 | ridge_pred_future_log_return | 130 | 284.7374 | 0.0541 |
| 6 | ma_gap_120 | 187 | 274.7624 | 0.0522 |
| 7 | ema_12_to_ema_26 | 159 | 249.4184 | 0.0474 |
| 8 | return_60 | 209 | 238.7244 | 0.0453 |
| 9 | volume_change_20 | 224 | 189.2667 | 0.0360 |
| 10 | ma_gap_5 | 127 | 178.2299 | 0.0339 |

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
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.4179 |
| Precision | 0.4211 |
| Recall | 0.2286 |
| F1 | 0.2963 |
| ROC-AUC | 0.4337 |
| Log Loss | 2.2722 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.0373 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5893 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 11], [27, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0007 |
| p25 | 0.0059 |
| median | 0.0198 |
| p75 | 0.0556 |
| max | 0.8700 |
| mean | 0.0830 |
| std | 0.1656 |
| count_ge_threshold | 19 |
| count_lt_threshold | 44 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 288 | 1221.5304 | 0.2264 |
| 2 | ma_gap_120 | 166 | 704.5955 | 0.1306 |
| 3 | ridge_pred_future_log_return | 200 | 388.2829 | 0.0720 |
| 4 | macd_hist_to_close | 252 | 320.4642 | 0.0594 |
| 5 | macd_to_close | 70 | 244.1491 | 0.0453 |
| 6 | ma_gap_60 | 128 | 232.6527 | 0.0431 |
| 7 | volume_change_20 | 215 | 220.4800 | 0.0409 |
| 8 | high_low_range | 181 | 208.6751 | 0.0387 |
| 9 | ema_12_to_ema_26 | 92 | 192.9076 | 0.0358 |
| 10 | volatility_20 | 155 | 156.4781 | 0.0290 |

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
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.5079 |
| Recall | 1.0000 |
| F1 | 0.6737 |
| ROC-AUC | 0.5433 |
| Log Loss | 2.3558 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.1479 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5549 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 31], [0, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.7715 |
| p25 | 0.9731 |
| median | 0.9930 |
| p75 | 0.9971 |
| max | 0.9995 |
| mean | 0.9762 |
| std | 0.0450 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 235 | 1161.5403 | 0.2158 |
| 2 | volatility_60 | 353 | 862.9907 | 0.1603 |
| 3 | macd_signal_to_close | 248 | 623.4733 | 0.1158 |
| 4 | volatility_20 | 173 | 435.1725 | 0.0808 |
| 5 | return_20 | 189 | 301.5006 | 0.0560 |
| 6 | macd_hist_to_close | 178 | 240.8470 | 0.0447 |
| 7 | return_60 | 197 | 217.2410 | 0.0404 |
| 8 | volume_change_20 | 216 | 209.8526 | 0.0390 |
| 9 | ma_gap_60 | 117 | 208.8476 | 0.0388 |
| 10 | high_low_range | 190 | 161.0427 | 0.0299 |

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
| Balanced Accuracy | 0.5249 |
| Precision | 0.6667 |
| Recall | 0.0769 |
| F1 | 0.1379 |
| ROC-AUC | 0.7173 |
| Log Loss | 0.9343 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.9763 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5345 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[36, 1], [24, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0128 |
| p25 | 0.0885 |
| median | 0.8745 |
| p75 | 0.9284 |
| max | 0.9822 |
| mean | 0.5740 |
| std | 0.4045 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12_to_ema_26 | 103 | 725.9758 | 0.1385 |
| 2 | volatility_60 | 410 | 711.6565 | 0.1357 |
| 3 | volatility_20 | 216 | 483.1211 | 0.0922 |
| 4 | macd_to_close | 169 | 426.1765 | 0.0813 |
| 5 | ma_gap_120 | 112 | 221.5204 | 0.0423 |
| 6 | volume_change_20 | 141 | 209.7086 | 0.0400 |
| 7 | return_10 | 87 | 202.1429 | 0.0386 |
| 8 | ma_gap_60 | 67 | 201.8968 | 0.0385 |
| 9 | volume_z_20 | 104 | 192.5186 | 0.0367 |
| 10 | return_60 | 123 | 184.4557 | 0.0352 |

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
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5607 |
| Precision | 0.4884 |
| Recall | 0.7500 |
| F1 | 0.5915 |
| ROC-AUC | 0.5867 |
| Log Loss | 2.3812 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.9818 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5517 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 22], [7, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.6012 |
| p25 | 0.9725 |
| median | 0.9922 |
| p75 | 0.9961 |
| max | 0.9989 |
| mean | 0.9625 |
| std | 0.0796 |
| count_ge_threshold | 43 |
| count_lt_threshold | 20 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 368 | 752.6586 | 0.1412 |
| 2 | volatility_20 | 337 | 576.7095 | 0.1082 |
| 3 | return_20 | 298 | 572.9705 | 0.1075 |
| 4 | ema_12_to_ema_26 | 188 | 532.3179 | 0.0999 |
| 5 | macd_hist_to_close | 121 | 464.3382 | 0.0871 |
| 6 | return_60 | 280 | 329.9360 | 0.0619 |
| 7 | macd_signal_to_close | 123 | 272.5379 | 0.0511 |
| 8 | macd_to_close | 106 | 224.1969 | 0.0421 |
| 9 | ma_gap_60 | 134 | 213.0871 | 0.0400 |
| 10 | ridge_pred_future_log_return | 99 | 210.2957 | 0.0395 |

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
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.4582 |
| Precision | 0.6000 |
| Recall | 0.5250 |
| F1 | 0.5600 |
| ROC-AUC | 0.4163 |
| Log Loss | 1.2010 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.2051 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5213 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 14], [19, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0473 |
| p25 | 0.1388 |
| median | 0.2266 |
| p75 | 0.3911 |
| max | 0.8489 |
| mean | 0.2782 |
| std | 0.1964 |
| count_ge_threshold | 35 |
| count_lt_threshold | 28 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_20 | 376 | 1063.2615 | 0.2057 |
| 2 | volatility_60 | 255 | 575.8994 | 0.1114 |
| 3 | ridge_pred_future_log_return | 238 | 433.1508 | 0.0838 |
| 4 | return_10 | 116 | 415.1004 | 0.0803 |
| 5 | ma_gap_120 | 146 | 342.3208 | 0.0662 |
| 6 | volume_z_20 | 144 | 248.1997 | 0.0480 |
| 7 | close_to_ema_26 | 105 | 235.8507 | 0.0456 |
| 8 | ma_gap_60 | 83 | 214.9361 | 0.0416 |
| 9 | return_60 | 181 | 202.0162 | 0.0391 |
| 10 | rsi_14 | 73 | 171.0479 | 0.0331 |

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
| Balanced Accuracy | 0.4138 |
| Precision | 0.7429 |
| Recall | 0.5200 |
| F1 | 0.6118 |
| ROC-AUC | 0.3569 |
| Log Loss | 1.0674 |
| Baseline Accuracy | 0.7937 |
| Decision Threshold | 0.5385 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6078 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 9], [24, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0276 |
| p25 | 0.2560 |
| median | 0.6650 |
| p75 | 0.8763 |
| max | 0.9824 |
| mean | 0.5802 |
| std | 0.3225 |
| count_ge_threshold | 35 |
| count_lt_threshold | 28 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 301 | 683.4682 | 0.1325 |
| 2 | ma_gap_120 | 244 | 498.4181 | 0.0966 |
| 3 | macd_signal_to_close | 263 | 489.4579 | 0.0949 |
| 4 | volatility_20 | 233 | 393.8009 | 0.0763 |
| 5 | return_60 | 233 | 345.0036 | 0.0669 |
| 6 | high_low_range | 128 | 215.7255 | 0.0418 |
| 7 | macd_hist_to_close | 120 | 191.8032 | 0.0372 |
| 8 | close_to_ema_26 | 55 | 189.8698 | 0.0368 |
| 9 | volume_change_20 | 113 | 166.7996 | 0.0323 |
| 10 | ema_12_to_ema_26 | 116 | 160.5890 | 0.0311 |

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
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.5300 |
| Precision | 1.0000 |
| Recall | 0.0600 |
| F1 | 0.1132 |
| ROC-AUC | 0.4369 |
| Log Loss | 2.3750 |
| Baseline Accuracy | 0.7937 |
| Decision Threshold | 0.9726 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5284 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 0], [47, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0030 |
| p25 | 0.0151 |
| median | 0.0281 |
| p75 | 0.6194 |
| max | 0.9843 |
| mean | 0.2785 |
| std | 0.3749 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 311 | 933.2931 | 0.1736 |
| 2 | volatility_60 | 106 | 505.8786 | 0.0941 |
| 3 | volatility_20 | 273 | 452.9302 | 0.0843 |
| 4 | ma_gap_120 | 210 | 374.8291 | 0.0697 |
| 5 | macd_hist_to_close | 150 | 330.1875 | 0.0614 |
| 6 | macd_signal_to_close | 205 | 270.8063 | 0.0504 |
| 7 | volume_change_20 | 147 | 265.9585 | 0.0495 |
| 8 | close_to_ema_26 | 81 | 223.5127 | 0.0416 |
| 9 | ma_gap_20 | 115 | 209.6119 | 0.0390 |
| 10 | return_2 | 103 | 208.0650 | 0.0387 |

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
| Accuracy | 0.5714 |
| Balanced Accuracy | 0.5803 |
| Precision | 0.6500 |
| Recall | 0.3939 |
| F1 | 0.4906 |
| ROC-AUC | 0.6162 |
| Log Loss | 1.4864 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.0639 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5556 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 7], [20, 13]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0039 |
| p25 | 0.0293 |
| median | 0.0457 |
| p75 | 0.0755 |
| max | 0.4100 |
| mean | 0.0806 |
| std | 0.0968 |
| count_ge_threshold | 20 |
| count_lt_threshold | 43 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 285 | 965.4376 | 0.1796 |
| 2 | return_60 | 303 | 909.2649 | 0.1692 |
| 3 | ma_gap_120 | 238 | 706.1196 | 0.1314 |
| 4 | volatility_20 | 250 | 405.8496 | 0.0755 |
| 5 | ridge_pred_future_log_return | 207 | 374.9416 | 0.0698 |
| 6 | macd_signal_to_close | 121 | 250.9321 | 0.0467 |
| 7 | return_2 | 130 | 187.9638 | 0.0350 |
| 8 | volume_change_20 | 156 | 174.9594 | 0.0326 |
| 9 | ema_12_to_ema_26 | 112 | 141.5082 | 0.0263 |
| 10 | close_to_high | 186 | 115.9908 | 0.0216 |
