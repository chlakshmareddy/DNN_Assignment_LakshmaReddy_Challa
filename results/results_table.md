| Experiment | Modification | Train Loss | Validation Loss | Train Accuracy | Validation Accuracy |
|---|---|---:|---:|---:|---:|
| Baseline CNN | 3-block CNN | 1.5146 | 1.8246 | 0.357 | 0.050 |
| Dropout | Adds dropout 0.50 | 1.5953 | 1.6037 | 0.243 | 0.200 |
| Batch Normalization | Adds batch normalization | 0.8109 | 1.9271 | 0.871 | 0.200 |
| More Filters | Uses 32-64-128 filters | 1.5270 | 1.7668 | 0.229 | 0.150 |
| Kernel Size 5 | Changes kernel size from 3 to 5 | 1.4992 | 1.7526 | 0.357 | 0.150 |
| Data Augmentation | Adds flip and rotation | 1.5556 | 1.6022 | 0.257 | 0.250 |
