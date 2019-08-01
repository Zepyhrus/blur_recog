# Blur Recognize

Binary classification: a regression between 0 and 1, will map it to 0 and 100 in the future.
Batch size: 128;
Baseline model: ResNet18, with 128 full connection;
Optimizer: Adam optimizer;
Learning rate: base: 0.003, decay rate: 0.7 every 3 epchos;

## Benchmark

ResNet18, 0/1/2 regression problem

ResNet18, binary classification problem
`blur_cam_test` dataset:
Total accuracy=0.4413, Accuracy=0.4413, Recall = 1.0000, at threshold: 0.0
Total accuracy=0.7845, Accuracy=0.6903, Recall = 0.9306, at threshold: 0.0
Total accuracy=0.8026, Accuracy=0.7258, Recall = 0.8911, at threshold: 0.0
Total accuracy=0.8081, Accuracy=0.7516, Recall = 0.8481, at threshold: 0.1
Total accuracy=0.8070, Accuracy=0.7659, Recall = 0.8150, at threshold: 0.1
Total accuracy=0.7977, Accuracy=0.7722, Recall = 0.7731, at threshold: 0.1
Total accuracy=0.7961, Accuracy=0.7835, Recall = 0.7488, at threshold: 0.1
Total accuracy=0.7840, Accuracy=0.7820, Recall = 0.7146, at threshold: 0.1
Total accuracy=0.7758, Accuracy=0.7877, Recall = 0.6802, at threshold: 0.2
Total accuracy=0.7675, Accuracy=0.7899, Recall = 0.6519, at threshold: 0.2


