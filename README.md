# Blur Recognize

Binary classification: a regression between 0 and 1, will map it to 0 and 100 in the future.
**Train dataset**: 
  1. 0 (clear): 24000, 1 (medium blur): 24000, 2 (blur): 24000;
  2. level: 0~9, blurring kernel size: 2 * level + 1;
**Test dataset**: 
  1. blur_cam_test: 0: 994, 1: 830;
  2. blur_cam_test2: 0: 37984, 1: 911;
**Batch size**: 128;
**Baseline model**: ResNet18, with 128 full connection;
**Optimizer**: Adam optimizer;
**Learning rate**: base: 0.003, decay rate: 0.7 every 3 epchos;
**Labels**: 0 for clear, 1 for mediate blur, 2 for blur;

## Benchmark
1. ResNet18, 0/1/2 regression problem, the model actually performs better
`blur_cam_test` dataset:
Total accuracy=0.8793859649122807
Confussion matrix:
label:        0       1
predict:
      0     851      50
      1     143     750

2. ResNet18, 0~9 levels, regression problem, the model performs the second best over `blur_cam_test` dataset:
Total accuracy = 0.8591008771929824
using threshold: 0~2: clear | 2~6: mediam blur | 6~10: blur
Confussion matrix:
label:        0       1
predict:
      0     886     117
      1     108     688

3. ResNet18, 0~9 levels, regression problem, using brightness augumentation, threshold 2.2, 19 epochs:
label:          0       1
pred:   0       910     179
        1       87      648
accuracy:       0.8542

4. ResNet18, 0~9 levels, regression problem, threshold 2.2, 79 epochs:
label:          0       1
pred:   0       899     165
        1       98      662
accuracy:       0.8558

5. ResNet18, 0~9 levels, regression problem, threshold 2.2, 99 epochs:
label:          0       1
pred:   0       908     175
        1       89      652
accuracy:       0.8553

4. ResNet18, 0~9 levels, using grayscale images, thershold 3.5:
label:	0	    1
pred:	0	883	  140
	    1	114	  687
accuracy:	0.8607

## Validate
