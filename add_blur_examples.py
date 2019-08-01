import numpy as np
import cv2

test_file = "/home/r/Projects/noise_evaluation/datasets/0.jpg"
image = cv2.imread( test_file )

#高斯滤波
kernel_size = 15
blurred = cv2.GaussianBlur(image,(kernel_size, kernel_size),100)
cv2.imshow("Gaussian",blurred)
cv2.waitKey(0)

#领域均值滤波
blurred = cv2.blur(image,(kernel_size,kernel_size))
cv2.imshow("Averaged",blurred)
cv2.waitKey(0)

#中值滤波
blurred = cv2.medianBlur(image,kernel_size)
cv2.imshow("Median",blurred)
cv2.waitKey(0)