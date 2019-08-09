from glob import glob
import os
from os.path import join, split
import matplotlib.pyplot as plt


import numpy as np
import numpy.random as npr
import cv2

from motion_blur import random_blur

test_file = "blur_cam_test/0/1501855183567___c_77_y_-2__p_-21__r_-9_aligned.png"
image = cv2.imread(test_file)

kernel_size = 15
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 100)  # 高斯滤波
b2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

blurred = cv2.blur(image, (kernel_size, kernel_size))  # 领域均值滤波
blurred = cv2.medianBlur(image, kernel_size)  # 中值滤波



images = glob(join("origin_images", "*"))

label_file = open("10_level_label.txt", "w")


for i, image in enumerate(images):
  img = cv2.imread(image)

  for level in range(10):
    if img is None:
      print("wrong image path")
    else:
      img_blurred = random_blur(img, level)
      
      if not os.path.isdir(join("test", str(level))):
        os.makedirs(join("test", str(level)))
      
      save_path = join("test", str(level), split(image)[-1])
      cv2.imwrite(save_path, img_blurred)
      label_file.write(join(str(level), split(image)[-1]) + \
        " " + str(level) + "\n")
  
  if i % 1000 == 0:
    print(f"{i} images generated!")

label_file.close()
