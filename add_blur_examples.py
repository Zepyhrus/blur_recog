from glob import glob
import os
from os.path import join, split
import matplotlib.pyplot as plt


import numpy as np
import numpy.random as npr
import cv2

from motion_blur import random_blur


images = glob(join("origin_aug_gray", "*"))
dest_folder = 'test_gray'

label_file = open("10_level_gray_label.txt", "w")


for i, image in enumerate(images):
  img = cv2.imread(image)

  for level in range(10):
    if img is None:
      print("wrong image path")
    else:
      img_blurred = random_blur(img, level)
      
      if not os.path.isdir(join(dest_folder, str(level))):
        os.makedirs(join(dest_folder, str(level)))
      
      save_path = join(dest_folder, str(level), split(image)[-1])
      cv2.imwrite(save_path, img_blurred)
      label_file.write(join(str(level), split(image)[-1]) + \
        " " + str(level) + "\n")
  
  if i % 1000 == 0:
    print(f"{i} images generated!")

label_file.close()
