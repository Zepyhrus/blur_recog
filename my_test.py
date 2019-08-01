from glob import glob
import cv2
import numpy as np
import numpy.random as npr



images = glob("blur_cam_test/1/*.jpg")

# sub_images = 

for i in range(100):
  img = cv2.imread(images[i])
  print(img.shape)



