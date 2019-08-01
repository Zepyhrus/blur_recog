#%% 
import os
from glob import glob
import cv2
import numpy as np
import numpy.random as npr
from os.path import join, split
from shutil import copyfile
import matplotlib.pyplot as plt



#%%
plt.rcParamsDefault["figure.autolayout"] = True


images = npr.choice(glob("train/2/*"), 16)

plt.figure(figsize=(16, 16))
for i, image in enumerate(images):
  img = cv2.imread(image)
  plt.subplot(4, 4, i+1)
  plt.imshow(img[:, :, ::-1])




#%%
