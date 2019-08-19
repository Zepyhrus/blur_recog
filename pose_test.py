#%%
import pandas as pd
import numpy as np
import numpy.random as npr

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import urllib
import os
from os.path import join, split

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

model = torch.jit.load('face_pose.pt')
gpu = 0
#%%
img = cv2.imread('../FaceRecog/image/33/right.jpg')[28:192, 53:181]

plt.subplot(1, 2, 1)
plt.imshow(img[:, :, ::-1])

# gamma = (np.mean(img) / 128)

# img = ((img / 255) ** gamma * 255).astype(np.uint8)

plt.subplot(1, 2, 2)
plt.imshow(img[:, :, ::-1])

#%%

transformations = transforms.Compose(
  [transforms.Scale(128),
  transforms.CenterCrop(128),
  transforms.ToTensor()])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, 'gray')

img = Image.fromarray(img)
img = transformations(img)
img_shape = img.size()
img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
img = Variable(img).cuda(gpu)



#%%
idx_tensor = torch.FloatTensor(range(66)).cuda(gpu)

yaw, pitch, roll = model(img)


yaw_predicted = F.softmax(yaw)
pitch_predicted = F.softmax(pitch)
roll_predicted = F.softmax(roll)


# Get continuous predictions in degrees.
yaw_predicted = torch.sum(
  yaw_predicted.data[0] * idx_tensor) * 3 - 99
pitch_predicted = torch.sum(
  pitch_predicted.data[0] * idx_tensor) * 3 - 99
roll_predicted = torch.sum(
  roll_predicted.data[0] * idx_tensor) * 3 - 99

yaw_predicted = yaw_predicted.cpu().data.numpy()
pitch_predicted = pitch_predicted.cpu().data.numpy()
roll_predicted = roll_predicted.cpu().data.numpy()

print('yaw:\t', yaw_predicted)
print('pitch:\t', pitch_predicted)
print('roll:\t', roll_predicted)

#%%
