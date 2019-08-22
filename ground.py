#%%
import pandas as pd
import numpy as np
import numpy.random as npr

import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

import urllib
import os
from os.path import join, split

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

res = pd.read_csv('std_new.csv', index_col=False)
res_cloud = pd.read_csv('std_new.cloud.csv', index_col=False)
#%%
names = list(res.columns)[1:]
for name in names:
  a1 = res[['pair', name]]
  a2 = res_cloud[['pair', name]]

  a_compare = a1.merge(a2, on='pair')

  a_compare['diffs'] = a_compare[name + '_x'] - \
      a_compare[name + '_y']

  plt.figure(figsize=(12, 6))
  plt.hist(a_compare['diffs'])
  plt.legend([name])
  plt.title(name)
  plt.savefig(name+'_new'+'.png')
  plt.close()

#%%