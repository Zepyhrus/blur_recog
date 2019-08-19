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

#%%
res = pd.read_csv('std.csv')
res_cloud = pd.read_csv('std.cloud.csv')


#%%
a = res['pair', ]


