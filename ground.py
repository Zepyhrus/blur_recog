#%% 
import os
from glob import glob
import cv2
import numpy as np
import numpy.random as npr
from os.path import join, split
from shutil import copyfile
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models
import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable


model = torch.load('blur_reg_resnet18_128_24.pt').eval()


example = torch.rand(1, 3, 112, 96).to(torch.device("cuda"))
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

torch.jit.save(traced_script_module, 'blur.pt')
loader = transforms.Compose([transforms.ToTensor()])


def image_loader(image_name):
	"""
		load image, returns cuda tensor
	"""
	image = Image.open(image_name)
	image = loader(image).float()
	image = Variable(image, requires_grad=True)
	image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
	return image.cuda()  #assumes that you're using GPU

for i in range(6):
  img = image_loader(f'/home/ubuntu/Workspace/FaceRecognition/image/{i}.png')

  # print(img)
  # print(img.size())
  print(model.forward(img).data.item())
