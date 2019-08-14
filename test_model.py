#%%
import cv2
import os
from os.path import join, split
from glob import glob
from shutil import copyfile

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloader import BlurImageDataset




#%%
IMAGE_PARENT = 'blur_cam_test'
IMAGE_W_LABEL_TXT = '../data_generator/class_id_to_files_70001_test.txt'
MODEL_NAME = 'blur_reg_resnet18_128_24'


load_test_from_file = False

train_transforms = transforms.Compose([
	transforms.ToTensor(),
])
if load_test_from_file:
    print(f'Load from file:{IMAGE_W_LABEL_TXT}')
    test_dataset = BlurImageDataset(IMAGE_PARENT, IMAGE_W_LABEL_TXT, transform=train_transforms)
else:
	print(f'Load from folder directly:{IMAGE_PARENT}')
	test_dataset = torchvision.datasets.ImageFolder(root=IMAGE_PARENT, transform=train_transforms)

map_pred_index_to_label = ['0', '1', '2']#%% load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_load = torch.load(f'./{MODEL_NAME}.pt')
model = model_load.eval()
model.to(device)

#%%
# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
loader = transforms.Compose([transforms.ToTensor()])
img_limit = 3000

def image_loader(image_name):
	"""
		load image, returns cuda tensor
	"""
	image = Image.open(image_name)
	image = loader(image).float()
	image = Variable(image, requires_grad=True)
	image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
	return image.cuda()  #assumes that you're using GPU


result = {"0": [], "1": [], "2": []}


for label in map_pred_index_to_label:
	print(f"{label}: ************************")
	images = glob(join(IMAGE_PARENT, label, "*"))

	if len(images) > img_limit:
		images = npr.choice(images, img_limit)

	for i, image in enumerate(images):
		img = image_loader(image)

		pred = model.forward(img).data.item()

		if label == "0" and pred >= 2:
			copyfile(image, image.replace("/0/", "/01/"))
		
		if pred <= 2 and label == "1":
			copyfile(image, image.replace("/1/", "/10/"))

		result[label].append(pred)
		# if i > 10:
		# 	break
		
#%%
plt.figure(figsize = (12, 6))
plt.hist(result["0"], np.arange(100)/10, log=True, alpha=0.7)
plt.hist(result["1"], np.arange(100)/10, log=True, alpha=0.7)
plt.hist(result["2"], np.arange(100)/10, log=True, alpha=0.7)
plt.legend(["0", "1", "2"])
plt.savefig('error.jpg')
plt.close()

#%%
accuracy = (len([x for x in result["0"] if x < 0.5]) +\
	len([x for x in result["1"] if x >= 0.5 and x < 1.5]) +\
	len([x for x in result["2"] if x >= 1.5])) /\
	(len(result["0"]) + len(result["1"]) + len(result["2"]))

print("accuracy: ", accuracy)