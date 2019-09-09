# coding: utf-8
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# from dataloader import BlurImageDataset
from torch.utils.data.sampler import SubsetRandomSampler


# ===== Fine tune all weights: large batch size dynamic lr
TRAIN_BATCH_SIZE = 1024
IMAGE_PARENT = '/home/ubuntu/Workspace/blur_recog/test_gray'
IMAGE_W_LABEL_TXT = '/home/ubuntu/Workspace/blur_recog/10_level_gray_label.txt'
MODEL_PREFIX = f'blur_reg_10_gray_resnet18_{TRAIN_BATCH_SIZE}'
MODE_FEATURE_EXTRACT = False
USE_PRETRAINED = False  # gray scale training is built from scratch
TRAIN_SET_RATIO = 0.99

# TODO: to external class
class BlurImageDataset(Dataset):
  def __init__(self, file_base_dir, txt_path, transform = None, target_transform = None):
    if not file_base_dir:
      raise Exception(f'file_base_dir: {file_base_dir} , is invalid')

    fh = open(txt_path, 'r')
    imgs = []
    for line in fh:
      line = line.rstrip()
      file_name, label = line.split()
      imgs.append((file_base_dir + '/' + file_name, int(label)))
    self.imgs = imgs
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    fn, label = self.imgs[index]
    # img = Image.open(fn).convert('RGB')
    img = Image.open(fn).convert('L')
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      label = self.target_transform(label)
    # For MSE loss
    label = float(label)
    # label = float((label + 1)/3)
    # return img, label, fn
    return img, label
  def __len__(self):
    return len(self.imgs)

## TODO: helper functions
def load_split_train_test(datadir, valid_size = .2):
  train_transforms = transforms.Compose([transforms.ToTensor()])

  test_transforms = transforms.Compose([transforms.ToTensor()])

  train_data = datasets.ImageFolder(datadir, transform=train_transforms)
  test_data = datasets.ImageFolder(datadir, transform=test_transforms)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(valid_size * num_train))
  np.random.shuffle(indices)
  train_idx, test_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  test_sampler = SubsetRandomSampler(test_idx)
  trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
  testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=TRAIN_BATCH_SIZE)
  return trainloader, testloader

# Prepare data
# Prepare data
train_transforms = transforms.Compose([
  transforms.ColorJitter(0.25, 0.5, 0.5, 0.25),
  transforms.ToTensor()])

# train_transforms_target = transforms.Compose([transforms.ToTensor()])
full_dataset = BlurImageDataset(IMAGE_PARENT, IMAGE_W_LABEL_TXT, transform=train_transforms)

train_size = int(TRAIN_SET_RATIO * len(full_dataset))
test_size = len(full_dataset) - train_size
print(f"Train size:\t{train_size}\tTest size\t{test_size}")

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare Model
model = models.resnet18(pretrained=USE_PRETRAINED)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
print("Model loaded!")

if MODE_FEATURE_EXTRACT:
  for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(512, 128),
             nn.ReLU(),
             nn.Dropout(0.4),
             nn.Linear(128, 1)
             )

criterion = nn.MSELoss()
# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

# Prepare training process
epochs = 100
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

print("Training start!")
for epoch in range(epochs):
  times = epoch
  unit = 0.003
  optimizer = optim.Adam(model.fc.parameters(), lr=unit* (0.7 ** times))

  for inputs, labels in trainloader:
    steps += 1
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = labels.to(dtype=torch.float32)
    labels = labels.unsqueeze(1)
    optimizer.zero_grad()
    logps = model.forward(inputs)
    loss = criterion(logps, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if steps % print_every == 0:
      train_losses.append(running_loss / len(trainloader))
      print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / print_every:.3f}.. "
          )

      running_loss = 0
      model.train()
  
  if epoch % print_every == print_every - 1:
    model_file_name = f'{MODEL_PREFIX}_{epoch}.pt'
    print(f'Saving model to :{model_file_name}')
    torch.save(model, model_file_name)

