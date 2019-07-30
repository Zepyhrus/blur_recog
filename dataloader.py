from torch.utils.data import Dataset
from PIL import Image
import torch

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
        img = Image.open(fn).convert('RGB')
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