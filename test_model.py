import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from dataloader import BlurImageDataset
IMAGE_PARENT = '/home/ubuntu/Workspace/blur_recog/blur_cam_test'
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
map_pred_index_to_label = ['0', '1', '2']

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_load = torch.load(f'./{MODEL_NAME}.pt')
model = model_load.eval()
model.to(device)
tested_cnt = 0
total_data_size = len(test_dataset)

for _, data in enumerate(test_loader, 0):
	tested_cnt += 1
	test_x, test_y = data
	test_x = test_x.to(device)
	test_y = test_y.to(device)
	pred = model.forward(test_x)
	probablity = pred.data.item()
	l1_distances = torch.abs(torch.Tensor([0, 1, 2]) - torch.Tensor([probablity])).cpu().to(device)
	y_hat = torch.argmin(l1_distances)

	y_hat_decoded = map_pred_index_to_label[y_hat]
	test_y_decoded = map_pred_index_to_label[test_y]
	if y_hat == test_y:
		correct += 1
	else:
		print(f'Wrong prediction({y_hat_decoded}=>{test_y_decoded}) for file::{1}')
	print(f'Current accuracy({tested_cnt}/{total_data_size}): {correct / tested_cnt} = {correct} / {tested_cnt}')

print("Accuracy={}".format(correct / total_data_size))