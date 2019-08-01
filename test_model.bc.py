import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from dataloader import BlurImageDataset
IMAGE_PARENT = '/home/ubuntu/Workspace/blur_recog/blur_cam_test2'
IMAGE_W_LABEL_TXT = '../data_generator/class_id_to_files_70001_test.txt'
MODEL_NAME = 'blur_cls_resnet18_128_24'

load_test_from_file = False
threshold = 0.5

train_transforms = transforms.Compose([
	transforms.ToTensor(),
])
if load_test_from_file:
    print(f'Load from file:{IMAGE_W_LABEL_TXT}')
    test_dataset = BlurImageDataset(IMAGE_PARENT, IMAGE_W_LABEL_TXT, transform=train_transforms)
else:
	print(f'Load from folder directly:{IMAGE_PARENT}')
	test_dataset = torchvision.datasets.ImageFolder(root=IMAGE_PARENT, transform=train_transforms)

map_pred_index_to_label = ['0', '1']

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_load = torch.load(f'./{MODEL_NAME}.pt')
model = model_load.eval()
model.to(device)
tested_cnt = 0
total_data_size = len(test_dataset)

for i in range(10):
	threshold = i / 50
	correct = 0
	TP = 0
	FP = 0
	FN = 0
	for test_x, test_y in test_loader:
		tested_cnt += 1
		test_x = test_x.to(device)
		test_y = test_y.to(device)
		pred = model.forward(test_x)
		probablity = pred.data.item()
		# l1_distances = torch.abs(torch.Tensor([0, 1, 2]) - torch.Tensor([probablity])).cpu().to(device)
		if probablity > threshold:
			y_hat = 1
		else:
			y_hat = 0

		# y_hat_decoded = map_pred_index_to_label[y_hat]
		# test_y_decoded = map_pred_index_to_label[test_y]

		if y_hat == test_y.data.item():
			correct += 1
			if y_hat == 1:
				TP += 1
		else:
			if y_hat == 1:
				FP += 1
			else:
				FN += 1
		# 	print(f'Wrong prediction({y_hat}=>{test_y.data.item()})')
		# print(f'Current accuracy({tested_cnt}/{total_data_size}): {correct / tested_cnt} = {correct} / {tested_cnt}')

	print(f"Total accuracy={correct/total_data_size:.4f}, Accuracy={TP / (TP + FP):.4f}, Recall = {TP / (TP + FN):.4f}, at threshold: {threshold:.2f}")