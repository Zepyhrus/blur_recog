#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:25:50 2019

@author: ubuntu
"""
import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='output/snapshots/hp_epoch.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path


    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [2,3, 2, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)


    transformations = transforms.Compose([transforms.Scale(128),
    transforms.CenterCrop(128), transforms.ToTensor()])

    model.cuda(gpu)
    

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    example=torch.rand(1,1,128,128).cuda()
    traced_script_module=torch.jit.trace(model,example)
    traced_script_module.save('face_pose.pt')
    #total = 0

   # idx_tensor = [idx for idx in range(66)]
    #idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    #video = cv2.VideoCapture(0)

    # New cv2
    #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object


    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))


'''
    frame_num = 1

    while video.isOpened():
        kk=cv2.waitKey(1)
        ret,frame = video.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                
                yaw_predicted =yaw_predicted.cpu().data.numpy()
                pitch_predicted =pitch_predicted.cpu().data.numpy()
                roll_predicted =roll_predicted.cpu().data.numpy()
                print(str(idx)+':')
                print(str(yaw_predicted)+' '+str(pitch_predicted)+' '+str(roll_predicted)+'\n')
                # Print new frame with cube and axis
                #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
        cv2.imshow("camera",frame)
        if(kk==ord('q')):
            break
        frame_num += 1


    video.release()
'''
'''
for i in range(10,34,1):
    if not os.path.exists("./face_collect/00"+str(i)+'d'):
        os.makedirs("./face_collect/00"+str(i)+'d')
    for filename in os.listdir(r"./face_collect/00"+str(i)):
        print(filename)
        img=cv2.imread('face_collect/00'+str(i)+'/'+filename)
        img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img1 = Image.fromarray(img1)

# Transform
        img1 = transformations(img1)
        img_shape = img1.size()
        img1 = img1.view(1, img_shape[0], img_shape[1], img_shape[2])
        img1 = Variable(img1).cuda(gpu)

        yaw, pitch, roll = model(img1)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
# Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
        yaw_predicted =yaw_predicted.cpu().data.numpy()
        pitch_predicted =pitch_predicted.cpu().data.numpy()
        roll_predicted =roll_predicted.cpu().data.numpy()
        if(yaw_predicted>50 or yaw_predicted<-50):
            os.remove('face_collect/00'+str(i)+'/'+filename)
            cv2.imwrite('face_collect/00'+str(i)+'d/'+filename,img)
'''        
