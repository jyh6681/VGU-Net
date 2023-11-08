import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
save_list = [3,8,47,54,87,97,130,15,26,40,83,122,148,163]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths,aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        # self.index=select_index ##for test

    def __len__(self):
        return len(self.img_paths)##1 for test

    def __getitem__(self, idx):
        # idx=self.index  ##test mode
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        # print("show name:",img_path)
        npimage = np.load(img_path)
        if idx in save_list:
            # print("idx:",idx,npimage.shape)
            cv2.imwrite("./saved_testinput_origin/flair_" + str(idx) + ".png", 255*npimage[:,:,0])
            cv2.imwrite("./saved_testinput_origin/t1_" + str(idx) + ".png", 255*npimage[:,:,1])
            cv2.imwrite("./saved_testinput_origin/t1ce_" + str(idx) + ".png", 255*npimage[:,:,2])
            cv2.imwrite("./saved_testinput_origin/t2_" + str(idx) + ".png", 255*npimage[:,:,3])
        #print("load image:",np.max(npimage))
        npmask = np.load(mask_path)
        # print("shape::::",npimage.shape)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel


        #读图片（如jpg、png）的代码
        '''
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = color.gray2rgb(image)
        #image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))       
        return image, mask
        '''

