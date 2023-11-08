# -*- coding: utf-8 -*-
from mayavi import mlab
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from dataset import Dataset
# import Unet
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import *
# from sklearn.externals import joblib
from hausdorff import hausdorff_distance
import imageio
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from skimage.io import imread
from baseline_model.unet2plus import UNet_2Plus
# from baseline_model.modeling.deeplab import *
# from baseline_model.UNet3Plus import *
from baseline_model.deeplabv3 import DeepLabV3
from baseline_model.attention_unet import AttU_Net
from VGUNet import *
from models.msugnet.best_model.ModUGNet import *
from dataset import Dataset
from baseline_model.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import nibabel as nib
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="attunet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--mode', default="GetPicture", )
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--saveout', default=False, type=str2bool)
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--pretrain', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args





def main():
    args =parse_args()# joblib.load('models/%s/args.pkl' %val_args.name)
    img_Files = "/home/jyh_temp1/Downloads/BRaTS2d_send/UNet2D_BraTs-master/" \
                "datasets/BraTs2019/3D_test/Brats18_2013_2_1/Brats18_2013_2_1"
    if not os.path.exists('output3d/%s' %args.name):
        os.makedirs('output3d/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    # create model
    print("=> creating model %s" % args.name)
    if args.name == "unet":
        model = PlainUnet(in_ch=4, out_ch=3)  # unet.__dict__[args.name](args)
    if args.name == "msugnet":
        model = MulUGnet(in_ch=4, out_ch=3)
    if args.name == "unet++":
        model = UNet_2Plus(in_channels=4, n_classes=3)
    if args.name == "deeplabv3":
        model = DeepLabV3(class_num=3)
    if args.name == "attunet":
        model = AttU_Net(img_ch=4, output_ch=3)
    if args.name == "swinunet":
        config = get_config(args)
        model = ViT_seg(config,  num_classes=3).cuda()
    if args.name=="dual":
        from baseline_model.DualGCN import DualSeg_res50
        import torch.nn.functional as F
        model = DualSeg_res50(num_classes=3)
    model = model.cuda()
    print("testing moe:",args.mode)
    if args.mode == "GetPicture":
        """
        获取并保存模型生成的标签图
        """
        model = model.cuda()
        # model = nn.DataParallel(model)##parallel train need parallel test
        pretrain_pth = "./models/" + args.name + "/" + args.name + "_normal.pth"#"_parallel_pretrained.pth"  # msugnet_pretrain_unet_softmax+norm_noclip_nofix_test75.99.pth"#
        # # "./models/" + args.name + "/" + args.name + "_plain91.08.pth"  # "msugnet_pretrain_unet_softmax+norm_noclip_nofix.pth"#args.name+"_normal.pth"#"msugnet_pretrain_on_unet89.58.pth"  # str(args.name)+'/'+str(args.name)+'_max_pool.pth'
        pretrained_model_dict = torch.load(pretrain_pth,map_location="cuda:0")
        weight_dict = {}
        for k,v in pretrained_model_dict.items(): ###parallel trained model have extra module in dict
            new_k = k.replace('module.','')
            weight_dict[new_k] = v
        model.load_state_dict(weight_dict)
        model.eval()
        flair_crop,t1_crop,t1ce_crop,t2_crop,mask_crop = load_files(img_Files)
        test_dataset = Slice_Dataset(args,num_slices=mask_crop.shape[0],flair_crop=flair_crop,t1_crop=t1_crop,\
                                     t1ce_crop=t1ce_crop,t2_crop=t2_crop,mask_crop=mask_crop)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,drop_last=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                out_slices = np.zeros((mask_crop.shape[0],160,160))
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    print("i:::",i)
                    input = input.cuda()
                    #target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                        if args.name == "dual":
                            b, c, h, w = input.shape
                            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                    else:
                        output = model(input)
                        if args.name == "dual":
                            b, c, h, w = input.shape
                            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                    #print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy() ##[b,channel,h,w]
                    #print("output_shape:%s"%str(output.shape))
                    maskPic = np.zeros([160, 160])
                    for j in range(output.shape[0]): ##batch = 1
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                # print("value:",output[i,0,idx,idy],output[i,1,idx,idy],output[i,2,idx,idy])
                                if output[j,0,idx,idy] > 0.5:
                                    maskPic[idx, idy] = 2
                                if output[j,1,idx,idy] > 0.5:
                                    maskPic[idx, idy] = 1
                                if output[j,2,idx,idy] > 0.5:
                                    maskPic[idx, idy] = 4
                    out_slices[i,:,:] = maskPic
            torch.cuda.empty_cache()
        origin = nib.load(img_Files+"_seg.nii.gz")
        out_file = recover_crop(img_Files,out_slices)
        print("affine:",origin.affine)
        out_file = nib.Nifti1Image(out_file,origin.affine)
        nib.save(out_file,img_Files+"_pred.nii.gz")
        print("Done!")

if __name__ == '__main__':
    main( )
