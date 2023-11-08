# -*- coding: utf-8 -*-

import argparse
from glob import glob
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from baseline_model.unet2plus import UNet_2Plus
# from baseline_model.modeling.deeplab import *
# from baseline_model.UNet3Plus import *
from baseline_model.deeplabv3 import DeepLabV3
from baseline_model.attention_unet import AttU_Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from baseline_model.Sparse_VGUNet import VGUnet
from dataset import Dataset
from metrics import iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
import unet
from baseline_model.vision_transformer import SwinUnet as ViT_seg
from skimage.io import imsave
import os
from torch.nn import SyncBatchNorm
from config import get_config
import torch.distributed as dist
arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def get_param_num(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total params: %d,  trainable params: %d' % (total, trainable))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default=train,help="train or test")
    parser.add_argument('--name', default="msugnet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument("-a",'--appdix', default="normal",)
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--saveout', default=False, type=str2bool)
    parser.add_argument('--dataset', default="jiu0Monkey",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('-e','--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early_stop', default=100, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--pretrain', default=True, type=str2bool,
                        help='nesterov')
    parser.add_argument('--cfg', type=str, required=False,default="./swinunet.pth", metavar="FILE", help='path to config file', )
    parser.add_argument('--local_rank', default=1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    device = torch.device("cuda", args.local_rank)
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)

        # compute output
        if args.deepsupervision:
            outputs = model(input)
            if args.name == "dual":
                b, c, h, w = input.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            if args.name == "dual":
                b,c,h,w = input.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def rgb_out(output):
    rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
    for idx in range(output.shape[1]):
        for idy in range(output.shape[2]):
            if output[0, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 0
                rgbPic[idx, idy, 1] = 128
                rgbPic[idx, idy, 2] = 0
            if output[1, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 255
                rgbPic[idx, idy, 1] = 0
                rgbPic[idx, idy, 2] = 0
            if output[2, idx, idy] > 0.5:
                rgbPic[idx, idy, 0] = 255
                rgbPic[idx, idy, 1] = 255
                rgbPic[idx, idy, 2] = 0
    return rgbPic

def validate(args, val_loader, model, criterion,save_output=False):
    device = torch.device("cuda", args.local_rank)
    losses = AverageMeter()
    ious = AverageMeter()
    if save_output==True:
        save_path = os.path.join("../datasets/BraTs2019/rgb_results/", args.name)
        os.makedirs(save_path,exist_ok=True)
        img_path = os.path.join("../datasets/BraTs2019/rgb_results/", "img")
        os.makedirs(img_path, exist_ok=True)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                if args.name == "dual":
                    b, c, h, w = input.shape
                    output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                if args.name == "dual":
                    b, c, h, w = input.shape
                    output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                if save_output==True:
                    # img = rgb_out(output.squeeze())
                    # path = os.path.join(save_path,str(i)+".png")
                    # imsave(path,img)
                    gt_path = os.path.join(save_path, str(i) + "gt.png")
                    gt = rgb_out(target.squeeze())
                    imsave(gt_path, gt)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                # print("save path:", iou)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def main():
    args = parse_args()
    #args.dataset = "datasets"

    device = torch.device("cuda", args.local_rank)
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.name)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.name)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    #joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainImage/*')
    mask_paths = glob(r'./datasets/2-MICCAI_BraTS_2018/BraTS2018_trainMask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))
    # create model
    print("=> creating model %s" %args.name)
    if args.name=="unet":
        model = PlainUnet(in_ch=4,out_ch=3)#unet.__dict__[args.name](args)
    if args.name=="vgunet":
        model = VGUnet(in_ch=4,out_ch=3)
    get_param_num(model)
    model = model.to(device)
    if args.pretrain==True:
        print("usig pretrained model!!!")
        pretrain_pth = './models/unet/'+"unet_plain91.08.pth"#str(args.name)+'/'+str(args.name)+'_max_pool.pth'
        # pretrain_pth = './models/'+str(args.name)+'/'+str(args.name)+'_parallel_pretrained.pth'
        pretrained_model_dict = torch.load(pretrain_pth)
        model_dict = model.state_dict()
        # for k, v in pretrained_model_dict.items():
        #     print("key:", k, v)
        pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}  # filter out unnecessary keys
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=3, verbose=True,min_lr=0.0000001)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    # train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,drop_last=False)
    ##parallel
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,shuffle=False,pin_memory=True,drop_last=False)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    print("start parallel!!")

    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    best_iou = 0
    trigger = 0  ###triger for early stop
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        scheduler.step(train_log['iou'])
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou :
            trigger = 0
            if torch.distributed.get_rank() == 0:
                save_pth = 'models/'+str(args.name)+'/'+str(args.name)+"_"+str(args.appdix)+'.pth'
                os.makedirs('models/'+str(args.name)+'/',exist_ok=True)
                torch.save(model.state_dict(), save_pth)
                best_iou = val_log['iou']
                print("=> saved best model")
                

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
                break
                break
            

        torch.cuda.empty_cache()
    return 0
def test():
    args = parse_args()
    #args.dataset = "datasets"
    device = torch.device("cuda", args.local_rank)

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.name)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.name)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    #joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob(r'./datasets/BraTs2019/testImage/*')
    mask_paths = glob(r'./datasets/BraTs2019/testMask/*')

    # create model
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
        model = ViT_seg(config,  num_classes=3)
    model = model.to(device)
    pretrain_pth = "msugnet_pretrain_unet_softmax+norm_noclip_nofix.pth"#args.name+"_normal.pth"#"./models/"+args.name+"/" + args.name+"_normal.pth"##"msugnet_pretrain_on_unet89.58.pth"  # str(args.name)+'/'+str(args.name)+'_max_pool.pth'
    pretrained_model_dict = torch.load(pretrain_pth)
    model.load_state_dict(pretrained_model_dict)
    # if args.pretrain==True:
    #     pretrain_pth = './models/msugnet/'+"msugnet_pretrain_on_unet_fixencoder.pth"#str(args.name)+'/'+str(args.name)+'_max_pool.pth'
    #     pretrained_model_dict = torch.load(pretrain_pth)
    #     model_dict = model.state_dict()
    #     # for k, v in pretrained_model_dict.items():
    #     #     print("key:", k, v)
    #     pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}  # filter out unnecessary keys
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    print(count_params(model))

    test_dataset = Dataset(args, img_paths, mask_paths)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    test_log = validate(args, test_loader, model, criterion,save_output=args.saveout)

    print('loss %.4f - iou %.4f'
        %(test_log['loss'], test_log['iou']))

    torch.cuda.empty_cache()
if __name__ == '__main__':
    args = parse_args()
    if args.action=="train":
        main()
    if args.action == "test":
        test()
