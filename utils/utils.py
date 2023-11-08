
# from mayavi import mlab
# from GetTestingSetsFrom2019 import *
import pandas as pd
from PIL import Image
import numpy as np
import SimpleITK as sitk
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

def plot_attention(image):
    # 假设您已经有一张名为image的图像，以及查询区域的坐标query_patch_coords和注意力较高区域的坐标attention_patch_coords列表
    # image = ...
    # query_patch_coords = [(x1, y1, x2, y2), ...]
    # attention_patch_coords = [(x1, y1, x2, y2), ...]

    # 将图像转换为OpenCV的格式
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 标注查询区域
    for (x1, y1, x2, y2) in query_patch_coords:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 标注注意力较高的区域
    for (x1, y1, x2, y2) in attention_patch_coords:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 连接查询区域和注意力较高区域
    for (x1, y1, x2, y2) in query_patch_coords:
        for (x3, y3, x4, y4) in attention_patch_coords:
            cv2.line(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (int((x3 + x4) / 2), int((y3 + y4) / 2)), (0, 255, 0), 2)

    # 显示标注后的图像
    cv2.imshow('Image with Annotations', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """

    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp
def crop_ceter(img,croph,cropw):
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:,starth:starth+croph,startw:startw+cropw]
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recover_crop(path,pred_mask,croph=160,cropw=160):
    mask_image = path + mask_name
    mask_image = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    mask_image = sitk.GetArrayFromImage(mask_image)
    print("array:",mask_image.shape,pred_mask.shape,np.max(pred_mask))
    height, width = mask_image[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    mask_image[:, starth:starth + croph, startw:startw + cropw] = pred_mask
    return mask_image.transpose((1,2,0)).transpose((1,0,2))

def load_files(path):
    # 获取病例的四个模态及Mask的路径
    flair_image = path + flair_name
    t1_image = path + t1_name
    t1ce_image = path + t1ce_name
    t2_image = path + t2_name
    mask_image = path + mask_name
    # 获取每个病例的四个模态及Mask数据
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)

    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)
    # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
    # flair_array_nor = sitk.GetArrayFromImage(flair_src)
    # t1_array_nor = sitk.GetArrayFromImage(t1_src)
    # t1ce_array_nor = sitk.GetArrayFromImage(t1ce_src)
    # t2_array_nor = sitk.GetArrayFromImage(t2_src)
    # mask_array = sitk.GetArrayFromImage(mask)
    ###mynorm
    # flair_array_nor = (flair_array_nor - flair_array_nor.min()) / (
    #             flair_array_nor.max() - flair_array_nor.min())  # sitk.GetArrayFromImage(flair_src)
    # t1_array_nor = (t1_array_nor - t1_array_nor.min()) / (
    #             t1_array_nor.max() - t1_array_nor.min())  # sitk.GetArrayFromImage(t1_src)
    # t1ce_array_nor = (t1ce_array_nor - t1ce_array_nor.min()) / (
    #             t1ce_array_nor.max() - t1ce_array_nor.min())  # sitk.GetArrayFromImage(t1ce_src)
    # t2_array_nor = (t2_array_nor - t2_array_nor.min()) / (
    #             t2_array_nor.max() - t2_array_nor.min())  # sitk.GetArrayFromImage(t2_src)
    # 对四个模态分别进行标准化,由于它们对比度不同####modified

    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    # 裁剪(偶数才行)
    flair_crop = crop_ceter(flair_array_nor, 160, 160)
    t1_crop = crop_ceter(t1_array_nor, 160, 160)
    t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
    t2_crop = crop_ceter(t2_array_nor, 160, 160)
    mask_crop = crop_ceter(mask_array, 160, 160)
    return flair_crop,t1_crop,t1ce_crop,t2_crop,mask_crop
def load_slice_img(flair_crop,t1_crop,t1ce_crop,t2_crop,mask_crop, n_slice):
    npmask = mask_crop[n_slice, :, :]

    FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float)
    flairImg = flair_crop[n_slice, :, :]
    flairImg = flairImg.astype(np.float)
    FourModelImageArray[:, :, 0] = flairImg
    t1Img = t1_crop[n_slice, :, :]
    t1Img = t1Img.astype(np.float)
    FourModelImageArray[:, :, 1] = t1Img
    t1ceImg = t1ce_crop[n_slice, :, :]
    t1ceImg = t1ceImg.astype(np.float)
    FourModelImageArray[:, :, 2] = t1ceImg
    t2Img = t2_crop[n_slice, :, :]
    t2Img = t2Img.astype(np.float)
    FourModelImageArray[:, :, 3] = t2Img

    npimage = FourModelImageArray.transpose((2, 0, 1))

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

    return npimage, nplabel

import torch
class Slice_Dataset(torch.utils.data.Dataset):
    def __init__(self, args,num_slices,flair_crop,t1_crop,t1ce_crop,t2_crop,mask_crop):
        self.args = args
        self.num = num_slices
        self.flair_crop, self.t1_crop, self.t1ce_crop, self.t2_crop, self.mask_crop = flair_crop,t1_crop,t1ce_crop,t2_crop,mask_crop
        # self.index=select_index ##for test

    def __len__(self):
        return self.num##1 for test

    def __getitem__(self, n_slice):
        npmask = self.mask_crop[n_slice, :, :]

        FourModelImageArray = np.zeros((self.flair_crop.shape[1], self.flair_crop.shape[2], 4), np.float)
        flairImg = self.flair_crop[n_slice, :, :]
        flairImg = flairImg.astype(np.float)
        FourModelImageArray[:, :, 0] = flairImg
        t1Img = self.t1_crop[n_slice, :, :]
        t1Img = t1Img.astype(np.float)
        FourModelImageArray[:, :, 1] = t1Img
        t1ceImg = self.t1ce_crop[n_slice, :, :]
        t1ceImg = t1ceImg.astype(np.float)
        FourModelImageArray[:, :, 2] = t1ceImg
        t2Img = self.t2_crop[n_slice, :, :]
        t2Img = t2Img.astype(np.float)
        FourModelImageArray[:, :, 3] = t2Img

        npimage = FourModelImageArray.transpose((2, 0, 1))

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

        return npimage, nplabel

