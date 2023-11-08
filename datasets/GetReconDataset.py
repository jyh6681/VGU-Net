#!/usr/bin/env python
# coding: utf-8

# ## 玖零猴的Demo

# In[4]:


import os
import numpy as np
import SimpleITK as sitk

# In[5]:


flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"
slice_list=[50,100]
# In[6]:

bratshgg_path = r"./2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG"
# In[7]:
bratslgg_path = r"./2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/LGG"
# In[8]:
outputImg_path = r"./2-MICCAI_BraTS_2018/BraTS2018_recon"
# In[10]:
def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files
if not os.path.exists(outputImg_path):
    os.makedirs(outputImg_path,exist_ok=True)
pathhgg_list = file_name_path(bratshgg_path)
pathlgg_list = file_name_path(bratslgg_path)
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
        # tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp = slice
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        tmp[tmp == tmp.min()] = 0 #黑色背景区域
        tmp = tmp/tmp.max()
        return tmp
def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]
# In[12]:


for subsetindex in range(len(pathhgg_list)):
    brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
    # 获取每个病例的四个模态及Mask的路径
    flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
    # 获取每个病例的四个模态及Mask数据
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    #对四个模态分别进行标准化,由于它们对比度不同
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    #裁剪(偶数才行)
    flair_crop = crop_ceter(flair_array_nor,160,160)
    t1_crop = crop_ceter(t1_array_nor,160,160)
    t1ce_crop = crop_ceter(t1ce_array_nor,160,160)
    t2_crop = crop_ceter(t2_array_nor,160,160)
    print(str(pathhgg_list[subsetindex]))
    for n_slice in slice_list:
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
        imagepath = outputImg_path + "//" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
        np.save(imagepath, FourModelImageArray)  # (160,160,4) np.float dtype('float64')
print("Done！")

# In[13]:

for subsetindex in range(len(pathlgg_list)):
    brats_subset_path = bratslgg_path + "/" + str(pathlgg_list[subsetindex]) + "/"
    # 获取每个病例的四个模态及Mask的路径
    flair_image = brats_subset_path + str(pathlgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t2_name
    # 获取每个病例的四个模态及Mask数据
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    #对四个模态分别进行标准化,由于它们对比度不同
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    #裁剪(偶数才行)
    flair_crop = crop_ceter(flair_array_nor,160,160)
    t1_crop = crop_ceter(t1_array_nor,160,160)
    t1ce_crop = crop_ceter(t1ce_array_nor,160,160)
    t2_crop = crop_ceter(t2_array_nor,160,160)
    print(str(pathhgg_list[subsetindex]))
    for n_slice in slice_list:
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
        imagepath = outputImg_path + "//" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
        np.save(imagepath, FourModelImageArray)  # (160,160,4) np.float dtype('float64')
print("Done!")

