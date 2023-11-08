import os
import argparse
import shutil
from utils import str2bool, count_params
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="msugnet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--appdix', default="normal",)
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
    parser.add_argument('--savegt', default=False, type=str2bool,)

    args = parser.parse_args()

    return args
args = parse_args()
name = args.name
###select result to show in paper
model_list = ["deeplabv3","unet","unet++","attunet","msugnet"]
for name in model_list:
    save_path = os.path.join("../datasets/BraTs2019/rgb_results/", name)
    new_path = os.path.join("../datasets/BraTs2019/rgb_results/rgb_send", name)
    os.makedirs(new_path,exist_ok=True)
    select_item = [3,8,47,54,87,97,130,15,26,40,83,122,148,187]
    for i in select_item:
        file = os.path.join(save_path,str(i)+".png")
        new_file = os.path.join(new_path, name+str(i) + ".png")
        shutil.copy(file,new_file)
        if name=="msugnet":
            gt_file = os.path.join(save_path,str(i)+"gt.jpg")
            new_gt = os.path.join(new_path,str(i)+"gt.jpg")
            shutil.copy(gt_file, new_gt)
