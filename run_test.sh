#CUDA_VISIBLE_DEVICES=2,3,4,5,7 python3 -m torch.distributed.launch --nproc_per_node=5  train.py  train --name swin2 -a parallel_pretrained -b 96 --lr 5e-4 --epochs 500

#CUDA_VISIBLE_DEVICES=5 python3 test.py --name attunet
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py  train --name unet -a 23.10.11_unet_skiponbottleneck
#python3 train.py  train --name swinunet -a 23.10.10
##CUDA_VISIBLE_DEVICES=5 python3 test.py --name unet
#CUDA_VISIBLE_DEVICES=5 python3 test.py --name swinunet --mode Calculate
#CUDA_VISIBLE_DEVICES=5 python3 test.py --name swinunet --mode Calculate
###train
#CUDA_VISIBLE_DEVICES=2 python3 test_sparse.py --name vgunet --mode GetPicture
#CUDA_VISIBLE_DEVICES=2 python3 test_sparse.py --name vgunet --mode Calculate
#CUDA_VISIBLE_DEVICES=2,6 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 666 train.py  train --name swinunet -a parallel_pretrained_window2 -b 64  --lr 5e-4 --epochs 1000##correct
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py  train --name swinunet  -a oldconfig_4and7_23.1.15_parallel_nopretrained -b 160 --pretrain False --lr 5e-5 --epochs 500
# CUDA_VISIBLE_DEVICES=5 python3 test.py --name swinunet --mode GetPicture
# CUDA_VISIBLE_DEVICES=5 python3 test.py --name swinunet --mode Calculate
#CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 train.py  train --name largemsugnet -a apex -b 18
#
#CUDA_VISIBLE_DEVICES=0,2,3,6 python3 -m torch.distributed.launch --nproc_per_node=4 train.py  train --name swinunet -a apex -b 18
#
#
#CUDA_VISIBLE_DEVICES=5,4,5,7 python3 -m torch.distributed.launch --nproc_per_node=4  train.py  train --name swinunet -a parallel_pretrained -b 128  --lr 5e-4 --epochs 1000
