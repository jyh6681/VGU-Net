#sleep 4h
# CUDA_VISIBLE_DEVICES=5 python3  test.py --name swinunet --mode GetPicture
#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 train_sparse.py train --name vgunet -a notfixed_2gin_lr1e-3_200_80_20sparse --lr 1e-3 -b 32 -e 500 --early_stop 30
#CUDA_VISIBLE_DEVICES=7,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 1236 train.py  train --name largeunet -a apex -b 18


# CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29503 train.py train --name msugnet -a testmodnotfixed_2gin_lr1e-3_80_20_10sparse --lr 1e-3 -b 50

#CUDA_VISIBLE_DEVICES=1 python3 test.py --name dual --mode GetPicture -b 1
#CUDA_VISIBLE_DEVICES=1 python3 test.py --name dual --mode Calculate -b 1
#CUDA_VISIBLE_DEVICES=5 python3 test.py --name unet++
#CUDA_VISIBLE_DEVICES=5 python3 test_3D.py --name attunet --mode GetPicture -b 1
#sleep 2.5h
CUDA_VISIBLE_DEVICES=4 python3 test.py --name unet --mode GetPicture
CUDA_VISIBLE_DEVICES=4 python3 test.py --name unet --mode Calculate
#CUDA_VISIBLE_DEVICES=2,7,0,1 python3 -m torch.distributed.launch --nproc_per_node=4 train.py  train --name swinunet -a 23.10.10_patch4_window5