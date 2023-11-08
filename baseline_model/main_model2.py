# -*- coding: utf-8 -*-
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import PolypDataset
from utils.utils import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#root = r'/home/xwq/my/renji/data'
root = r'./data/sim_data/'
data_path = root
pred_path = os.path.join(root, 'test_pred_jyh')
compare_path = os.path.join(pred_path, 'compare')
compare_path2 = os.path.join(pred_path, 'compare2')

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
y_transforms = transforms.ToTensor()
# writer = SummaryWriter()

def aug(x,augtiems=4):
    x = x.numpy()    ####tensor to numpy
    #print("x:",x.shape[0],x.shape[1],x.shape[2],x.shape[3])
    out = np.zeros([x.shape[0]*augtiems,x.shape[1],x.shape[2],x.shape[3]])
    if augtiems==1:
        out[0:x.shape[0],:,:,:]=x
    if augtiems==2:
        out[0:x.shape[0], :, :, :] = x
        for i in range(0,x.shape[0]):
            out[x.shape[0]+i, 0, :, :] = np.flipud(x[i, 0, :, :])
    if augtiems == 3:
        out[0:x.shape[0],:,:,:]=x
        for i in range(0, x.shape[0]):
            out[x.shape[0] + i, 0, :, :] = np.flipud(x[i, 0, :, :])
            out[x.shape[0]*2+i, 0, :, :] = np.rot90(x[i, 0, :, :],k=2)
    if augtiems==4:
        out[0:x.shape[0], :, :, :] = x
        for i in range(0, x.shape[0]):
            out[x.shape[0] + i, 0, :, :] = np.flipud(x[i, 0, :, :])
            out[x.shape[0] * 2 + i, 0, :, :] = np.rot90(x[i, 0, :, :], k=2)
            out[x.shape[0] * 3 + i, 0, :, :] = np.flip(np.rot90(x[i, 0, :, :], k=3))
    return torch.from_numpy(out)




def train_model(model1, model2, optimizer, dataload, num_epochs, model_save_path, start_epoch,dist_alpha,data_index,band,ratio,res_alpha,scheduler,tmp_alpha):
    model_save_path = "epoch_" + str(100) + "_" +"dist_alpha"+str(dist_alpha)+ model_save_path
    Loss_list = []
    epoch_list = []
    dice_list = []
    criterion = Multi_DiceLoss()
    for e in range(num_epochs):
        epoch = e + start_epoch
        epoch_loss = 0
        epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4 = 0, 0, 0, 0
        step = 0
        augtimes=4
        min =1000
        mid_epoch = 29
        if epoch > mid_epoch:
            epoch_list.append(e)
        for x, y, z in dataload:
            step += 1
            x = aug(x,augtimes)
            x = x.type(torch.FloatTensor)
            inputs = x.to(device)                  #######################x with size [4, 1, 128, 128],在此将x 增广
            #print("img_x"*10,x.size())
            y = aug(y, augtimes)
            y = y.type(torch.FloatTensor)
            labels1 = y.to(device)                 ######################label1
            z = aug(z, augtimes)
            z = z.type(torch.FloatTensor)
            labels2 = z.to(device)                 ######################label2

            optimizer.zero_grad()
            outputs1 = model1(inputs)                  ###################input to model##############
            # loss1 = dice_and_BCE(outputs1, labels1, alpha=0.0)
            loss1 = criterion(outputs1,labels1)
            # mid_epoch = math.floor(num_epochs/2)
            if epoch < mid_epoch+1:
                loss = loss1
                loss2 = loss1
                loss3 = loss1
            if epoch > mid_epoch:
                # print("start aforetrain")
                # mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=1)
                #outputs2 = model2(torch.cat([mask1,torch.mul(mask1,inputs)],dim=1))
                # print("try:", np.array(outputs1.cpu()))
                # mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=1)
                mask1 = torch.argmax(outputs1,dim=1)
                mask1 = torch.unsqueeze(mask1,dim=1)
                # gall_tmp = expand_img(mask1.cpu(),ratio = ratio)###narrow band
                # mask1 = gall_tmp.to(device)
                outputs2 = model2(torch.mul(mask1, inputs))
                # loss2 = dice_and_BCE(outputs2, labels2, alpha=0.0)
                loss2 = criterion(outputs2,labels2)
                #loss3 = init_loss(outputs1, outputs2, alpha=2)
                # mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=1)
                # mask2 = thres_process(outputs2.sigmoid(), args.threshold, max=1)
                mask2 = torch.argmax(outputs2,dim=1)

                loss = 0.05 * loss1 + loss2 #+ loss3  ###################total loss########
                # print("try:",np.array(mask1.cpu()))
                loss3 = dist_loss(mask1.cpu(), mask2.cpu(), alpha=dist_alpha, band= band)     #######using 1-mask1 dist=d(1-mask1,mask2)
                # print("loss3:",loss3,type(loss3))
                ###resloss
                loss4 = res_loss(labels1,labels2,outputs1[:,1,:,:],outputs2[:,1,:,:],alpha=res_alpha)
                # print("loss1:", loss1, "loss2:", loss2,"loss4:",loss4)
                loss = loss + tmp_alpha*loss3 + loss4
                epoch_loss4 += loss4
            scheduler.step(loss)
            # print("xxxx:", tmp_alpha.item(), tmp_alpha.grad)
            loss.backward()
            # print("before:",tmp_alpha.item(),tmp_alpha.grad)
            optimizer.step()
            # print("after:",tmp_alpha.item(),tmp_alpha.grad)
            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            #epoch_loss3 += loss3.item()

        if epoch > mid_epoch:
            dce2 = criterion(outputs2, labels2)
        else:
            dce2 = 0
        if (epoch+1) % 10 == 0 or (epoch+1) == num_epochs:
            dce1 = criterion(outputs1, labels1)
            print('-' * 10)
            print("epoch %d loss:%f , dice1: %f, dice2: %f" % (epoch+1, epoch_loss/step, dce1, dce2))
            print("loss1:%f , loss2: %f, loss3: %f, loss4: %f"
                  % (epoch_loss1 / step, epoch_loss2 / step, 0 / step, epoch_loss4 / step))
            state = {'net1': model1.state_dict(), 'net2': model2.state_dict(),
                     'optim': optimizer.state_dict(), 'epoch': epoch+1}
            #print("path",model_save_path)
            if epoch>60:
                if loss < min:
                    min = loss
                    print("min:",min)
                    model_save_path = "./file21_11_8/res_alpha"+str(res_alpha)+"band_"+ str(band)+"ratio_"+ str(ratio)+"epoch_" + str(epoch+1) + "_"   + "dsit_alpha" + str(dist_alpha)+"data_"+str(data_index) + '.pth'
                    torch.save(state, model_save_path)
        if epoch > mid_epoch:
            dice_list.append(float(dce2))
            Loss_list.append(epoch_loss/step)
    print(epoch_list)
    print([round(i,4) for i in Loss_list])
    print("dice:",[round(i,4) for i in dice_list])
    # np.save("./tmp/epoch_" + str(epoch+1) + "_"   + "a2_" + str(dist_alpha)+"data_"+str(data_index) + 'epoch_list.npy',epoch_list)
    # np.save("./tmp/epoch_" + str(epoch + 1) + "_" + "a2_" + str(dist_alpha) + "data_" + str(data_index) + 'Loss_list.npy',
    #         Loss_list)
    # np.save("./tmp/epoch_" + str(epoch + 1) + "_" + "a2_" + str(dist_alpha) + "data_" + str(data_index) + 'dice_list.npy',
    #         dice_list)
    ax = plt.gca()
    ax.set_ylim(0.0,3.2)
    plt.plot(epoch_list, Loss_list, 'r-',label="loss")
    plt.plot(epoch_list, dice_list, 'b-',label="dice")
    plt.title('Training loss&dice vs. epoches')
    plt.xlabel("epoches")
    plt.ylabel('Training loss&dice')
    plt.legend()

    plt.savefig("./file21_11_8/epoch_" + str(epoch+1) + "_"   + "dist_alpha" + str(dist_alpha)+"data_"+str(data_index) + '.png', bbox_inches='tight')
        # writer.add_scalar('train/loss', epoch_loss/step, epoch)
        # writer.add_scalar('train/loss1', epoch_loss1 / step, epoch)
        # writer.add_scalar('train/loss2', epoch_loss2 / step, epoch)
        # writer.add_scalar('train/loss3', epoch_loss3 / step, epoch)
        # writer.add_scalar('train/loss4', epoch_loss4 / step, epoch)



def train(args):
    data_index = args.data
    dist_alpha = args.dist_alpha
    model1 = Unet().to(device)
    #model2 = Unet(in_ch=2).to(device)
    model2 = Unet(in_ch=1).to(device)
    tmp_alpha = torch.autograd.Variable(torch.tensor(0.5), requires_grad=True)
    optimizer = optim.Adam([{'params':model1.parameters()},{'params':model2.parameters()},{'params':tmp_alpha}],
                             lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=5, verbose=True,
                                                           min_lr=0.00000001)
    if args.Continue is True:                     #####判断是否继续训练
        # model_save_path = "epoch_" + str(100) + "_" + "a_" + str(args.alpha) + str(args.ckpt)
        model_save_path = "./tmp/alpha20.2band_2.0ratio_1.1epoch_70_a2_0.1data_100.pth"
        if os.path.exists(model_save_path):
            traindata = txt_read('tmptrain_epoch100.txt')
            checkpoint = torch.load(model_save_path)                # 加载之前已经训练过的模型参数
            model_dict1 = model1.state_dict()
            model_dict2 = model2.state_dict()
            pretrained_dict1 = {k: v for k, v in checkpoint.items() if
                               k in model_dict1}  # filter out unnecessary keys
            pretrained_dict2 = {k: v for k, v in checkpoint.items() if
                                k in model_dict2}  # filter out unnecessary keys
            model_dict1.update(pretrained_dict1)
            model1.load_state_dict(model_dict1)
            model_dict2.update(pretrained_dict2)
            model2.load_state_dict(model_dict2)
            # model1.load_state_dict(checkpoint['net1'])
            # model2.load_state_dict(checkpoint['net2'])
            optimizer.load_state_dict(checkpoint['optim'])
            start_epoch = checkpoint['epoch']
            print('-----Read Checkpoint Success-----')
    else:
        start_epoch = 0
        train_path = "./sim_train_epoch" + str(args.data) + ".txt"
        if args.data==100:
            traindata = txt_read('tmptrain_epoch100.txt')
        if args.data==140:
            traindata = txt_read('train_epoch140.txt')
        if args.data==150:
            traindata = txt_read('train_epoch150.txt')
        if args.data == 130:
            traindata = txt_read('sim_train_epoch130.txt')
        if args.data == 170:
            traindata = txt_read('sim_train_epoch170.txt')
        # if os.path.exists(train_path):
        #     traindata = txt_read(train_path)
        # else:
        #     print("--------making dataset--------")
        #     data = random_dataset(data_path)
        #     random.shuffle(data)
        #     testdata = data[:len(data) // 10]
        #     traindata = data[len(data) // 10:]
        #     txt_save('sim_train_epoch%d.txt'%args.epochs, traindata) #############固定用同一个数据集 #./data\train\image\3\25.bmp ./data\train\mask\3\25.bmp ./data\train\mask\3\25mask.bmp
        #     txt_save('sim_test_epoch%d.txt'%args.epochs, testdata)

    #traindata = make_dataset(os.path.join(data_path, 'train')) ######traindata包含全部数据路径的三元组
    pancreas_dataset = PolypDataset(traindata, transform=x_transforms, target_transform=y_transforms)#####对traindata读取数据，归一化和转为tensor，输出仍为三元组
    dataloaders = DataLoader(pancreas_dataset, batch_size=args.batch_size, shuffle=True)
    get_param_num(model1)
    get_param_num(model2)
    print('-----Wait For Training-----')
    train_model(model1, model2, optimizer, dataloaders, args.epochs, args.ckpt, start_epoch,dist_alpha,data_index,args.band, args.ratio,args.res_alpha,scheduler,tmp_alpha)     ######################train model
    # writer.close()


def len_print(list,lenth):
    for index in range(0,len(list)):
        tmp=index+1
        if tmp%lenth == 0:
            print(round(list[index],4),"//",end='')
        if tmp%lenth!=0 and tmp != len(list):
            print(round(list[index],4),end=' ')
        if tmp == len(list):
            print(round(list[index],4))


def test(args):
    model1 = Unet()
    #model2 = Unet(in_ch=2)
    model2 = Unet(in_ch=1)#.to(device)
    epoch_state = range(65,70,10)
    beta = [0.55]
    len_beta = len(beta)
    ratio_list = [1.05]
    index_epoch=[]
    index_band = []
    beta_index=[]
    gall_dice=[]
    polyp_dice=[]
    scale_gall = []
    scale_index = []
    gall_acc = []
    polyp_acc = []
    res_alpha = 0.55
    band = 2.0
    dist_alpha = 0.0
    data_index = 100
    for index in epoch_state:
        for ratio in ratio_list:
            # for b in beta:
            #     temp_path = "./bestmodel/"+"Mhuber0.015loss1_res_alpha"+str(res_alpha)+"band_"+ str(band)+"ratio_"+ str(ratio)+"epoch_" + str(index) + "_" + "dist_alpha" + str(dist_alpha)+"data_"+str(data_index) + '.pth'
                temp_path = "./bestmodel/Mhuber0.015loss1_res_alpha0.55band_2.0ratio_1.05epoch_65_dsit_alpha0.0data_100.pth"
                checkpoint = torch.load(temp_path, map_location='cpu')
                model1.load_state_dict(checkpoint['net1'])
                model2.load_state_dict(checkpoint['net2'])
                print('-----Read Checkpoint Success-----')
                #testdata = make_dataset(os.path.join(data_path, 'test'))
                testdata = txt_read('test_epoch100.txt')              ###固定用同一个数据集十分之一测试 剩余训练
                pancreas_dataset = PolypDataset(testdata, transform=x_transforms, target_transform=y_transforms)
                dataloaders_test = DataLoader(pancreas_dataset, batch_size=1)
                model1.eval()
                model2.eval()
                num = 0
                average_dice = 0
                average_dice2 = 0
                average_acc1 = 0
                average_acc2 = 0
                average_scalegall = 0
                average_scale = 0
                mkdir(compare_path)
                mkdir(compare_path2)
                with torch.no_grad():            # 不更新梯度
                    for x, y, z in dataloaders_test:
                        outputs1 = model1(x)
                        mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=1)
                        prediction1 = outputs1.sigmoid()
                        # mask1 = expand_img(mask1.cpu(), ratio=ratio)
                        prediction2 = model2(torch.mul(mask1,x)).sigmoid()
                        x = torch.reshape(x, (1, x.shape[2], x.shape[3]))
                        x_np = tensor2im(x)
                        pred_y = torch.squeeze(prediction1).numpy()
                        pred_z = torch.squeeze(prediction2).numpy()
                        pred_y[pred_y > args.threshold] = 1
                        pred_y[pred_y <= args.threshold] = 0
                        pred_z[pred_z > args.threshold] = 1
                        pred_z[pred_z <= args.threshold] = 0
                        result = pred_y.astype(np.uint8)
                        result = get_max_region(result)   ##only one gallbladder
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
                        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                        cv2.imwrite(os.path.join(pred_path, str(num) + '.jpg'), result) #####写预测图像到pred_path文件夹 bmp->jpg


                        result2 = pred_z.astype(np.uint8)
                        result2 = get_max_region(result2)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        result2 = cv2.morphologyEx(result2, cv2.MORPH_CLOSE, kernel)
                        cv2.imwrite(os.path.join(pred_path, str(num) + 'mask.jpg'), result2)

                        # 计算dice
                        y = torch.squeeze(y).numpy()
                        pred_img = result.copy()
                        pred_img[pred_img > 0] = 1
                        dice = get_dice(pred_img, y)
                        average_dice += dice
                        acc1 = accuracy(pred_img, y)
                        average_acc1 += acc1

                        print('Gallbladder Dice of Number %s is %s' % (num, dice))

                        z = torch.squeeze(z).numpy()
                        pred_img2 = result2.copy()
                        pred_img2[pred_img2 > 0] = 1
                        dice2 = get_dice(pred_img2, z)
                        average_dice2 += dice2
                        scale = get_scale(pred_img2, z)
                        scale_gall_tmp = get_scale(pred_img, y)
                        average_scale += scale
                        average_scalegall += scale_gall_tmp
                        print("average_scalga:", average_scalegall)
                        acc2 = accuracy(pred_img2, z)
                        average_acc2 += acc2
                        print('Polyp Dice of Number %s is %s' % (num, dice2))

                        # 画对比图
                        ct_img = x_np.copy()
                        gallmask = y.copy()        # 不复制会报错
                        polypmask = z.copy()        # 不复制会报错
                        OverlayMaskOnCT(ct_img, gallmask, polypmask, pred_img, pred_img2, num, compare_path)

                        num += 1
                    index_band.append(ratio)
                    # beta_index.append(b)
                    index_epoch.append(index)
                    gall_dice.append(average_dice/num)
                    polyp_dice.append(average_dice2 /num)
                    scale_index.append(average_scale / num)
                    scale_gall.append(average_scalegall / num)
                    gall_acc.append(average_acc1 / num)
                    polyp_acc.append(average_acc2 / num)
                    print('Average dice is %s' % (average_dice/num))
                    print('Average dice2 is %s' % (average_dice2 / num))
    print('epoch:',index_epoch)
    print('beta:', beta_index)
    print("band:",index_band)
    print("gall_dice:", end='')
    len_print(gall_dice,len_beta)
    print("polyp_dice:", end='')
    len_print(polyp_dice, len_beta)
    print("scale_polyp:", end='')
    len_print(scale_index, len_beta)
    print("scale_gall:", end='')
    len_print(scale_gall, len_beta)
    print("gall_accuracy:", end='')
    len_print(gall_acc, len_beta)
    print("polyp_accuracy:", end='')
    len_print(polyp_acc, len_beta)
    plt.barh(index, polyp_dice)  # 横放条形图函数 barh
    plt.title('polyp dice ')
    #plt.show()
    return average_dice/num


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, default=train,help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--learning_rate", type=float, default=1e-4)
    parse.add_argument("--epochs", type=int, default=120)
    parse.add_argument("--threshold", type=float, default=0.3)
    parse.add_argument("--ckpt", type=str, default='polyp.pth', help="the path of model weight file")
    parse.add_argument('--Continue', action='store_true', default=False, help='continue training')
    parse.add_argument('--dist_alpha', type=float, default=0.1)
    parse.add_argument('--band', type=float, default=2.0)
    parse.add_argument("--data", type=int, default=100)
    parse.add_argument("--ratio", type=float, default=1.05)
    parse.add_argument('--res_alpha', type=float, default=0.7)
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    if args.action == "test":
        test(args)
