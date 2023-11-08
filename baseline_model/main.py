import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from unet import Unet
from resnet import resnet18
from dataset import PolypDataset, CatDataset
import cv2
import os
# from torch.utils.tensorboard import SummaryWriter
import math
from unet2plus import UNet_2Plus
from modeling.deeplab import *
from UNet3Plus import *
from deeplabv3 import DeepLabV3
from attention_unet import AttU_Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#root = r'/home/xwq/my/renji/data'
root = r'./data'
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


def train_model(model1, model2, optimizer, dataload, num_epochs, model_save_path, start_epoch,weight_alpha,dist_alpha):
    model_save_path = "epoch_" + str(num_epochs) + "_" +"a1_"+str(weight_alpha)+"a2_"+str(dist_alpha)+ model_save_path
    for e in range(num_epochs):
        epoch = e + start_epoch
        epoch_loss = 0
        epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4 = 0, 0, 0, 0
        step = 0
        augtimes=4
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
            loss1 = dice_and_BCE(outputs1, labels1, alpha=2)
            mid_epoch = math.floor(num_epochs/2)
            if epoch < mid_epoch+1:
                loss = loss1
                loss2 = loss1
                #loss3 = loss1
            if epoch > mid_epoch:
                mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=255)
                outputs2 = model2(torch.mul(mask1,inputs))
                loss2 = dice_and_BCE(outputs2, labels2, alpha=2)
                # loss3 = init_loss(outputs1, outputs2, alpha=2)
                mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=255)
                mask2 = thres_process(outputs2.sigmoid(), args.threshold, max=255)
                loss = weight_alpha * loss1 + loss2# + loss3  ###################total loss########
                loss4 = dist_loss(mask1.cpu(), mask2.cpu(), alpha=dist_alpha)
                #loss5 = scale_loss(mask2.cpu(), labels2.cpu(),alpha=dist_alpha)
                #loss4 = 0
                loss = loss + loss4
                epoch_loss4 += loss4
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            #epoch_loss3 += loss3.item()

        if (epoch+1) % 5 == 0 or (epoch+1) == num_epochs:
            dce1 = get_dice2(outputs1, labels1)
            if epoch>mid_epoch:
                dce2 = get_dice2(outputs2, labels2)
            else:
                dce2 = 0
            print('-' * 10)
            print("epoch %d loss:%f , dice1: %f, dice2: %f" % (epoch+1, epoch_loss/step, dce1, dce2))
            print("loss1:%f , loss2: %f, loss3: %f, loss4: %f"
                  % (epoch_loss1 / step, epoch_loss2 / step, 0, epoch_loss4 / step))
            state = {'net1': model1.state_dict(), 'net2': model2.state_dict(),
                     'optim': optimizer.state_dict(), 'epoch': epoch+1}
            #print("path",model_save_path)
            torch.save(state, model_save_path)
        # writer.add_scalar('train/loss', epoch_loss/step, epoch)
        # writer.add_scalar('train/loss1', epoch_loss1 / step, epoch)
        # writer.add_scalar('train/loss2', epoch_loss2 / step, epoch)
        # writer.add_scalar('train/loss3', epoch_loss3 / step, epoch)
        # writer.add_scalar('train/loss4', epoch_loss4 / step, epoch)

def train_segnet(args, model1, optimizer, dataload, num_epochs, base_model_save_path, start_epoch,data_index, weight_alpha,dist_alpha,model,scheduler):
    model_save_path = str(args.model)+"_"+"epoch_" + str(num_epochs) + "_" +"a1_"+str(weight_alpha)+"a2_"+str(dist_alpha)+ base_model_save_path
    for e in range(num_epochs):
        epoch = e + start_epoch
        epoch_loss = 0
        epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4 = 0, 0, 0, 0
        step = 0
        augtimes=4
        min=1000
        for x, y, z in dataload:
            step += 1
            x = aug(x,augtimes)
            x = x.type(torch.FloatTensor)
            inputs = x.to(device)                  #######################x with size [4, 1, 128, 128],在此将x 增广
            #print("img_x"*10,x.size())
            y = aug(y, augtimes)
            y = y.type(torch.FloatTensor)
            labels1 = y.to(device)                 ######################label1 gallbladder
            z = aug(z, augtimes)
            z = z.type(torch.FloatTensor)
            labels2 = z.to(device)                 ######################label2 polyp

            optimizer.zero_grad()
            outputs1 = model1(inputs)                  ###################input to model##############
            #print("output1:",outputs1)
            loss = dice_and_BCE(outputs1, labels2, alpha=0)
            # scheduler.step(loss)
            loss.backward()
            optimizer.step()
            dce = get_dice2(outputs1, labels2)
            epoch_loss += loss.item()
            epoch_loss1 += loss.item()
            epoch_loss2 += loss.item()
            #epoch_loss3 += loss3.item()

        if (epoch+1) % 5 == 0 or (epoch+1) == num_epochs:

            print('-' * 10)
            print("epoch %d loss:%f , dice1: %f, dice2: %f" % (epoch+1, epoch_loss/step, dce, dce))
            print("loss1:%f , loss2: %f, loss3: %f, loss4: %f"
                  % (epoch_loss1 / step, epoch_loss2 / step, 0, epoch_loss4 / step))
            state = {'net1': model1.state_dict(),
                     'optim': optimizer.state_dict(), 'epoch': epoch+1}
            #print("path",model_save_path)
            # if epoch>29:
            #     model_save_path = str(args.model) + "_" + "epoch_" + str(epoch) + "_" + "a1_" + str(
            #         weight_alpha) + "a2_" + str(dist_alpha) + base_model_save_path
            #     torch.save(state, model_save_path)
            if epoch>29:
                if loss < min:
                    min = loss
                    print("min:",min)
                    model_save_path = str(model)+"_epoch_" + str(epoch+1) + "_1.0combine"+"_" +"data_gall"+str(data_index) + '.pth'
                    torch.save(state, model_save_path)

        # writer.add_scalar('train/loss', epoch_loss/step, epoch)
        # writer.add_scalar('train/loss1', epoch_loss1 / step, epoch)
        # writer.add_scalar('train/loss2', epoch_loss2 / step, epoch)
        # writer.add_scalar('train/loss3', epoch_loss3 / step, epoch)
        # writer.add_scalar('train/loss4', epoch_loss4 / step, epoch)



def train(args):
    weight_alpha = args.alpha1
    dist_alpha=args.alpha2
    print(args.model)
    if args.model == "Unet":
        model1 = Unet().to(device)
    if args.model == "UNet_2Plus":
        model1 = UNet_2Plus().to(device)
    if args.model == "deeplabv3+":
        model1 = DeepLab(num_classes=1,backbone="resnet").to(device)
    if args.model == "deeplabv3":
        model1 = DeepLabV3(class_num=1).to(device)
    if args.model == "UNet3Plus":
        model1 = UNet3Plus(n_channels=1, n_classes=1).to(device)
    if args.model == "AttU_Net":
        model1 = AttU_Net(img_ch=1, output_ch=1).to(device)



    optimizer = optim.Adam([{'params': model1.parameters()}],lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=5, verbose=True,
                                                           min_lr=0.00000001)
    if args.Continue is True:                     #####判断是否继续训练
        model_save_path = "epoch_" + str(100) + "_" + "a1_" + str(weight_alpha) +"a2_"+str(dist_alpha)+ str(args.ckpt)
        if os.path.exists(model_save_path):
            traindata = txt_read('train_epoch100.txt')
            checkpoint = torch.load(model_save_path)                # 加载之前已经训练过的模型参数
            model1.load_state_dict(checkpoint['net1'])
            optimizer.load_state_dict(checkpoint['optim'])
            start_epoch = checkpoint['epoch']
            print('-----Read Checkpoint Success-----')
    else:
        start_epoch = 0
        # train_path = "./train_epoch" + str(args.epochs) + ".txt"
        if args.data == 100:
            traindata = txt_read('train_epoch100.txt')
        if args.data == 140:
            traindata = txt_read('train_epoch140.txt')
        if args.data == 150:
            traindata = txt_read('train_epoch150.txt')
        if args.data == 130:
            traindata = txt_read('sim_train_epoch130.txt')
        if args.data == 160:
            traindata = txt_read('sim_train_epoch160.txt')
        # if os.path.exists(train_path):
        #     traindata = txt_read(train_path)
        # else:
        #     print("--------making dataset--------")
        #     data = random_dataset(data_path)
        #     random.shuffle(data)
        #     testdata = data[:len(data) // 10]
        #     traindata = data[len(data) // 10:]
        #     txt_save('train_epoch%d.txt'%args.epochs, traindata) #############固定用同一个数据集 #./data\train\image\3\25.bmp ./data\train\mask\3\25.bmp ./data\train\mask\3\25mask.bmp
        #     txt_save('test_epoch%d.txt'%args.epochs, testdata)

    #traindata = make_dataset(os.path.join(data_path, 'train')) ######traindata包含全部数据路径的三元组
    pancreas_dataset = PolypDataset(traindata, transform=x_transforms, target_transform=y_transforms)#####对traindata读取数据，归一化和转为tensor，输出仍为三元组
    dataloaders = DataLoader(pancreas_dataset, batch_size=args.batch_size)
    get_param_num(model1)
    print('-----Wait For Training-----')
    #train_model(model1, model2, optimizer, dataloaders, args.epochs, args.ckpt, start_epoch,weight_alpha,dist_alpha)     ######################train model
    train_segnet(args, model1, optimizer, dataloaders, args.epochs, args.ckpt, start_epoch, args.data, weight_alpha,
            dist_alpha,args.model,scheduler)  ######################train model
    # writer.close()


def test(args):
    model1 = Unet()
    model2 = Unet(in_ch=1)
    temp_path = r".\performance\a1_0.3a2_0.04_epoch90dice72.91\a1_0.3a2_0.04_epoch90dice72.91\epoch_90_a1_0.3a2_0.04polyp.pth"
    checkpoint = torch.load(temp_path, map_location='cpu')
    model1.load_state_dict(checkpoint['net1'])
    model2.load_state_dict(checkpoint['net2'])
    print('-----Read Checkpoint Success-----')
    #testdata = make_dataset(os.path.join(data_path, 'test'))
    testdata = txt_read(r'.\test_epoch150.txt')              ###固定用同一个数据集十分之一测试 剩余训练
    pancreas_dataset = PolypDataset(testdata, transform=x_transforms, target_transform=y_transforms)
    dataloaders_test = DataLoader(pancreas_dataset, batch_size=1)
    model1.eval()
    model2.eval()
    num = 0
    average_dice = 0
    average_dice2 = 0
    average_scale = 0
    mkdir(compare_path)
    mkdir(compare_path2)
    with torch.no_grad():            # 不更新梯度
        for x, y, z in dataloaders_test:
            outputs1 = model1(x)
            mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=255)
            prediction1 = outputs1.sigmoid()
            prediction2 = model2(torch.mul(mask1,x)).sigmoid()
            x = torch.reshape(x, (1, x.shape[2], x.shape[3]))
            x_np = tensor2im(x)
            pred_y = torch.squeeze(prediction1).numpy()
            pred_z = torch.squeeze(prediction2).numpy()
            pred_y[pred_y > args.threshold] = 255
            pred_y[pred_y <= args.threshold] = 0
            pred_z[pred_z > args.threshold] = 255
            pred_z[pred_z <= args.threshold] = 0
            result = pred_y.astype(np.uint8)
            result = get_max_region(result)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join(pred_path, str(num) + '.bmp'), result)  #####写预测图像到pred_path文件夹

            result2 = pred_z.astype(np.uint8)
            result2 = get_max_region(result2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result2 = cv2.morphologyEx(result2, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join(pred_path, str(num) + 'mask.bmp'), result2)

            # 计算dice
            y = torch.squeeze(y).numpy()
            pred_img = result.copy()
            pred_img[pred_img > 0] = 1
            dice = get_dice(pred_img, y)
            average_dice += dice
            print('Gallbladder Dice of Number %s is %s' % (num, dice))

            z = torch.squeeze(z).numpy()
            pred_img2 = result2.copy()
            pred_img2[pred_img2 > 0] = 1
            dice2 = get_dice(pred_img2, z)
            average_dice2 += dice2
            scale = get_scale(pred_img2, z)
            average_scale += scale
            print('Polyp Dice of Number %s is %s' % (num, dice2))
            print('Polyp Scale of Number %s is %s' % (num, scale))

            # 画对比图
            ct_img = x_np.copy()
            mask_img = y.copy()  # 不复制会报错
            OverlayMaskOnCT(ct_img, mask_img, pred_img, num, compare_path)
            ct_img = x_np.copy()
            mask_img = z.copy()  # 不复制会报错
            OverlayMaskOnCT(ct_img, mask_img, pred_img2, num, compare_path2)

            num += 1
    print('Average dice is %s' % (average_dice / num))
    print('Average dice2 is %s' % (average_dice2 / num))
    print('Average scale is %s' % (average_scale / num))
    return average_dice/num


def seg_test(args):
    model1 = AttU_Net()
    #epoch_state = [50,60,70,80,90,100,110,120]
    epoch_state = [100]
    index_epoch = []
    gall_dice = []
    polyp_dice = []
    polyp_acc = []
    scale_index = []
    for index in epoch_state:
        temp_path = "./AttU_Net_epoch_" + str(index) + "_1.0combine"+"_" +"data_test"+str(100) + '.pth'
        #temp_path = r".\data\Unet++\test_pred_jyh_30_data100/"+ "epoch_30_a1_0.25a2_0.1polyp.pth"
        checkpoint = torch.load(temp_path, map_location='cpu')
        model1.load_state_dict(checkpoint['net1'])
        print('-----Read Checkpoint Success-----')
        # testdata = make_dataset(os.path.join(data_path, 'test'))
        testdata = txt_read(r'.\test_epoch100.txt')  ###
        pancreas_dataset = PolypDataset(testdata, transform=x_transforms, target_transform=y_transforms)
        dataloaders_test = DataLoader(pancreas_dataset, batch_size=1)
        model1.eval()
        num = 0
        average_dice = 0
        average_dice2 = 0
        average_scale = 0
        average_acc = 0

        mkdir(compare_path)
        mkdir(compare_path2)
        with torch.no_grad():  # 不更新梯度
            for x, y, z in dataloaders_test:
                outputs1 = model1(x)
                prediction1 = outputs1.sigmoid()
                print(x.size(),x)
                x = torch.reshape(x, (1, x.shape[2], x.shape[3]))
                x_np = tensor2im(x)
                pred_y = torch.squeeze(prediction1).numpy()
                pred_y[pred_y > args.threshold] = 255
                pred_y[pred_y <= args.threshold] = 0
                result = pred_y.astype(np.uint8)
                result = get_max_region(result)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite(os.path.join(pred_path, str(num) + '.jpg'), result)  #####写预测图像到pred_path文件夹

                y = torch.squeeze(y).numpy()
                z = torch.squeeze(z).numpy()
                pred_img2 = result.copy()
                pred_img2[pred_img2 > 0.3] = 1
                pred_img2[pred_img2 <= 0.3] = 0
                dice2 = get_dice(pred_img2, z)
                average_dice2 += dice2
                scale = get_scale(pred_img2,z)
                average_scale += scale
                acc = accuracy(pred_img2,z)
                average_acc += acc

                print('Polyp Dice of Number %s is %s' % (num, dice2))
                # print('Polyp Scale of Number %s is %s' % (num, scale))
                #print('Polyp accuracy of Number %s is %s' % (num, acc))
                # 画对比图
                ct_img = x_np.copy()
                gallmask = y.copy()  # 不复制会报错
                polypmask = z.copy()  # 不复制会报错
                #OverlayMaskOnCT(ct_img, mask_img, pred_img2, num, compare_path2)
                OverlayMaskOnCT_nocascade(ct_img, gallmask, polypmask, pred_img2, num, compare_path)


                num += 1
            index_epoch.append(index)
            gall_dice.append(average_dice2 / num)
            polyp_dice.append(average_dice2 / num)
            scale_index.append(average_scale / num)
            polyp_acc.append(average_acc / num)
            print('Average dice is %s' % (average_dice / num))
            print('Average dice2 is %s' % (average_dice2 / num))
        print('epoch:', index_epoch)
        print("gall_dice:", [round(i, 4) for i in gall_dice])
        print("polyp_dice:,", [round(i, 4) for i in polyp_dice])
        print("scale_dice:,", [round(i, 2) for i in scale_index])
        print("accuracy:,", [round(i, 5) for i in polyp_acc])
    return average_dice / num



def class_train(args):
    model1 = Unet()
    model2 = Unet(in_ch=1)
    model3 = resnet18().to(device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam([{'params': model3.parameters()}],lr=args.learning_rate)
    start_epoch = 0
    num_epochs = args.class_num
    model_save_path = "./class_"+str(num_epochs)+"epochs.pth"

    if args.Continue is True:                     #####判断是否继续训练
        if os.path.exists(model_save_path):
            checkpoint2 = torch.load(model_save_path, map_location='cpu')
            model3.load_state_dict(checkpoint2['net3'])
            optimizer.load_state_dict(checkpoint2['optim'])
            start_epoch = checkpoint2['epoch']

    print('-----Wait For Training-----')
    temp_path = r".\epoch_150_a_0.15polyp150data.pth"
    checkpoint = torch.load(temp_path, map_location='cpu')
    model1.load_state_dict(checkpoint['net1'])
    model2.load_state_dict(checkpoint['net2'])
    print('-----Read Checkpoint Success ready to segment polyp-----')
    # testdata = make_dataset(os.path.join(data_path, 'test'))
    traindata = txt_read(r'.\train_epoch150.txt')  ###固定用同一个数据集十分之一测试 剩余训练
    pancreas_dataset = CatDataset(traindata, transform=x_transforms, target_transform=y_transforms)
    dataloaders_train = DataLoader(pancreas_dataset, batch_size=2,shuffle=True)
    model1.eval()
    model2.eval()
    for e in range(num_epochs):
        epoch = e + start_epoch
        epoch_loss = 0
        step = 0
        for x, y, z in dataloaders_train:
            step += 1
            with torch.no_grad():  # 不更新梯度
                outputs1 = model1(x)
                prediction1 = outputs1.sigmoid()
                mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=255)
                prediction2 = model2(torch.mul(mask1, x)).sigmoid()
                prediction2 = thres_process(prediction2, args.threshold, max=255)
            x=x.to(device)
            y = y.to(device)
            prediction1 = prediction1.to(device)
            prediction2 = prediction2.to(device)

            #class_input = torch.cat([x, torch.mul(prediction2, x)], dim=1)   ###not using torch.mul since cannot guantante right segmentation
            class_input = torch.mul(prediction2, x)
            optimizer.zero_grad()
            output = model3(class_input)
            output = torch.nn.Softmax(dim=1)(output)
            # print("out:",output,"label:",y)
            loss = criterion(output,y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            print('-' * 10)
            print("epoch %d loss:%f " % (epoch + 1, epoch_loss / step))
            state = {'net3': model3.state_dict(),'optim': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, model_save_path)



def class_test(args):
    model1 = Unet()
    model2 = Unet(in_ch=1)
    model3 = resnet18()

    num_epochs = args.class_num
    model_save_path = "./class_" + str(num_epochs) + "epochs.pth"
    print('-----Wait For Testing-----')

    temp_path = r".\epoch_150_a_0.15polyp150data.pth"
    checkpoint = torch.load(temp_path, map_location='cpu')
    model1.load_state_dict(checkpoint['net1'])
    model2.load_state_dict(checkpoint['net2'])

    checkpoint2 = torch.load(model_save_path,map_location='cpu')
    model3.load_state_dict(checkpoint2['net3'])
    print('-----Read Checkpoint Success ready to segment polyp-----')

    # testdata = make_dataset(os.path.join(data_path, 'test'))
    testdata = txt_read(r'.\test_epoch150.txt')  ###固定用同一个数据集十分之一测试 剩余训练
    pancreas_dataset = CatDataset(testdata, transform=x_transforms, target_transform=y_transforms)
    dataloaders_test = DataLoader(pancreas_dataset, batch_size=1)
    model1.eval()
    model2.eval()
    model3.eval()
    total = 0
    right = 0
    with torch.no_grad():
        for x, y,z in dataloaders_test:
            outputs1 = model1(x)
            prediction1 = outputs1.sigmoid()
            mask1 = thres_process(outputs1.sigmoid(), args.threshold, max=255)
            prediction2 = model2(torch.mul(mask1, x)).sigmoid()
            prediction2 = thres_process(prediction2, args.threshold, max=255)
            #print("pred",prediction2)

            #class_input = torch.cat([x, torch.mul(prediction2, x)], dim=1)   ###not using torch.mul since cannot guantante right segmentation
            class_input = torch.mul(prediction2, x)
            output = model3(class_input)
            output = torch.nn.Softmax(dim=1)(output)
            total +=1
            print(output)
            if output[0][0]>output[0][1]:
                if int(y[0][0])==1:
                    right +=1
            else:
                if int(y[0][1])==1:
                    right +=1
            accuracy = right/total
            print("current label is",y,"model predict label is",output)

    print("total accuracy:",accuracy)



def class_solo_train(args):
    model3 = resnet18().to(device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam([{'params': model3.parameters()}],lr=args.learning_rate)
    start_epoch = 0
    num_epochs = args.class_num
    model_save_path = "./class_"+str(num_epochs)+"epochs.pth"

    if args.Continue is True:                     #####判断是否继续训练
        if os.path.exists(model_save_path):
            checkpoint2 = torch.load(model_save_path, map_location='cpu')
            model3.load_state_dict(checkpoint2['net3'])
            optimizer.load_state_dict(checkpoint2['optim'])
            start_epoch = checkpoint2['epoch']


    print('-----Wait For Training-----')

    # testdata = make_dataset(os.path.join(data_path, 'test'))
    traindata = txt_read(r'.\train_epoch100.txt')  ###固定用同一个数据集十分之一测试 剩余训练
    pancreas_dataset = CatDataset(traindata, transform=x_transforms, target_transform=y_transforms)
    dataloaders_train = DataLoader(pancreas_dataset, batch_size=2,shuffle=True)
    for e in range(num_epochs):
        epoch = e + start_epoch
        epoch_loss = 0
        step = 0
        for x, y,z in dataloaders_train:
            step += 1
            x=x.to(device)
            z=z.to(device)

            y = y.to(device)

            #class_input = torch.cat([x, torch.mul(prediction2, x)], dim=1)   ###not using torch.mul since cannot guantante right segmentation
            class_input = torch.mul(z, x)
            optimizer.zero_grad()
            output = model3(class_input)
            output = torch.nn.Softmax(dim=1)(output)
            # print("out:",output,"label:",y)
            loss = criterion(output,y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            print('-' * 10)
            print("epoch %d loss:%f " % (epoch + 1, epoch_loss / step))
            state = {'net3': model3.state_dict(),'optim': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, model_save_path)

def class_solo_test(args):
    model3 = resnet18()

    num_epochs = args.class_num
    model_save_path = "./class_" + str(num_epochs) + "epochs.pth"
    print('-----Wait For Testing-----')

    checkpoint2 = torch.load(model_save_path,map_location='cpu')
    model3.load_state_dict(checkpoint2['net3'])
    print('-----Read Checkpoint Success ready to segment polyp-----')

    # testdata = make_dataset(os.path.join(data_path, 'test'))
    testdata = txt_read(r'.\test_epoch100.txt')  ###固定用同一个数据集十分之一测试 剩余训练
    pancreas_dataset = CatDataset(testdata, transform=x_transforms, target_transform=y_transforms)
    dataloaders_test = DataLoader(pancreas_dataset, batch_size=1)
    model3.eval()
    total = 0
    right = 0
    with torch.no_grad():
        for x, y, z in dataloaders_test:
            #class_input = torch.cat([x, torch.mul(prediction2, x)], dim=1)   ###not using torch.mul since cannot guantante right segmentation
            class_input = torch.mul(z, x)
            output = model3(class_input)
            output = torch.nn.Softmax(dim=1)(output)
            total +=1
            print(output)
            if output[0][0]>output[0][1]:
                if int(y[0][0])==1:
                    right +=1
            else:
                if int(y[0][1])==1:
                    right +=1
            accuracy = right/total
            print("current label is",y,"model predict label is",output)

    print("total accuracy:",accuracy)


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, default=test,help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--learning_rate", type=float, default=1e-4)
    parse.add_argument("--epochs", type=int, default=100)
    parse.add_argument("--threshold", type=float, default=0.3)
    parse.add_argument("--ckpt", type=str, default='polyp.pth', help="the path of model weight file")
    parse.add_argument('--Continue', action='store_true', default=False, help='continue training')
    parse.add_argument('--alpha1', type=float, default=0.25)
    parse.add_argument('--alpha2', type=float, default=0.1)
    parse.add_argument('--class_num', type=int, default=9)
    parse.add_argument("--data", type=int, default=100)
    parse.add_argument('--model', type=str, default=Unet)  ##Unet() UNet_2Plus()
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    if args.action == "test":
        test(args)
    if args.action == "seg_test":
        seg_test(args)
    if args.action == "class_train":
        class_train(args)
    if args.action == "class_test":
        class_test(args)
    if args.action == "class_solo_train":
        class_solo_train(args)
    if args.action == "class_solo_test":
        class_solo_test(args)