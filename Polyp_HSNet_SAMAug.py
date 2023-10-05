import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import HSNet_with_aux
from dataloader import get_loader_with_aux, test_dataset_with_aux
from utils import clip_gradient, adjust_lr, AvgMeter
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import logging
import pdb

seednumber=10
random.seed(seednumber)     # python random generator
np.random.seed(seednumber)  # numpy random generator

torch.manual_seed(seednumber)
torch.cuda.manual_seed_all(seednumber)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def structure_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def l1_loss(pred, mask):
    return (pred - mask).abs().mean()

def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images_og/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    aux_root = '{}/images_stabilityscore/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset_with_aux(image_root, gt_root, aux_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, aux,name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        aux=aux.cuda()
        res,res1,res2,res3,_,_,_,_ = model(image)
        res = F.upsample(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)      
        indicator1=np.mean(np.abs(res-0.5))
        Ares,Ares1,Ares2,Ares3,_,_,_,_ = model(image+aux)
        Ares = F.upsample(Ares + Ares1 + Ares2 + Ares3, size=gt.shape, mode='bilinear', align_corners=False)
        Ares = Ares.sigmoid().data.cpu().numpy().squeeze()
        Ares = (Ares - Ares.min()) / (Ares.max() - Ares.min() + 1e-8)
        indicator2=np.mean(np.abs(Ares-0.5))            
        if indicator1>indicator2:
            input = res
        else:
            input = Ares            
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75,1,1.25]
    loss_P2_record = AvgMeter()
    loss_P2_record_=AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            # ---- data prepare ----
            images, gts , aux = pack
            images = (images).cuda()
            gts = (gts).cuda()
            aux = (aux).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                aux = F.upsample(aux, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            optimizer.zero_grad()
            P1,P2,P3,P4,P1_,P2_,P3_,P4_= model(images)            
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4 
            # ---- backward ----
            loss.backward()
            optimizer.step()
                              
            optimizer.zero_grad()
            P1,P2,P3,P4,P1_,P2_,P3_,P4_= model(images+aux)            
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4 
            loss.backward()
            optimizer.step()
                        
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P4.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()), optimizer.param_groups[0]['lr'])
                         
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    global dict_plot

    test1path = '/media/yizhe/SSD/PraNet-master/PraNet-master/data/TestDataset/'
    if (epoch + 1) % 1 == 0:#'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

            
def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()


if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'startingwithpretrainedweights_seed1_61'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=10, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str, 
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_seed1_61.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = HSNet_with_aux().cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr,weight_decay=0)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20],gamma=0.2)
    image_root = '/media/yizhe/SSD/PraNet-master/PraNet-master/data/TrainDataset/images_og/'.format(opt.train_path)
    gt_root = '/media/yizhe/SSD/PraNet-master/PraNet-master/data/TrainDataset/masks/'.format(opt.train_path)
    aux_root = '/media/yizhe/SSD/PraNet-master/PraNet-master/data/TrainDataset/images_stabilityscore/'.format(opt.train_path)

    train_loader = get_loader_with_aux(image_root, gt_root, aux_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=False)
    total_step = len(train_loader)
    for epoch in range(1, opt.epoch):
         train(train_loader, model, optimizer, epoch, opt.test_path)
         scheduler.step()

