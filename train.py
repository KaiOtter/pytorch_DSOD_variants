from __future__ import division
from utils.augmentations import SSDAugmentation
from models.demo.DSOD_6416 import *
from layers.modules import MultiBoxLoss
from data import *
import torch.nn as nn
import collections
import os
import sys
import time
import datetime
import argparse
import torch.nn.init as init
import torch
import torch.nn.init as init
import argparse
from torch.autograd import Variable
from shutil import copyfile
import torch.backends.cudnn as cudnn
import torch.optim as optim
from shutil import copyfile

import copy
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_set', type=str, default='0, 50', help='start and end number of epochs')
parser.add_argument('--voc_root', type=str, default='Pascal_Voc\VOCdevkit', help='path to dataset')
parser.add_argument('--voc_set', type=str, default='2007,2012', help='')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--resume', type=str, default='weights/xxx.pth', help='trained weights, use None for new')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_dir', type=str, default='weights/abc', help='path to save checkpoints')
parser.add_argument('--log', type=str, default='log/abc.txt', help='path where to save the loss log')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--imsize', type=int, default=320, help='the size of input')
parser.add_argument('--fine_tune', type=bool, default=False, help='Must edit tune_list when True')
parser.add_argument('--optim', type=str, default='Adam', help='SGD, Adam')
parser.add_argument('--init', type=bool, default=False, help='Whether init params in fine_tune layers')
opt = parser.parse_args()

cuda = torch.cuda.is_available() and opt.cuda
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def check_tune(name, list):
    r = False
    for l in list:
        if name.count(l) > 0:
            r = True
            break
    return r


def train():
    # # Initiate model
    num_classes = 21
    model = DSOD_64_16(num_classes)
    fine_tune_param = []

    if opt.resume is not None:
        # model.load_weights(opt.resume)
        print('Resume training. Loading weights from %s' % opt.resume)
        if opt.fine_tune:
            # collections.OrderedDict.
            dict_from = torch.load(opt.resume)
            dict_to = model.state_dict()
            #  'RBF', 'normalize', 'trans1', 'loc_layers', 'cls_layers'
            tune_list = ['SEblock']
            for name, param in model.named_parameters():
                if not opt.init:
                    print(name, "Copy weights of tune layers from saved pth")
                    dict_to[name] = dict_from.get(name)
                if not check_tune(name, tune_list):
                    if opt.init:
                        print(name, "Copy weights for frozen layers.")
                        dict_to[name] = dict_from.get(name)
                    param.requires_grad = False
            model.load_state_dict(dict_to)
            print("In fine_tune mode, it will do with :")
            for name, param in model.named_parameters():
                if check_tune(name, tune_list):
                    # print(name)
                    if name.count('normalize') == 0:
                        if opt.init:
                            print(name, 'Init the weights of layers for fine tuning.')
                            if name.count('weight') > 0:
                                if name.count('cls_layers') > 0 or name.count('loc_layers') > 0:
                                    nn.init.normal(param, std=0.01)
                                elif len(param.shape) >= 2:
                                    nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
                                else:
                                    nn.init.normal_(param, std=0.01)
                        param.requires_grad = True
                        fine_tune_param.append(param)
                    else:
                        nn.init.constant_(param, 20.0)
                        param.requires_grad = False
                if name == "normalize.0.weight":
                    print(param)
                    pass
        else:
            model.load_state_dict(torch.load(opt.resume), strict=True)
            for name, param in model.named_parameters():
                if name.count('normalize') == 0:
                    param.requires_grad = True
                    fine_tune_param.append(param)
                else:
                    nn.init.constant_(param, 20.0)
                    print(name)
                    print(param)
                    param.requires_grad = False
    else:
        fine_tune_param = model.parameters()

    if cuda:
        model = model.cuda()
    model.train()
    years = opt.voc_set.split(',')
    img_sets = []
    for y in years:
        if y == '2007':
            img_sets.append(('2007', 'trainval'))
        elif y == '2012':
            img_sets.append(('2012', 'trainval'))

    # (104, 117, 123)
    # (127, 127, 127)
    dataset = VOCDetection(root=opt.voc_root,
                           image_sets=img_sets,
                           transform=SSDAugmentation(opt.imsize, (127, 127, 127))
                           )
    dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size, num_workers=opt.n_cpu,
                                             collate_fn=detection_collate,
                                             shuffle=True, pin_memory=True)
    print('Loading the dataset...')
    epoch_size = len(dataloader)
    print("dataset size : {},  epoch_size: {}".format(len(dataset), epoch_size))

    if opt.optim == "SGD":
        optimizer = optim.SGD(fine_tune_param, lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    elif opt.optim == 'Adam':
        optimizer = optim.Adam(fine_tune_param, lr=opt.lr)
    else:
        print("Wrong optimizer name")
        exit(0)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, use_gpu=cuda)

    start_epoch, end_epoch = opt.epoch_set.split(',')
    start_epoch, end_epoch = int(start_epoch), int(end_epoch)
    save_interval = 1
    save_num_max = 3
    best_loss = 1e3
    prev_best = None
    saved_names = []

    save_dir, save_title = os.path.split(opt.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    log_dir, log_file = os.path.split(opt.log)
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(start_epoch, end_epoch+1):
        loc_loss = []
        conf_loss = []
        t0 = time.time()
        for batch_i, (imgs, targets) in enumerate(dataloader):
            tb_0 = time.time()
            if opt.cuda:
                images = imgs.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(imgs)
                targets = [Variable(ann) for ann in targets]

            out = model(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss.append(loss_l.item())
            conf_loss.append(loss_c.item())
            tb_1 = time.time()
            s = ('[Epoch %d/%d, Batch %d/%d] [Loc: %.4f, conf: %.4f, loss: %.4f] time cost: %.3f'
             % (epoch, end_epoch, batch_i + 1, len(dataloader),
                loss_l.data, loss_c.data, loss.data, (tb_1 - tb_0)))
            print(s)
            if batch_i % int(epoch_size/10) == 0:
                with open(os.path.join(log_dir, log_file), 'a') as f:
                    f.write(s + '\n')

        # done one epoch, check for saving
        epoch_loc = sum(loc_loss)/len(loc_loss)
        epoch_conf = sum(conf_loss)/len(conf_loss)
        epoch_loss = epoch_loc + epoch_conf
        learn_rate = optimizer.param_groups[0]['lr']
        print('avg loc: %.4f, avg conf: %.4f avg loss: %.4f, lr: %f' % (epoch_loc, epoch_conf, epoch_loss, learn_rate))
        t1 = time.time()
        print('epoch time cost: %.2f' % (t1 - t0))
        title, ext = os.path.splitext(log_file)
        if ext != '.txt':
            log_file = log_file + '.txt'
        with open(os.path.join(log_dir, log_file), 'a') as f:
            s = 'sum: {}|loss:{}|loc:{}|conf:{}|lr:{}\n'.format(epoch, epoch_loss, epoch_loc, epoch_conf, learn_rate)
            f.write(s)
        if epoch % save_interval == 0:
            print('Saving state, epoch: ', epoch)
            temp = "{}_{}_loss_{}.pth".format(save_title, epoch,
                                              str(round(epoch_loss, 4)))
            temp = os.path.join(save_dir, temp)
            saved_names.append(temp)
            torch.save(model.state_dict(), temp)
            # model.save_weights(temp)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                dest = temp.replace('loss', 'best')
                copyfile(temp, dest)
                if prev_best is not None:
                    os.remove(prev_best)
                prev_best = dest
            if len(saved_names) > save_num_max:
                del_save = saved_names[0]
                os.remove(del_save)
                saved_names.pop(0)


if __name__ == "__main__":
    train()