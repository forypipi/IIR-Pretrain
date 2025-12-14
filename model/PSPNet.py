import os
import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from model.PPM import PPM
from pytorch_optimizer import ScheduleFreeAdamW


class OneModel(nn.Module):
    def __init__(self, args, mode=None):
        super(OneModel, self).__init__()

        self.layers = args.layers
        self.dataset = args.data_set
        self.mode = mode
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.cls_bce = nn.CrossEntropyLoss()
        self.cls_kl = nn.KLDivLoss(reduction='batchmean')

        self.pretrained = True
        if self.dataset=='pascal':
            self.classes = 16
        elif self.dataset=='coco':
            self.classes = 61
        
        assert self.layers in [50, 101, 152]
    
        print('INFO: Using ResNet {}'.format(self.layers))

        if self.layers == 50:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif self.layers == 101:
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            self.teacher = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool, self.fc = resnet.avgpool, resnet.fc
        

        # Base Learner
        self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
        fea_dim = 2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

        self.T = 3

    def get_optim(self, args, LR):

        if self.mode == 'seg':    
            optimizer = ScheduleFreeAdamW(
                    [
                    {'params': self.encoder.parameters()},
                    {'params': self.ppm.parameters()},
                    {'params': self.cls.parameters()},
                    ], lr=LR, weight_decay=args.weight_decay)
        elif self.mode == 'cls':
            optimizer = ScheduleFreeAdamW([
                    {'params': self.fc.parameters()},
                    ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)

        return optimizer

    def forward(self, x, y, binary_y: torch.Tensor):

        self.teacher.eval()

        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        if self.mode == 'seg':

            x_cls_tmp = self.encoder(x)

            with torch.no_grad():

                binary_y = binary_y.unsqueeze(dim=1)

                binary_y = F.interpolate(binary_y.float(), size=x_cls_tmp.shape[-2:], mode='nearest')
                x_cls = x_cls_tmp * binary_y
                x_cls:torch.Tensor = self.avgpool(x_cls)
                x_cls = x_cls.view(x_size[0], -1)
                x_cls = self.fc(x_cls)
                x_cls_arg = x_cls.argmax(dim=1)

                x_teacher = self.teacher.conv1(x)
                x_teacher = self.teacher.bn1(x_teacher)
                x_teacher = self.teacher.relu(x_teacher)
                x_teacher = self.teacher.maxpool(x_teacher)
                x_teacher = self.teacher.layer1(x_teacher)
                x_teacher = self.teacher.layer2(x_teacher)
                x_teacher = self.teacher.layer3(x_teacher)
                x_teacher = self.teacher.layer4(x_teacher)

                x_teacher = x_teacher * binary_y
                x_teacher = self.teacher.avgpool(x_teacher)
                x_teacher = x_teacher.view(x_size[0], -1)                
                x_teacher = self.teacher.fc(x_teacher)
                y_cls = x_teacher.argmax(dim=1)

                cls_acc:torch.Tensor = x_cls_arg==y_cls
                cls_loss = self.cls_bce(x_cls, y_cls) + self.T**2*self.cls_kl(F.log_softmax(x_cls/self.T, dim=1), F.softmax(x_teacher/self.T, dim=1))
            
            x_seg = self.ppm(x_cls_tmp)
            x_seg = self.cls(x_seg)

        elif self.mode == 'cls':

            self.encoder.eval()
            self.ppm.eval()
            self.cls.eval()

            with torch.no_grad():
                x_cls_tmp = self.encoder(x)

            binary_y = F.interpolate(binary_y.float().unsqueeze(dim=1), size=x_cls_tmp.shape[-2:], mode='nearest')

            with torch.no_grad():

                x_seg = self.ppm(x_cls_tmp)
                x_seg = self.cls(x_seg)

                x_teacher = self.teacher.conv1(x)
                x_teacher = self.teacher.bn1(x_teacher)
                x_teacher = self.teacher.relu(x_teacher)
                x_teacher = self.teacher.maxpool(x_teacher)
                x_teacher = self.teacher.layer1(x_teacher)
                x_teacher = self.teacher.layer2(x_teacher)
                x_teacher = self.teacher.layer3(x_teacher)
                x_teacher = self.teacher.layer4(x_teacher)

                x_teacher = x_teacher * binary_y
                x_teacher = self.teacher.avgpool(x_teacher)
                x_teacher = x_teacher.view(x_size[0], -1)                
                x_teacher = self.teacher.fc(x_teacher)
                y_cls = x_teacher.argmax(dim=1)

            x_cls = x_cls_tmp * binary_y
            x_cls:torch.Tensor = self.avgpool(x_cls)
            x_cls = x_cls.view(x_size[0], -1)
            x_cls = self.fc(x_cls)
            x_cls_arg = x_cls.argmax(dim=1)

            cls_acc:torch.Tensor = x_cls_arg==y_cls
            cls_loss = self.cls_bce(x_cls, y_cls) + self.T**2*self.cls_kl(F.log_softmax(x_cls/self.T, dim=1), F.softmax(x_teacher/self.T, dim=1))

        x_seg = F.interpolate(x_seg, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            seg_loss = self.seg_criterion(x_seg, y.long())
            return x_seg.max(1)[1], seg_loss, cls_loss, cls_acc.to(dtype=torch.float)
        else:
            return x_seg, cls_loss, cls_acc.to(dtype=torch.float)

    def freeze_modules(self):
        for params in self.parameters():
            params.requires_grad = True

        if self.mode == 'seg':
            for module in [self.teacher, self.fc]:
                for params in module.parameters():
                    params.requires_grad = False
        elif self.mode == 'cls':
            for module in [self.teacher, self.encoder, self.ppm, self.cls]:
                for params in module.parameters():
                    params.requires_grad = False

    def get_state_dict_without_teacher(self):
        state_dict = self.state_dict()
        # 移除 self.teacher 的参数
        for key in list(state_dict.keys()):
            if 'teacher' in key:
                del state_dict[key]
        return state_dict