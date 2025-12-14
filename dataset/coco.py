import json
import os
import os.path
from pathlib import Path
import cv2
import numpy as np
import copy

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


# -------------------------- Pre-Training --------------------------

class BaseData(Dataset):
    def __init__(self, 
                 split=3, 
                 mode=None, 
                 data_root=None, 
                 data_list=None, 
                 use_split_coco=False, 
                 transform=None, 
                 batch_size=None, 
                 **kwargs):

        assert mode in ['trn', 'val', 'test']

        self.num_classes = 80

        self.mode = mode
        self.split = split 
        self.data_root = data_root
        self.batch_size = batch_size

        if use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            self.class_list = list(range(1, 81))
            if self.split == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.split == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.split == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.split == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
        else:
            print('INFO: using COCO (PANet)')
            self.class_list = list(range(1, 81))
            if self.split == 3:
                self.sub_list = list(range(1, 61))
                self.sub_val_list = list(range(61, 81))
            elif self.split == 2:
                self.sub_list = list(range(1, 41)) + list(range(61, 81))
                self.sub_val_list = list(range(41, 61))
            elif self.split == 1:
                self.sub_list = list(range(1, 21)) + list(range(41, 81))
                self.sub_val_list = list(range(21, 41))
            elif self.split == 0:
                self.sub_list = list(range(21, 81)) 
                self.sub_val_list = list(range(1, 21)) 
            
        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        self.data_list = []  
        list_read = open(data_list).readlines()
        print("Processing data...")

        for line in tqdm(list_read):
            line = line.strip()
            line_split = line.split(' ')
            image_name = os.path.join(self.data_root, line_split[0])
            label_name = os.path.join(self.data_root, line_split[1])
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_tmp = label.copy()

        cls_label_list = np.unique(label_tmp)
        cls_label_list = np.delete(cls_label_list, np.where(cls_label_list == 0))
        if 255 in cls_label_list:
            cls_label_list = np.delete(cls_label_list, np.where(cls_label_list == 255))

        for cls in range(1, self.num_classes+1):
            select_pix = np.where(label_tmp == cls)
            if cls in self.sub_list:
                label[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
            else:
                label[select_pix[0],select_pix[1]] = 0
                
        raw_label = label.copy()

        if self.transform is not None:
            image, label = self.transform(image, label)

        binary_label = label.clone()
        valid_mask = (binary_label != 0) & (binary_label != 255)
        valid_values = binary_label[valid_mask].unique()
        if len(valid_values) > 0:
            random_idx = torch.randint(0, len(valid_values), (1,)).item()
            random_value = valid_values[random_idx]
            binary_label = (binary_label == random_value).to(torch.float32)
        else:
            binary_label = torch.zeros_like(binary_label, dtype=torch.float32)
            binary_label[binary_label == 255] = 0
            
        # Return
        if (self.mode== 'val' or self.mode== 'test') and self.batch_size == 1:
            return image, label, binary_label, raw_label
        else:
            return image, label, binary_label
        