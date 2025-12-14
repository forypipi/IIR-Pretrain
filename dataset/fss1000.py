import glob
import json
import os
import os.path
from pathlib import Path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class BaseData(Dataset):
    def __init__(self, 
                 split='', 
                 mode=None, 
                 data_root=None, 
                 data_list=None, 
                 transform=None, 
                 batch_size=None, 
                 label_list=None, **kwargs):

        assert mode in ['trn', 'val', 'test']

        self.mode = mode
        self.split = split 
        self.data_root = data_root
        self.batch_size = batch_size

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

        if label_list:
            with open(label_list, 'r', encoding='utf-8') as file:
                # 逐行读取并去除行末的换行符
                self.label_list = [line.strip() for line in file]

        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        cls_name = os.path.basename(os.path.dirname(image_path))
        cls_index = self.label_list.index(cls_name)
        label[label == 1] = cls_index + 1

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