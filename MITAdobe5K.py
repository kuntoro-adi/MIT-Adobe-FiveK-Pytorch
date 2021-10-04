import os
import rawpy
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import albumentations as albu
import albumentations.pytorch.transforms as t_albu

DATASET_DIR = '../5K-jpgs'
TRAIN_PATH = DATASET_DIR + '/train/'
TEST_PATH = DATASET_DIR + '/test/'
VAL_PATH = DATASET_DIR + '/val/'

train_files = os.listdir(TRAIN_PATH)
val_files = os.listdir(VAL_PATH)
test_files = os.listdir(TEST_PATH)

class MITAdobe5K(Dataset):
    
    def __init__(self, split = 'train', transform = None):
        assert(split == 'train' or split == 'val' or split == 'test')
        self.split = split
        self.data = train_files if split == 'train' else (val_files if split == 'val' else test_files)
        self.path = TRAIN_PATH if split == 'train' else (VAL_PATH if split == 'val' else TEST_PATH)
        self.transform = transform
        if transform == None:
            self.transform = albu.Compose([
                albu.Resize(512,512),
                albu.Normalize(mean=0., std=1.),
                t_albu.ToTensorV2()
            ])
        
    def __getitem__(self, idx):
        img = cv2.imread(self.path + self.data[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform != None:
            img = self.transform(image=img)['image']
        return img
    
    def __len__(self):
        return self.count
