import os
import numpy as np
import itertools
from scipy.spatial.distance import cdist

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def color_jitter(x):
    x = np.array(x, 'int32')
    for ch in range(x.shape[-1]):
        x[:,:,ch] += np.random.randint(-2,2)
    x[x>255] = 255
    x[x<0]   = 0
    return x.astype('uint8')


# Edit this only if you need to manipulate dimensions. Also use the commented transforms.Resize inside the (self.lr_transform, self.hr_transform) if you do so.
hr_height, hr_width = 96, 236

#Processed_DIR = "./Output" 
#source_path = "./Sample/"


class CDataset(Dataset):
    '''
    Dataloader Function
    
    + mode: ('train', 'val', 'test') -> To choose the a particular set
    + normalize: (Bool) Data normalization. Recommended for training!
    
    Note: Change variables with data_path_[$$] name to the directories with data. 
        $$ lr -> low-resolution Climate projections
        $$ hr -> high-resolution Reanalysis
    '''
    def __init__(self, mode='train', normalize=True):
        self.preload = True
        
        #mean_data, std_data = np.array([0.5064, 0.4263, 0.3845]), np.array([0.1490, 0.1443, 0.1469])
        # transforms.Resize((hr_height // 4, hr_width // 4), transforms.InterpolationMode.BICUBIC), 
        self.lr_transform  = transforms.Compose(
            [transforms.ToTensor()]
            )
        self.hr_transform  = transforms.Compose(
            [transforms.ToTensor()]
            )

        if mode=='train':
            data_path_lr = "../../../Data/Processed/NP_acc_Aug/Coarse_adjusted/coarse_train.npy"
            data_path_hr = "../../../Data/Processed/NP_acc_Aug/Fine_adjusted/fine_train.npy"
        
        elif mode=='val':
            data_path_lr = "../../../Data/Processed/NP_acc_Aug/Coarse_adjusted/coarse_val.npy"
            data_path_hr = "../../../Data/Processed/NP_acc_Aug/Fine_adjusted/fine_val.npy"

        elif mode=='test':
            data_path_lr = "../../../Data/Processed/NP_acc_Aug/Coarse_adjusted/coarse_test.npy"
            data_path_hr = "../../../Data/Processed/NP_acc_Aug/Fine_adjusted/fine_test.npy"
        
        self.all_lr = np.load(data_path_lr, allow_pickle=True)
        self.all_hr = np.load(data_path_hr, allow_pickle=True)
        
        if normalize:
            max_lr = np.max(self.all_lr)
            self.all_lr = (self.all_lr/max_lr)
            
            max_hr = np.max(self.all_hr)
            self.all_hr = (self.all_hr/max_hr)
        
    def __getitem__(self, idx):
        img_lr = self.lr_transform(self.all_lr[idx])
        img_hr = self.hr_transform(self.all_hr[idx])

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        self.n_files = len(self.all_lr)
        return self.n_files