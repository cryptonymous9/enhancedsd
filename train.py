'''
Strict Changes-
+ Change the data directories (Read more in get_data function)
+ Change Hyperparameters and other configurations accordingly (line 37)
'''

import os
import sys
import argparse
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pathlib import Path
from torchsummary import summary

from data.datasets import CDataset
from enhancedsd.networks.esd import ESD_net
from enhancedsd.losses import pytorch_ssim


#torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Some Hyperparameters
samples_DIR = "./Samples_training/exp1/"         # Directory for saving samples during training
checkpoints_DIR = "./Model_checkpoints/exp1/"    # Directory for saving training checkpoints
BATCH_SIZE = 14              # Batch Size
EPOCHS = 500                 # Training Iterations
save_logs = True             # Save training history
LEARNING_RATE = 0.0002       # learning rate
BETA_1, BETA_2 = 0.9,0.999   # Adam coefficients

 
def get_data(batch_size=14):
    '''
    Data loader function
    
    + Modify DIR values in data/datasets.py to the file locations
    + Data needs to be in npy format
    + Dividing data into 3 sets is recommended (train, validation and test)
    
    '''
    train_loader = CDataset(mode='train', normalize=True)
    train_iterator = DataLoader(train_loader, batch_size=batch_size, pin_memory=True)

    val_loader = CDataset(mode='val') 
    val_iterator = DataLoader(val_loader, batch_size=batch_size, pin_memory=True)

    test_loader = CDataset(mode='test') 
    test_iterator = DataLoader(test_loader, batch_size=batch_size, pin_memory=True)
    print("Data loaded")
    return  train_iterator, val_iterator, test_iterator



class Trainer:
    '''
    Main Trainer Class
    
    + data_iterator:  Torch Iterator for training data
    + dir_checkpoint: Directory for saving training checkpoints 
    + dir_samples:    Directory for saving samples during training
    
    '''
    def __init__(self, data_iterator, dir_checkpoint, dir_samples):
        self.model = None
        self.data_iterator = data_iterator
    
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model = None 
        
        self.save_samples_DIR = dir_samples
        self.save_checkpoints_DIR = dir_checkpoint
        Path(self.save_samples_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.save_checkpoints_DIR).mkdir(parents=True, exist_ok=True)

    def validation(self, net, val_iterator):
        '''
        Validation Function
        
        + net: Model for validation
        + val_iterator: Validation data-iterator
        '''
        net.eval()
        with torch.no_grad():
            loss_pixel, loss_ssim = 0, 0
            for i, imgs in enumerate(val_iterator):
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))           
                
                gen_hr = net(imgs_lr)
                loss_pixel+= criterion_pixel(gen_hr, imgs_hr).item()
                loss_ssim += (1-ssim_loss(gen_hr, imgs_hr).item())
            
        n_batches = len(val_iterator)
        return loss_pixel/n_batches, loss_ssim/n_batches
            

    def save_logs(self, *logs):
        '''
        Saving logs. Saves the following
        + Train MSE loss
        + Train SSIM loss
        + Validation MSE loss
        + Validation SSIM loss
        '''
        log_dic = {'train_pixel':logs[0],
                    'train_ssim':logs[1],
                    'val_pixel':logs[2],
                    'val_ssim':logs[3]
        }
        return log_dic


    def train(self, model, n_epochs, log_batches='default', log_epochs=1, sample_interval=1, checkpoint_interval=5):
        '''
        Train Function
        + model (Torch model): Model to be trained
        + n_epochs    (int): No. of epochs for training
        + log_batches (int): Frequency of batch logs. Default value: 'default' -> Produces batch logs two times per epoch
        + log_epochs  (int): Frequency of epoch logs. Default value one
        + sample_interval (int):  Frequency of saving samples. Default once per epoch
        + checkpoint_interval (int): Frequency of saving checkpoints
        '''
        
        lr = LEARNING_RATE        
        b1, b2 = BETA_1, BETA_2  
        
        criterion_pixel = torch.nn.MSELoss().to(device)
        ssim_loss = pytorch_ssim.SSIM().to(device)

        train_iterator, val_iterator, test_iterator = self.data_iterator
        n_batches = len(train_iterator)


        if log_batches == 'default':
            log_batches = len(train_iterator)//2

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, min_lr=1e-6, verbose=True)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        list_avg_pixel_losses, list_avg_ssim = [], []
        val_pixel_list, val_ssim_list = [], []
        list_batch_losses = []
        print("starting training ---",end='\n')
        for epoch in range(n_epochs):
            
            avg_loss, avg_pixel_loss, avg_ssim_loss = 0, 0, 0
            
            model.train()
            for i, imgs in enumerate(train_iterator):

                batches_done = epoch * n_batches + i

                
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))
                                
                optimizer.zero_grad()
                
                gen_hr = model(imgs_lr)

                # Measure loss against ground truth
                loss_pixel = criterion_pixel(gen_hr, imgs_hr)
                loss_ssim = 1-ssim_loss(gen_hr, imgs_hr)
                joint_loss = loss_pixel + 0.001*loss_ssim
                joint_loss.backward()
                
                optimizer.step()

                avg_pixel_loss+=loss_pixel.item()
                avg_ssim_loss+=loss_ssim.item()
                avg_loss+=joint_loss.item()
                
                list_batch_losses.append(joint_loss.item())
                
                if batches_done % log_batches == 0:
                    print("\t [Epoch %d/%d] [Batch %d/%d] [Pixel loss: %f] [SSIM: %f] [Joint Loss: %f]"
                        % (
                            epoch,
                            n_epochs,
                            i,
                            n_batches,
                            loss_pixel.item(),
                            1 - loss_ssim.item(),
                            joint_loss.item()
                        )
                    )
            
            avg_pixel_loss = avg_pixel_loss/n_batches
            avg_ssim_loss = avg_ssim_loss/n_batches
            avg_loss = avg_loss/n_batches
            
            list_avg_pixel_losses.append(avg_pixel_loss)
            list_avg_ssim.append(avg_ssim_loss)
            
            val_pixel, val_ssim = self.validation(model, val_iterator)
            val_pixel_list.append(val_pixel)
            val_ssim_list.append(val_ssim)
            
            val_joint = val_pixel + 0.001*val_ssim

            if val_joint < self.best_loss:
                self.best_model = generator.state_dict()
                self.best_epoch = epoch
                print(f"Model improved... Best Loss: {val_joint} Previous Best: {best_loss}")
                self.best_loss = val_joint
            
            scheduler.step(val_joint)

            print("[Epoch %d/%d] [Pixel: %f,  Best Pixel: %f] [SSIM: %f] [Val Pixel: %f  Val SSIM:L %f]"
                    % (
                            epoch,
                            n_epochs,
                            avg_loss,
                            best_loss,
                            avg_ssim_loss,
                            val_pixel,
                            val_ssim
                        )
                    )
            
            if epoch % sample_interval == 0:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr[:4], scale_factor=4)
                img_grid = torch.cat((imgs_lr, gen_hr[:4], imgs_hr[:4]), -1)
                save_image(img_grid, self.save_samples_DIR + f"ep_{epoch}.png", nrow=1, normalize=True)

            if epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), self.save_checkpoints_DIR+f"ep_{epoch}.pth")
        
        return self.save_logs(list_avg_pixel_losses, list_avg_ssim, val_pixel_list, val_ssim_list)


if __name__ == "__main__":
    data_iter = get_data(batch_size=BATCH_SIZE) 
    trainer = Trainer(data_iterator=data_iter, dir_checkpoint=checkpoints_DIR, dir_samples=samples_DIR)
    net = ESD_net(channels=1,ICNR_initialization=True).cuda()
    logs = trainer.train(model=net, n_epochs=EPOCHS)
    print("Training finished")
    if save_logs == True:
        with open(checkpoints_DIR+'logs.pkl', 'wb') as f:
            pickle.dump(logs, f)
        print("\n logs saved ---",end='\n')