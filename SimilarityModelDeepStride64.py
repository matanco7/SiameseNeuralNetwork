

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import date
import os
from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
from scipy import ndimage, misc



class SimilarityNetwork(nn.Module):    
    def __init__(self):
        Num_filters = [16,32,64,128,256,512] 
        Dropout_vec = [.1,.1,.1,0.05,0.05]  
        fc_out = 64
        self.Num_filters = Num_filters
        self.Dropout_vec = Dropout_vec
        
        super(SimilarityNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ZeroPad2d((6,6)),
            nn.Conv2d(1,Num_filters[0], kernel_size = 3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[0]),
            nn.Dropout2d(p=Dropout_vec[0]),
            )
            
        self.cnn2 =  nn.Sequential(nn.Conv2d(Num_filters[0], Num_filters[1], kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[1]),
            nn.Dropout2d(p=Dropout_vec[1]),
            )

        self.cnn3 = nn.Sequential(nn.Conv2d(Num_filters[1], Num_filters[2], kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[2]),
            nn.Dropout2d(p=Dropout_vec[2]),
        )

        self.cnn4 = nn.Sequential(nn.Conv2d(Num_filters[2], Num_filters[3], kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[3]),
            nn.Dropout2d(p=Dropout_vec[3]),
        )
        
        self.cnn5 = nn.Sequential(nn.Conv2d(Num_filters[3], Num_filters[4], kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[4]),
            nn.Dropout2d(p=Dropout_vec[3]),
        )
        
        
        self.cnn6 = nn.Sequential(nn.Conv2d(Num_filters[4], Num_filters[5], kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(Num_filters[5]),
            nn.Dropout2d(p=Dropout_vec[3]),
        )
        
       
        
        self.fc1 = nn.Sequential(
            nn.Linear(Num_filters[5]*16*16, fc_out),


        )
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = self.cnn3(output)
        output = self.cnn4(output)
        output = self.cnn5(output)
        output = self.cnn6(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, output_anchor, output_pos, output_neg):
        dis_pos = F.pairwise_distance(output_anchor,output_pos)
        dis_neg = F.pairwise_distance(output_anchor,output_neg)
        loss  = torch.mean(torch.maximum(dis_pos - dis_neg + self.margin,torch.tensor(0)))
        dis_pos_mean = torch.mean(dis_pos)
        dis_neg_mean = torch.mean(dis_neg)
        acc = torch.mean((dis_pos<dis_neg).float())
        return loss,dis_pos_mean,dis_neg_mean,acc
    
        
class SimilarityNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset    
        self.image_list = [i for i in os.listdir(self.imageFolderDataset) if not i[-5]=='r']
        
        
    def Rotate_images(self,Im):
        Im = Im.astype('float32')
        Im_r = Im.copy()
        rotate_opt_vec = list(range(-8,-3)) + list(range(4,9)) 
        angle = random.choice(rotate_opt_vec)
        
        Im_r[:,:] = ndimage.rotate(Im[:,:], angle, reshape=False)
        
        return Im_r.astype('float16')



    def do_augment(self,im):
        augment = random.choice([0,1])
        if not augment:
            return im
        else:
            return self.Rotate_images(im)
           
    
    def __getitem__(self,index):
        Image_name =  self.image_list[index]
        Images = np.load(self.imageFolderDataset + '\\' + Image_name)
        flip = random.choice([0,1])
        if flip:
            Images = np.flip(Images,axis=2).copy()
        
        img_anchor = torch.from_numpy(self.do_augment(Images[0,:,:])[np.newaxis,...]).type(torch.FloatTensor)
        
        
        
        img_pos = torch.from_numpy(self.do_augment(Images[1,:,:])[np.newaxis,...]).type(torch.FloatTensor)
        img_neg = torch.from_numpy(self.do_augment(Images[2,:,:])[np.newaxis,...]).type(torch.FloatTensor)
        return img_anchor, img_pos, img_neg
    
    def __len__(self):
        return len(self.image_list)
    
    

