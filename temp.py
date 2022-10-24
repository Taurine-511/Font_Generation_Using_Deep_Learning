# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch as T
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
import datetime

if os.path.exists('/Users/taurine511/Desktop/abc32/.DS_Store'):
    os.remove('/Users/taurine511/Desktop/abc32/.DS_Store')


def Writer(nb_name):
    dt_now = datetime.datetime.now()
    exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
    return SummaryWriter("runs/"+nb_name+'___'+exp_time)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


def upload(writer,loss,epoch,mode):
    for i in loss.keys():
        writer.add_scalar(f'{mode}_{i}',loss[i],epoch)
        
def get_path(ind,ds,root,y_dim):
    return f'{root}/{ds[ind//y_dim]}/{ind%y_dim}.png'

class NormalDataset(Dataset):
    def __init__(self, root, y_dim,transform=torchvision.transforms.ToTensor()):
        super().__init__()

        self.y_dim = y_dim
        self.root = root
        self.ds = os.listdir(root)
        self.len = len(self.ds)*y_dim
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        path = get_path(index,self.ds,self.root,self.y_dim)
        image = np.array(Image.open(path)).astype(np.float32).transpose((2,0,1))[0]
        image /= 255 
        
        char = F.one_hot(T.tensor(index%self.y_dim),self.y_dim)
        image = self.transform(image)

        return image, char


class MyDataset(Dataset):
    def __init__(self, root, y_dim,category,transform=None):
        super().__init__()

        self.y_dim = y_dim
        self.root = root
        self.ds = os.listdir(root)
        self.len = len(self.ds)
        self.transform = transform
        self.category = category
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        path = get_path(26*index+self.category,self.ds,self.root,self.y_dim)
        image = np.array(Image.open(path)).astype(np.float32).transpose((2,0,1))[0]
        image /= 255 
        
        char = F.one_hot(T.tensor(self.category),self.y_dim)
        
        if self.transform:
            image = self.transform(image)
        return image, char

def get_colab_path(ind,ds,root,y_dim):
    font = ind//(y_dim*y_dim)
    rest = ind%(y_dim*y_dim)
    st = f'{root}/{ds[font]}/{rest%y_dim}.png'
    en = f'{root}/{ds[font]}/{rest//y_dim}.png'
    return [st,en]


class ColabDataset(Dataset):
    def __init__(self, root, y_dim,transform=None):
        super().__init__()

        self.y_dim = y_dim
        self.root = root
        self.ds = os.listdir(root)
        self.len = len(self.ds)*(y_dim**2)
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        label = [index%self.y_dim,index%(self.y_dim**2)//self.y_dim]
        path = get_colab_path(index,self.ds,self.root,self.y_dim)
        image_1 = self.trans(path[0])
        image_2 = self.trans(path[1])
        char_1 = F.one_hot(T.tensor(label[0]),self.y_dim)
        char_2 = F.one_hot(T.tensor(label[1]),self.y_dim)
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        return image_1,image_2, char_1, char_2
    
    def trans(self,path):
        image = np.array(Image.open(path)).astype(np.float32).transpose((2,0,1))[0]
        image /= 255
        return image
    
def get_rand_loader(root,y_dim,batch_size):
    dataset = ColabDataset(root,y_dim,torchvision.transforms.ToTensor())
    t = int(len(dataset)*0.9)

    train_dataset, test_dataset = T.utils.data.random_split(
        dataset,
        [t, len(dataset)-t]
    )

    train_loader = T.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = T.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
      

def get_loaders(root,y_dim,batch_size,Dataset=MyDataset):
    train_loaders,test_loaders = [], []
    for i in range(y_dim):
        dataset = Dataset(root,y_dim,i,torchvision.transforms.ToTensor())
        t = int(len(dataset)*0.9)
        
        train_dataset, test_dataset = T.utils.data.random_split(
            dataset,
            [t, len(dataset)-t]
        )
        
        train_loader = T.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
        )
        
        test_loader = T.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    return train_loaders, test_loaders

class Residual(nn.Module):
    def __init__(self,inc,h_dim,n_res_h):
        super(Residual,self).__init__()
        
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inc,n_res_h,3,1,1),
            nn.ReLU(),
            nn.Conv2d(n_res_h,h_dim,1,1),
        )


        
    def forward(self,x):
        h = self.layer(x)
        return x+h

class Res_stack(nn.Module):
    def __init__(self,inc,h_dim,n_res_h,n_layer):
        super(Res_stack,self).__init__()
        self.n_layer = n_layer
        self.layer = nn.Sequential(
            *[Residual(inc,h_dim,n_res_h) for _ in range(n_layer)],
            nn.ReLU()
        )
        
    def forward(self,x):
        return self.layer(x)
    
def block(in_dim,out_dim,activate=nn.ReLU(),norm=False):
    layer = [nn.Conv2d(in_dim,out_dim,4,2,1)]
    layer.append(activate)
    if norm:
        layer.append(nn.BatchNorm2d(out_dim))
    return layer


def block_t(in_dim,out_dim,activate=nn.ReLU(),norm=False):
    layer = [nn.ConvTranspose2d(in_dim,out_dim,4,2,1)]
    if norm:
        layer.append(nn.BatchNorm2d(out_dim))
    layer.append(activate)
    return layer

class Flatten(nn.Module):
    def forward(self, x):
        N, _, _, _ = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
