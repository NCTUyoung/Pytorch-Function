from __future__ import print_function,division
import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
class BDD_Dataset(Dataset):
    """BDD Segmentation Dataset For Pytorch Training"""
    def __init__(self,image_list,label_list,image_root_dir,label_root_dir,transform=None):
        """
        Args:
            image_list (List):List contain all filename,should be sorted first.
            label_list (List):List contain all Label,should be sorted first.
            root_dir (string):Diretory with all the image.
            transform (callable,optional):Optional transform to be applied on a sample
        """
        self.image_list = image_list
        self.label_list = label_list
        self.image_root_dir = image_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        image = io.imread(os.path.join(self.image_root_dir,img_name))
        label = io.imread(os.path.join(self.label_root_dir,label_name))
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarry in sample to Tensors"""
    def __call__(self,sample):
        image,label = sample['image'],sample['label']
        image = (image.transpose((2,0,1))/255.0).astype(np.float32)
        return {'image':torch.from_numpy(image),'label':torch.from_numpy(label)}
def crop_center_numpy(img,cropx=704,cropy=1280):
    
    if len(img.shape)==3 :
        y,x,c = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2
        return img[starty:starty+cropy, startx:startx+cropx, :]
    else:
        y,x = img.shape[0],img.shape[1]
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2
        return img[starty:starty+cropy, startx:startx+cropx]

class CenterCrop(object):
    """Crop the center of image"""
    def __call__(self,sample):
        image,label = sample['image'],sample['label']
        image = crop_center_numpy(image,1280,704)
        label = crop_center_numpy(label,1280,704)
        return {'image':image,'label':label}
def show_sample(image,figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_batch(sample_batched,figsize =(15,15)):
    """Show image with label for a batch of sample"""
    images_batch,labels_batch = sample_batched['image'],sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.shape[2]
    
    grid_img = utils.make_grid(images_batch)
    grid_label = utils.make_grid(labels_batch.reshape(-1,1,labels_batch.shape[1],labels_batch.shape[2]))
    
    
    show_sample(grid_img.numpy().transpose(1,2,0),figsize=figsize)
    show_sample(grid_label.numpy().transpose(1,2,0)[:,:,0],figsize=figsize)  