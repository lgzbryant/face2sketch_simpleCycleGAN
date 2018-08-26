import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
model=G(n_res=6)

model.load_state_dict(torch.load('G_a2b_190.pth'))                   
                        
img_resize=transforms.Resize((256,256),Image.BICUBIC)
img_randomcrop =transforms.RandomCrop(256,256)
img_to_tensor = transforms.ToTensor()
img_normlize=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))


img=Image.open('5.jpg')
tensor=img_resize(img) 
tensor=img_to_tensor(tensor)
tensor=img_normlize(tensor)

print("prediction:")    
tensor=tensor.resize_(1,3,256,256)    
real_A=Variable(tensor)
fake_B=model(real_A)

img_sample = torch.cat((real_A.data, fake_B.data), 0)
save_image(img_sample, 's.png' , nrow=1, normalize=True)        











             