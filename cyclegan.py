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


parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch',type=int, default=0)
parser.add_argument('---total_epoch',type=int,default=200)
parser.add_argument('--dataset',type=str,default='CUHK')
parser.add_argument('--batchsize',type=int,default=1)
parser.add_argument('--lr',type=float,default=0.00002)
parser.add_argument('--b1',type=float,default=0.5)
parser.add_argument('--b2',type=float,default=0.999)
parser.add_argument('--decay_epoch',type=int,default=100)
parser.add_argument('--n_cpu',type=int,default=6)
parser.add_argument('--img_width',type=int,default=256)
parser.add_argument('--img_height',type=int,default=256)
parser.add_argument('--channels',type=int,default=3)
parser.add_argument('--sample_interval',type=int,default=100)
parser.add_argument('--checkpoint_interval',type=int,default=10)
parser.add_argument('--n_residual',type=int,default=6)

opt=parser.parse_args()
print(opt)

folder = os.path.exists('images/')
if not folder:
    os.makedirs('images/%s'%opt.dataset)
    
folder= os.path.exists('model/')
if not folder:
    os.makedirs('model/%s'%opt.dataset)


criterion_GAN=torch.nn.MSELoss()
criterion_cycle=torch.nn.L1Loss()
criterion_identity=torch.nn.L1Loss()

cuda =True if torch.cuda.is_available() else False

G_a2b=G(n_res=opt.n_residual)
G_b2a=G(n_res=opt.n_residual)
D_a=D()
D_b=D()

if cuda:
    G_a2b=G_a2b.cuda()
    G_b2a=G_b2a.cuda()
    D_a=D_a.cuda()
    D_b=D_b.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
  
if opt.start_epoch!=0:
    G_a2b.load_state_dict(torch.load('save/%s/G_a2b_%d.pth'%(opt.dataset,opt.epoch)))
    G_b2a.load_state_dict(torch.load('save/%s/G_b2a_%d.pth'%(opt.dataset,opt.epoch)))
    D_a.load_state_dict(torch.load('save/%s/D_a_%d.pth'%(opt.dataset,opt.epoch)))
    D_b.load_state_dict(torch.load('save/%s/D_b_%d.pth'%(opt.dataset,opt.epoch)))
else:
    G_a2b.apply(weight_init)
    G_b2a.apply(weight_init)
    D_a.apply(weight_init)
    D_b.apply(weight_init)

lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

lambda_cycle=10
lambda_id=0.5*lambda_cycle

optimizer_G_a2b=torch.optim.Adam(G_a2b.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_G_b2a=torch.optim.Adam(G_b2a.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_D_a=torch.optim.Adam(D_a.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_D_b=torch.optim.Adam(D_b.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))


lr_scheduler_G_a2b=torch.optim.lr_scheduler.LambdaLR(optimizer_G_a2b,\
                  lr_lambda=LambdaLR(opt.total_epoch,opt.start_epoch,opt.decay_epoch).step)
                  
lr_scheduler_G_b2a=torch.optim.lr_scheduler.LambdaLR(optimizer_G_b2a,\
                  lr_lambda=LambdaLR(opt.total_epoch,opt.start_epoch,opt.decay_epoch).step)
                  
lr_scheduler_D_a=torch.optim.lr_scheduler.LambdaLR(optimizer_D_a,\
                  lr_lambda=LambdaLR(opt.total_epoch,opt.start_epoch,opt.decay_epoch).step)
                  
lr_scheduler_D_b=torch.optim.lr_scheduler.LambdaLR(optimizer_D_b,\
                  lr_lambda=LambdaLR(opt.total_epoch,opt.start_epoch,opt.decay_epoch).step)
                 
                                                      
                 
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
     
                        
fake_A_pool=ImageBuffer()
fake_B_pool=ImageBuffer()

train_dataloader = DataLoader(ImageDataset("../../PyTorch-GAN/data/%s" \
% opt.dataset, transforms_=transforms_, unaligned=True),batch_size=opt.batchsize, shuffle=True, num_workers=opt.n_cpu)
val_dataloader = DataLoader(ImageDataset("../../PyTorch-GAN/data/%s" \
% opt.dataset, transforms_=transforms_, unaligned=True, mode='test'),batch_size=6, shuffle=True, num_workers=1)                            

def save_sample(now_batch):
    imgs=next(iter(val_dataloader))
    real_A=Variable(imgs['A'].type(Tensor))
    fake_B=G_a2b(real_A)
    real_B=Variable(imgs['B'].type(Tensor))
    fake_A=G_b2a(real_B)
    img_sample=torch.cat((real_A.data,fake_B.data,real_B.data,fake_A.data),0)
    save_image(img_sample,'images/%s/%s.jpg'%(opt.dataset,now_batch),nrow=6,normalize=True)
                        

start_time=time.time()
for epoch in range(opt.start_epoch,opt.total_epoch):
    for i, batch in enumerate(train_dataloader):    
        real_A=Variable(batch['A'].type(Tensor))
        real_B=Variable(batch['B'].type(Tensor))
        
        real_target=Variable(Tensor(real_A.size(0),*(1,opt.img_height//2**4,opt.img_width//2**4)).fill_(1.0),requires_grad=False)
        fake_target=Variable(Tensor(real_A.size(0),*(1,opt.img_height//2**4,opt.img_width//2**4)).fill_(0.0),requires_grad=False)
        
        # ------------------
        #  Train Generators
        # ------------------
        
        optimizer_G_a2b.zero_grad()
        optimizer_G_b2a.zero_grad()
        
        loss_id_A=criterion_identity(G_a2b(real_B),real_B)
        loss_id_B=criterion_identity(G_b2a(real_A),real_A)
        loss_id=(loss_id_A+loss_id_B)/2
        
        fake_B=G_a2b(real_A)
        fake_A=G_b2a(real_B)
        
        loss_GAN_a2b=criterion_GAN(D_b(fake_B),real_target)
        loss_GAN_b2a=criterion_GAN(D_a(fake_A),real_target)
        loss_GAN=(loss_GAN_a2b+loss_GAN_b2a)/2
        
        re_A=G_b2a(fake_B)
        re_B=G_a2b(fake_A)
        
        loss_cycle_a=criterion_cycle(re_A,real_A)
        loss_cycle_b=criterion_cycle(re_B,real_B)
        loss_cycle=(loss_cycle_a+loss_cycle_b)/2
        
        loss_G=loss_id*lambda_id+loss_cycle*lambda_cycle+loss_GAN
        
        loss_G.backward()
        optimizer_G_a2b.step()
        optimizer_G_b2a.step()
        
        
        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_a.zero_grad()
        loss_real=criterion_GAN(D_a(real_A),real_target)
        
        fake_A=fake_A_pool.push_and_pop(fake_A)
        loss_fake=criterion_GAN(D_a(fake_A.detach()),fake_target)
        
        loss_D_a=(loss_real+loss_fake)/2
        loss_D_a.backward()
        optimizer_D_a.step()
        
        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_b.zero_grad()
        loss_real=criterion_GAN(D_b(real_B),real_target)
        
        fake_B=fake_B_pool.push_and_pop(fake_B)
        loss_fake=criterion_GAN(D_b(fake_B.detach()),fake_target)
        
        loss_D_b=(loss_real+loss_fake)/2
        loss_D_b.backward()
        optimizer_D_b.step()
        
        loss_D=(loss_D_a+loss_D_b)/2
          
        now_batch=epoch*len(train_dataloader)+i 
        batchsize_remaining=opt.total_epoch*len(train_dataloader)-now_batch
        
        time_left=datetime.timedelta(seconds=batchsize_remaining*(time.time()-start_time))
        start_time=time.time()
        
        print('[epoch %d/%d][batch %d/%d][D_loss %f][G_loss %f, gan: %f, cycle: %f, identity: %f]time_left: %s'%\
                (epoch,opt.total_epoch,i,len(train_dataloader),loss_D.data[0], \
                loss_G.data[0],loss_GAN.data[0],loss_cycle.data[0],loss_id.data[0],time_left))
                
        if now_batch %opt.sample_interval==0:
            save_sample(now_batch)


    # Update learning rates
    lr_scheduler_G_a2b.step()
    lr_scheduler_G_b2a.step()
    lr_scheduler_D_a.step()
    lr_scheduler_D_b.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_a2b.state_dict(), 'model/%s/G_a2b_%d.pth' % (opt.dataset, epoch))
        torch.save(G_b2a.state_dict(), 'model/%s/G_b2a_%d.pth' % (opt.dataset, epoch))
        torch.save(D_a.state_dict(), 'model/%s/D_a_%d.pth' % (opt.dataset, epoch))
        torch.save(D_b.state_dict(), 'model/%s/D_b_%d.pth' % (opt.dataset, epoch))
