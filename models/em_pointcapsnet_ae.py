#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
@author: Chris Murray
"""

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(BASE_DIR, 'nndistance'))
#from modules.nnd import NNDModule
import torch_nndistance as NND
#distChamfer = NNDModule()
distChamfer = NND.nnd
USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor

# Implementation of the Latent Capsule Layer using EM routing instead of Routing by Agreement
# Only have a single layer here
class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64,num_iterations=1):
        super(EM_LatentCapsLayer, self).__init__()
        self.p_vec_size = prim_vec_size
        self.l_vec_size = latent_vec_size
        self.B = prim_caps_size
        self.C = latent_caps_size

        #specific to EM routing
        self.beta_v = nn.Parameter(torch.randn(1))
        self.beta_a = nn.Parameter(torch.randn(laten_caps_size))
        self.K = 0
        self.coordinate_add = coordinate_add
        self.transform_share = True
        self.stride = 1

        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
        self.num_iterations = num_iterations

    def forward(self, x, lambda_,):
#        t = time()
        b = x.size(0) #batchsize
        width_in = x.size(2)  #12
        use_cuda = next(self.parameters()).is_cuda
        pose = x[:,:-self.B,:,:].contiguous() #b,16*32,12,12
        pose = pose.view(b,16,self.B,width_in,width_in).permute(0,2,3,4,1).contiguous() #b,B,12,12,16
        activation = x[:,-self.B:,:,:] #b,B,12,12
        w = width_out = int((width_in-self.K)/self.stride+1) if self.K else 1 #5
        if self.transform_share:
            if self.K == 0:
                self.K = width_in # class Capsules' kernel = width_in
            W = self.W.view(self.B,1,1,self.C,self.p_vec_size,l_vec_size).expand(self.B,self.K,self.K,self.C,p_vec_size,l_vec_size).contiguous()
        else:
            W = self.W #B,K,K,C,4,4

        #used to store every capsule i's poses in each capsule c's receptive field
        poses = torch.stack([pose[:,:,self.stride*i:self.stride*i+self.K,
                       self.stride*j:self.stride*j+self.K,:] for i in range(w) for j in range(w)], dim=-1) #b,B,K,K,w*w,16
        poses = poses.view(b,self.B,self.K,self.K,1,w,w,p_vec_size,l_vec_size) #b,B,K,K,1,w,w,4,4
        W_hat = W[None,:,:,:,:,None,None,:,:]                #1,B,K,K,C,1,1,4,4
        votes = torch.matmul(W_hat, poses) #b,B,K,K,C,w,w,4,4

        #Coordinate Addition
        add = [] #K,K,w,w
        if self.coordinate_add:
            for i in range(self.K):
                for j in range(self.K):
                    for x in range(w):
                        for y in range(w):
                            #compute where is the V_ic
                            pos_x = self.stride*x + i
                            pos_y = self.stride*y + j
                            add.append([pos_x/width_in, pos_y/width_in])
            add = Variable(torch.Tensor(add)).view(1,1,self.K,self.K,1,w,w,2)
            add = add.expand(b,self.B,self.K,self.K,self.C,w,w,2).contiguous()
            if use_cuda:
                add = add.cuda()
            votes[:,:,:,:,:,:,:,0,:2] = votes[:,:,:,:,:,:,:,0,:2] + add

#        print(time()-t)
        #Start EM
        Cww = w*w*self.C
        Bkk = self.K*self.K*self.B
        R = np.ones([b,self.B,width_in,width_in,self.C,w,w])/Cww
        V_s = votes.view(b,Bkk,Cww,l_vec_size) #b,Bkk,Cww,16
        for iterate in range(self.iteration):
#            t = time()
            #M-step
            r_s,a_s = [],[]
            for typ in range(self.C):
                for i in range(width_out):
                    for j in range(width_out):
                        r = R[:,:,self.stride*i:self.stride*i+self.K,  #b,B,K,K
                                self.stride*j:self.stride*j+self.K,typ,i,j]
                        r = Variable(torch.from_numpy(r).float())
                        if use_cuda:
                            r = r.cuda()
                        r_s.append(r)
                        a = activation[:,:,self.stride*i:self.stride*i+self.K,
                                self.stride*j:self.stride*j+self.K] #b,B,K,K
                        a_s.append(a)


            r_s = torch.stack(r_s,-1).view(b, Bkk, Cww) #b,Bkk,Cww
            a_s = torch.stack(a_s,-1).view(b, Bkk, Cww) #b,Bkk,Cww
            r_hat = r_s*a_s #b,Bkk,Cww
            r_hat = r_hat.clamp(0.01) #prevent nan since we'll devide sth. by r_hat
            sum_r_hat = r_hat.sum(1).view(b,1,Cww,1).expand(b,1,Cww,l_vec_size) #b,Cww,16
            r_hat_stack = r_hat.view(b,Bkk,Cww,1).expand(b, Bkk, Cww,l_vec_size) #b,Bkk,Cww,16
            mu = torch.sum(r_hat_stack*V_s, 1, True)/sum_r_hat #b,1,Cww,16
            mu_stack = mu.expand(b,Bkk,Cww,16) #b,Bkk,Cww,16
            sigma = torch.sum(r_hat_stack*(V_s-mu_stack)**2,1,True)/sum_r_hat #b,1,Cww,16
            sigma = sigma.clamp(0.01) #prevent nan since the following is a log(sigma)
            cost = (self.beta_v + torch.log(sigma)) * sum_r_hat #b,1,Cww,16
            beta_a_stack = self.beta_a.view(1,self.C,1).expand(b,self.C,w*w).contiguous().view(b,1,Cww)#b,Cww
            a_c = torch.sigmoid(lambda_*(beta_a_stack-torch.sum(cost,3))) #b,1,Cww
            mus = mu.view(b,self.C,w,w,l_vec_size) #b,C,w,w,16
            sigmas = sigma.view(b,self.C,w,w,l_vec_size) #b,C,w,w,16
            activations = a_c.view(b,self.C,w,w) #b,C,w,w
#            print(time()-t)
#            t = time(

            #E-step
            for i in range(width_in):
                #compute the x axis range of capsules c that i connect to.
                x_range = (max(floor((i-self.K)/self.stride)+1,0),min(i//self.stride+1,width_out))
                #without padding, some capsules i may not be convolutional layer catched, in mnist case, i or j == 11
                u = len(range(*x_range))
                if not u:
                    continue
                for j in range(width_in):
                    y_range = (max(floor((j-self.K)/self.stride)+1,0),min(j//self.stride+1,width_out))

                    v = len(range(*y_range))
                    if not v:
                        continue
                    mu = mus[:,:,x_range[0]:x_range[1],y_range[0]:y_range[1],:].contiguous() #b,C,u,v,16
                    sigma = sigmas[:,:,x_range[0]:x_range[1],y_range[0]:y_range[1],:].contiguous() #b,C,u,v,16
                    mu = mu.view(b,1,self.C,u,v,l_vec_size).expand(b,self.B,self.C,u,v,l_vec_size).contiguous()#b,B,C,u,v,16
                    sigma = sigma.view(b,1,self.C,u,v,l_vec_size).expand(b,self.B,self.C,u,v,l_vec_size).contiguous()#b,B,C,u,v,16
                    V = []; a = []
                    for x in range(*x_range):
                        for y in range(*y_range):
                            #compute where is the V_ic
                            pos_x = self.stride*x - i
                            pos_y = self.stride*y - j
                            V.append(votes[:,:,pos_x,pos_y,:,x,y,:,:]) #b,B,C,4,4
                            a.append(activations[:,:,x,y].contiguous().view(b,1,self.C).expand(b,self.B,self.C).contiguous()) #b,B,C
                    V = torch.stack(V,dim=3).view(b,self.B,self.C,u,v,l_vec_size) #b,B,C,u,v,16
                    a = torch.stack(a,dim=3).view(b,self.B,self.C,u,v) #b,B,C,u,v
                    p = torch.exp(-(V-mu)**2)/torch.sqrt(2*pi*sigma) #b,B,C,u,v,16
                    p = p.prod(dim=5)#b,B,C,u,v
                    p_hat = a*p  #b,B,C,u,v
                    sum_p_hat = p_hat.sum(4).sum(3).sum(2) #b,B
                    sum_p_hat = sum_p_hat.view(b,self.B,1,1,1).expand(b,self.B,self.C,u,v)
                    r = (p_hat/sum_p_hat) #b,B,C,u,v --> R: b,B,12,12,32,5,5

                    if use_cuda:
                        r = r.cpu()
                    R[:,:,i,j,:,x_range[0]:x_range[1],        #b,B,u,v,C
                      y_range[0]:y_range[1]] = r.data.numpy()
#            print(time()-t)

        mus = mus.permute(0,4,1,2,3).contiguous().view(b,self.C*l_vec_size,w,w)#b,16*C,5,5
        output = torch.cat([mus,activations], 1) #b,C*16,5,5
        return output

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size=latent_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()


class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        reconstructions = self.caps_decoder(latent_capsules)
        return latent_capsules, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss

# This is a single network which can decode the point cloud from pre-saved latent capsules
class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size,digit_vec_size, num_points)
    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return  reconstructions

if __name__ == '__main__':
    USE_CUDA = True
    batch_size=8

    prim_caps_size=1024
    prim_vec_size=16

    latent_caps_size=32
    latent_vec_size=16

    num_points=2048

    point_caps_ae = PointCapsNet(prim_caps_size,prim_vec_size,latent_caps_size,latent_vec_size,num_points)
    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()

    rand_data=torch.rand(batch_size,num_points, 3)
    rand_data = Variable(rand_data)
    rand_data = rand_data.transpose(2, 1)
    rand_data=rand_data.cuda()

    codewords,reconstruction=point_caps_ae(rand_data)

    rand_data_ = rand_data.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()

    dist1, dist2 = distChamfer(rand_data_, reconstruction_)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    print(loss.item())
