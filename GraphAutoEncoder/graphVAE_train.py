# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import itertools

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np
import graphVAESSW as vae_model
import graphAE_param_iso as Param
import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData # the ply loader here is using. It is suggested to use faster load function to reduce the io overhead

import random
import json


SCALE = 0.001  

bTrain = False
# bTrain = True

###############################################################################
def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)


if len(sys.argv) < 2:
    print('usage: python this_script config_file')
    quit()
config_file = str(sys.argv[1])



param=Param.Parameters()
param.read_config(config_file)


framelist = np.genfromtxt(param.framelist, dtype=np.str)

device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
print(device)
if bTrain:
    device_ids = [0,1,2,3]
else:
    # device_ids = [0]
    device_ids = [0,1,2,3]


batsize_perdev = param.batch

renderfrrange_st = 500 #62250
renderfrrange_end = 4000 #62934 #106000

if bTrain:
    renderfrrange_st = 500 #62250
    renderfrrange_end = 4000 #62934 #106000



class Dataset(torch.utils.data.Dataset):
    def __init__(self, param, framelist, faceidx):        

        self.trackpath = param.mesh_train
        self.framelist = []
        countpose = 0
        for i, x in enumerate(framelist):        
            f = (x,countpose,)  
            frid = (int)(x)
            if frid>= renderfrrange_st and frid<=renderfrrange_end:
                self.framelist.append(f)
                countpose = countpose + 1
        print(len(self.framelist))
        
        self.faceidx = faceidx.long()        
        

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, idxIn):
        idx = idxIn
        frame, indexfr = self.framelist[idx]    

        #loading unposed meshes
        meshname = f'{self.trackpath}{(int)(frame):06d}.ply'
        if os.path.isfile(meshname):            
            plydata = PlyData.read(meshname)
            x=np.array(plydata['vertex']['x'])
            y=np.array(plydata['vertex']['y'])
            z=np.array(plydata['vertex']['z'])        
            meshvet = torch.from_numpy(np.array([x,y,z])).permute(1,0)         
        else:
            print(f'cannot find {meshname}')

        numvertices = meshvet.size(0)
        
        pc_aug = meshvet * SCALE

        vet0 = index_selection_nd(pc_aug,self.faceidx[:,0],0)
        vet1 = index_selection_nd(pc_aug,self.faceidx[:,1],0)
        vet2 = index_selection_nd(pc_aug,self.faceidx[:,2],0)

        nor = (vet1 - vet0).cross(vet2 - vet1)
        nor = nor/(nor.pow(2).sum(1,keepdim=True).sqrt() + 1e-6)


        return pc_aug, nor, frame, indexfr








class Net_autoenc(nn.Module):
    def __init__(self,param, facedata):
        super(Net_autoenc, self).__init__()

        self.weight_num = 17
        self.motdim = 94

        # build the mesh convolution structure, which contains the connectivity at each layer; 
        # note this class donesnot have any learnable parameters
        # here is defining for encoder graph connectivity structure
        self.mcstructureenc = vae_model.MCStructure(param, param.point_num, self.weight_num, bDec =False)

        # corresponding to vc in the paper and is connectivity related
        # this defines a parameter class for spatially varying coefficients
        # this vc enables to apply traditional conv with learnable conv kernels
        # this vc can be potentially shared across different network as long as it is defined with the same MCStructure (graph connectivity)
        self.mcvcoeffsenc = vae_model.MCVcoeffs(self.mcstructureenc, self.weight_num)       
        
        # here defines the input and output channel numbers
        encgeochannel_list =  [3, 32,64,128, 256,512,64]
        # here defines the encoder with learnable conv kernels
        self.net_geoenc = vae_model.MCEnc(self.mcstructureenc, encgeochannel_list, self.weight_num)
        # self.net_texenc = vae_model.MCEnc(self.mcstructureenc, encgeochannel_list, self.weight_num)


        self.nrpt_latent = self.net_geoenc.out_nrpts


        # build the mesh convolution structure, which contains the connectivity at each layer; 
        # note this class donesnot have any learnable parameters
        # here is defining for decoder graph connectivity structure
        self.mcstructuredec = vae_model.MCStructure(param,self.nrpt_latent,self.weight_num, bDec = True)       

        # corresponding to vc in the paper and is connectivity related
        # this defines a parameter class for spatially varying coefficients
        # this vc enables to apply traditional conv with learnable conv kernels
        # this vc can be potentially shared across different network as long as it is defined with the same MCStructure (graph connectivity)
        self.mcvcoeffsdec = vae_model.MCVcoeffs(self.mcstructuredec, self.weight_num)       

        # here defines the input and output channel numbers
        decgeochannel_list3 =  [64, 512,256,128, 64,32,3]
        # here defines the decoder with deconv layers with learnable conv kernels
        self.net_geodec = vae_model.MCDec(self.mcstructuredec, decgeochannel_list3, self.weight_num)        
        # self.net_texdec = vae_model.MCDec(self.mcstructuredec, decgeochannel_list3, self.weight_num)


        self.net_loss = vae_model.MCLoss(param)
        
        
        self.register_buffer('t_facedata', facedata.long())


        self.w_pose = param.w_pose 
        self.w_laplace = param.w_laplace #0.5
        self.klweight = 1e-5 #0.00001
        self.w_nor = 10.0

        self.write_tmp_folder =  param.write_tmp_folder #+"%07d"%iteration+"_%02d_out"%n+suffix+".ply"
        

    def forward(self, in_pc_batch, iteration, frame, t_idx, t_nor, bDebug = False): # meshvertices: B N 3, meshnormals: B N 3    

        nbat = in_pc_batch.size(0)
        npt = in_pc_batch.size(1)
        nch = in_pc_batch.size(2)


        t_mu, t_logstd = self.net_geoenc(in_pc_batch, self.mcvcoeffsenc) # in in mm and out in dm
        t_std = t_logstd.exp()

        t_eps = torch.ones_like(t_std).normal_() #torch.FloatTensor(t_std.size()).normal_().to(device)
        t_z = t_mu + t_std * t_eps 

        
        klloss = torch.mean(-0.5 - t_logstd + 0.5 * t_mu ** 2 + 0.5 * torch.exp(2 * t_logstd))

        out_pc_batchfull = self.net_geodec(t_z, self.mcvcoeffsdec)
        out_pc_batch = out_pc_batchfull[:,:,0:3]


        dif_pos = out_pc_batch - in_pc_batch
        vet0 = index_selection_nd(dif_pos,self.t_facedata[:,0],1)
        vet1 = index_selection_nd(dif_pos,self.t_facedata[:,1],1)
        vet2 = index_selection_nd(dif_pos,self.t_facedata[:,2],1)

        loss_normal = ((vet1 + vet0 + vet2)/3.0 * t_nor).sum(2).pow(2).mean()
        

        loss_pose_l1 = self.net_loss.compute_geometric_loss_l1(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])
        loss_laplace_l1 = self.net_loss.compute_laplace_loss_l2(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])



        loss = loss_pose_l1*self.w_pose +  loss_laplace_l1 * self.w_laplace  + klloss * self.klweight + loss_normal * self.w_nor

             
        outvetgeo = (out_pc_batch/ SCALE)
        gtvetgeo = (in_pc_batch/ SCALE)
        
        
        if bDebug and in_pc_batch.get_device() == 0:  

            wtid = t_idx[0].cpu()
            wtid = (int)(frame[0])
            print(wtid)

            err_pose_l2 = self.net_loss.compute_geometric_mean_euclidean_dist_error(in_pc_batch[:1,:,0:3], out_pc_batch[:1,:,0:3])
            err_pose_l1 = self.net_loss.compute_laplace_loss_l1(in_pc_batch[:1,:,0:3], out_pc_batch[:1,:,0:3])

            with open(f'{self.write_tmp_folder}/err{iteration}.txt','w') as f:
                f.write('{0:d} {1:f} {2:f} {3:f}\n'.format(iteration,loss_pose_l1.mean(),err_pose_l2, err_pose_l1))
                f.closed

            ##just output some meshes


                           
        return loss[None],loss_pose_l1[None],loss_laplace_l1[None], klloss[None], loss_normal[None]


def getoptim(learningrate, net_autoenc):    
    ae_params = filter(lambda x: x.requires_grad,
                       itertools.chain(
                                       net_autoenc.module.net_geodec.parameters(),
                                       net_autoenc.module.net_geoenc.parameters(),
                                       net_autoenc.module.mcvcoeffsdec.parameters(),
                                       net_autoenc.module.mcvcoeffsenc.parameters(),
                                       ))
    ae_optim = torch.optim.Adam(ae_params, lr=learningrate, betas=(0.9, 0.999))    

    return ae_optim


print ("*********Load Template ply************")
# loading template mesh
templydata = PlyData.read(param.template_ply_fn)
tri_idx = templydata['face']['vertex_indices']
temply_facedata = torch.from_numpy(np.vstack(tri_idx))


# construct the Dataset
dataset_train = Dataset(param, framelist, temply_facedata)

# construct the main training class and make it data parallel
net_autoenc = Net_autoenc(param, temply_facedata)
net_autoenc = nn.DataParallel(net_autoenc, device_ids=device_ids).to(device)

# using Adam for optimizer
optimizer = getoptim(param.lr, net_autoenc)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# if need to preload to initialize the network
if(param.read_weight_path!=""):
    print ("load "+param.read_weight_path)
    checkpoint = torch.load(param.read_weight_path)
    net_autoenc.module.net_geoenc.load_state_dict(checkpoint['encgeo_state_dict'])
    net_autoenc.module.net_geodec.load_state_dict(checkpoint['decgeo_state_dict'])    
    net_autoenc.module.mcvcoeffsenc.load_state_dict(checkpoint['mcvcoeffsenc_dict'])    
    net_autoenc.module.mcvcoeffsdec.load_state_dict(checkpoint['mcvcoeffsdec_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=param.batch * len(device_ids), shuffle=True, num_workers=20)  

min_geo_error=123456

iteration = param.start_iter
for epoch in range(10000):
    for element in dataloader_train:           
        pc, nor, frame, idx = element

        pc_batch = pc.to(device)
        t_nor = nor.to(device)       
        t_idx = idx.long().to(device)
        

        if iteration%param.save_tmp_iter==1:
            loss,loss_pose_l1,loss_laplace_l1, klloss, loss_normal  = net_autoenc(pc_batch, iteration,frame, t_idx, t_nor, True)
        else:
            loss,loss_pose_l1,loss_laplace_l1, klloss, loss_normal  = net_autoenc(pc_batch, iteration,frame, t_idx, t_nor, False)


        optimizer.zero_grad()
        loss.backward(torch.ones(loss.size(0), device=device))
        optimizer.step()
        
    
        if(iteration%20 == 0):
            print ("###Iteration", epoch, iteration)
            print ("loss_pose_l1:", loss_pose_l1.mean().item())
            print ("loss_laplace_l1:", loss_laplace_l1.mean().item())
            print ("loss_kl:", klloss.mean().item())
            print ("normal_loss:", loss_normal.mean().item())
            print ("loss:", loss.mean().item())

            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        
        if(iteration%5 == 0): 
            param.logger.add_scalars('Train_without_weight', {'Loss_pose_l1': loss_pose_l1.mean().item()},iteration)
            param.logger.add_scalars('Train_without_weight', {'Loss_laplace_l1': loss_laplace_l1.mean().item()},iteration)
            #param.logger.add_scalars('Train_without_weight', {'rate_nonzero_w_weights:': rate_nonzero_w_weights.mean()},iteration)
        
            param.logger.add_scalars('Train_with_weight', {'Loss_pose_l1': (loss_pose_l1*param.w_pose).mean().item()},iteration)
            param.logger.add_scalars('Train_with_weight', {'Loss_laplace_l1': (loss_laplace_l1*param.w_laplace).mean().item()},iteration)

        #just for testing
        if(iteration%param.evaluate_iter==0):           
            print ("###Save Weight#####")
            path = param.write_weight_folder + "model_%07d"%iteration +"_best.weight"
            torch.save({
            'encgeo_state_dict':net_autoenc.module.net_geoenc.state_dict(),\
            'decgeo_state_dict':net_autoenc.module.net_geodec.state_dict(),\
            'mcvcoeffsdec_dict': net_autoenc.module.mcvcoeffsdec.state_dict(),\
            'mcvcoeffsenc_dict': net_autoenc.module.mcvcoeffsenc.state_dict(),\
            'optimizer_state_dict': optimizer.state_dict()},path)

        iteration += 1
        if iteration>param.end_iter:
            break

    scheduler.step()
     
