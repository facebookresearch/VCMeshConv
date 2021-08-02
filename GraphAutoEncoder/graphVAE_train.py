"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import trimesh
import pathlib

import io
from PIL import Image

import numpy as np
import graphVAESSW as vae_model
import graphAE_param_iso as Param
import graphAE_dataloader as Dataloader
from datetime import datetime

import random
import json

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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


device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
print(device)
if bTrain:
    device_ids = [0,1,2,3]
else:
    # device_ids = [0]
    device_ids = [0,1,2,3]


batsize_perdev = param.batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, param):

        self.trackpath = pathlib.Path(param.mesh_train)
        self._files_list = [name
                            for name in os.listdir(self.trackpath)
                            if name.lower().endswith(".obj")]

    def __len__(self):
        return len(self._files_list)

    def __getitem__(self, idx):

        # idx = idx % 1

        file_name = self._files_list[idx]

        mesh = trimesh.load(self.trackpath / file_name)
        x = np.array(mesh.vertices[:,0]).astype(np.float32)
        y = np.array(mesh.vertices[:,1]).astype(np.float32)
        z = np.array(mesh.vertices[:,2]).astype(np.float32)
        meshvet = torch.from_numpy(np.array([x,y,z])).permute(1,0)
        numvertices = meshvet.size(0)

        pc_aug = meshvet * SCALE

        mesh_faces = torch.from_numpy(np.array(mesh.faces))
        vet0 = index_selection_nd(pc_aug, mesh_faces[:,0],0)
        vet1 = index_selection_nd(pc_aug, mesh_faces[:,1],0)
        vet2 = index_selection_nd(pc_aug, mesh_faces[:,2],0)

        nor = (vet1 - vet0).cross(vet2 - vet1)
        nor = nor/(nor.pow(2).sum(1,keepdim=True).sqrt() + 1e-6)
        return pc_aug, nor, mesh_faces


class Net_autoenc(nn.Module):
    def __init__(self,param):
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


        self.nrpt_latent = self.net_geoenc.out_number_points


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
        decgeochannel_list3 = encgeochannel_list.copy()
        decgeochannel_list3.reverse()
        # here defines the decoder with deconv layers with learnable conv kernels
        self.net_geodec = vae_model.MCDec(self.mcstructuredec, decgeochannel_list3, self.weight_num)
        # self.net_texdec = vae_model.MCDec(self.mcstructuredec, decgeochannel_list3, self.weight_num)


        self.net_loss = vae_model.MCLoss(param)

        self.w_pose = param.w_pose
        self.w_laplace = param.w_laplace #0.5
        self.klweight = 1e-5 #0.00001
        self.w_nor = 10.0

        self.write_tmp_folder =  param.write_tmp_folder #+"%07d"%iteration+"_%02d_out"%n+suffix+".ply"

    def forward(self, in_pc_batch, iteration, t_nor, faces, bDebug = False): # meshvertices: B N 3, meshnormals: B N 3
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

        faces_long = faces.long()
        vet0 = index_selection_nd(dif_pos, faces[:, :, 0], 1)
        vet1 = index_selection_nd(dif_pos, faces[:, :, 1], 1)
        vet2 = index_selection_nd(dif_pos, faces[:, :, 2], 1)

        loss_normal = ((vet1 + vet0 + vet2)/3.0 * t_nor).sum(2).pow(2).mean()

        loss_pose_l1 = self.net_loss.compute_geometric_loss_l1(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])
        loss_laplace_l1 = self.net_loss.compute_laplace_loss_l2(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])

        loss = (loss_pose_l1*self.w_pose +
                loss_laplace_l1 * self.w_laplace +
                klloss * self.klweight +
                loss_normal * self.w_nor)

        outvetgeo = (out_pc_batch/ SCALE)
        gtvetgeo = (in_pc_batch/ SCALE)

        # to plot a graph in tensorboard
        if isinstance(iteration, torch.Tensor):
            iteration = 1
            bDebug = False

        if iteration % 500 == 0:
            gt_verts = gtvetgeo.cpu().detach()
            out_verts = outvetgeo.cpu().detach()
            faces_ = faces.cpu().detach()
            param.logger.add_mesh("GT_mesh", vertices=gt_verts[:1], faces=faces_[:1],
                                  global_step=iteration)
            param.logger.add_mesh("OUT_mesh", vertices=out_verts[:1], faces=faces_[:1],
                                  global_step=iteration)

        if bDebug and in_pc_batch.get_device() == 0:

            err_pose_l2 = self.net_loss.compute_geometric_mean_euclidean_dist_error(in_pc_batch[:1,:,0:3], out_pc_batch[:1,:,0:3])
            err_pose_l1 = self.net_loss.compute_laplace_loss_l1(in_pc_batch[:1,:,0:3], out_pc_batch[:1,:,0:3])

            with open(f'{self.write_tmp_folder}/err{iteration}.txt','w') as f:
                f.write('{0:d} {1:f} {2:f} {3:f}\n'.format(iteration,loss_pose_l1.mean(),err_pose_l2, err_pose_l1))
                f.closed

        return loss[None],loss_pose_l1[None],loss_laplace_l1[None], klloss[None], loss_normal[None]


def add_grad_histogram(params, name, iteration):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in params:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())

    _limits = np.array([float(i) for i in range(len(ave_grads))])
    _num = len(ave_grads)
    ave_grads = np.array(ave_grads)
    param.logger.add_histogram_raw(
        tag=f"{name}/grads/abs_mean", min=0.0, max=0.5, num=_num,
        sum=ave_grads.sum(), sum_squares=np.power(ave_grads, 2).sum(),
        bucket_limits=_limits, bucket_counts=ave_grads, global_step=iteration)

def add_grad_histogram_per_layer(module, name, iteration):
    ave_grads = []
    sighned_ave_grads = []
    max_grads= []
    layers = []
    for n, p in module.layer_list.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            sighned_ave_grads.append(p.grad.mean().cpu())
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())

    _limits = np.array([float(i) for i in range(len(ave_grads))])
    _num = len(ave_grads)
    sighned_ave_grads = np.array(sighned_ave_grads)
    ave_grads = np.array(ave_grads)
    max_grads = np.array(max_grads)

    param.logger.add_histogram_raw(
        f"{name}/grads/abs_mean", min=0.0, max=0.5, num=_num,
        sum=ave_grads.sum(), sum_squares=np.power(ave_grads, 2).sum(),
        bucket_limits=_limits, bucket_counts=ave_grads, global_step=iteration)
    _signed_mean = {}
    _mean = {}
    _maxes = {}
    for i, layer_name in enumerate(layers):
        _signed_mean[layer_name] = sighned_ave_grads[i]
        _mean[layer_name] = ave_grads[i]
        _maxes[layer_name] = max_grads[i]
    param.logger.add_scalars(name+"/grads/per_layer/signed_mean",
                             _signed_mean, global_step=iteration)
    param.logger.add_scalars(name+"/grads/per_layer/abs_mean",
                             _mean, global_step=iteration)
    param.logger.add_scalars(name+"/grads/per_layer/abs_max",
                             _maxes, global_step=iteration)


def get_network_named_params(net_autoenc):
    return filter(lambda x: x[1].requires_grad,
                       itertools.chain(
                                       net_autoenc.module.net_geodec.named_parameters(),
                                       net_autoenc.module.net_geoenc.named_parameters(),
                                       net_autoenc.module.mcvcoeffsdec.named_parameters(),
                                       net_autoenc.module.mcvcoeffsenc.named_parameters(),
                                       ))

def get_network_params(net_autoenc):
    return filter(lambda x: x.requires_grad,
                       itertools.chain(
                                       net_autoenc.module.net_geodec.parameters(),
                                       net_autoenc.module.net_geoenc.parameters(),
                                       net_autoenc.module.mcvcoeffsdec.parameters(),
                                       net_autoenc.module.mcvcoeffsenc.parameters(),
                                       ))

def getoptim(learningrate, net_autoenc):
    ae_params = get_network_params(net_autoenc)
    ae_optim = torch.optim.Adam(ae_params, lr=learningrate, betas=(0.9, 0.999))

    return ae_optim


# construct the Dataset
dataset_train = Dataset(param)

# construct the main training class and make it data parallel
net_autoenc = Net_autoenc(param)
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


dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=param.batch * len(device_ids), shuffle=True, num_workers=0)

min_geo_error=123456

def train():

    iteration = param.start_iter
    first_run = True

    for epoch in range(10000):
        for element in dataloader_train:
            param.logger.add_scalar('Iteration', iteration, iteration)
            pc, nor, faces = element

            if first_run:
                param.logger.add_graph(net_autoenc, (pc, torch.Tensor(iteration), nor, faces, torch.BoolTensor([True])))
                first_run = False

            pc_batch = pc.to(device)
            t_nor = nor.to(device)
            t_faces = faces.to(device)

            if iteration%param.save_tmp_iter==1:
                loss,loss_pose_l1,loss_laplace_l1, klloss, loss_normal  = net_autoenc(pc_batch, iteration, t_nor, t_faces, True)
            else:
                loss,loss_pose_l1,loss_laplace_l1, klloss, loss_normal  = net_autoenc(pc_batch, iteration, t_nor, t_faces, False)


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
                param.logger.add_scalar('Loss', loss.mean(), iteration)
                param.logger.add_scalar('Loss_pose_l1', loss_pose_l1.mean(), iteration)
                param.logger.add_scalar('Loss_laplace_l1', loss_laplace_l1.mean(), iteration)
                param.logger.add_scalar('Loss_kl', klloss.mean(), iteration)
                param.logger.add_scalar('Loss_normal', loss_normal.mean(), iteration)

                param.logger.add_scalar('Weighted_Loss_pose_l1',
                                        (loss_pose_l1 * param.w_pose).mean(), iteration)

                param.logger.add_scalar('Weighted_Loss_laplace_l1',
                                        (param.w_laplace * loss_laplace_l1).mean(), iteration)

                param.logger.add_scalar('Weighted_Loss_kl',
                                        (klloss * 1e-5).mean(), iteration)

                param.logger.add_scalar('Weighted_Loss_normal',
                                        (loss_normal * 10.0).mean(), iteration)
                add_grad_histogram_per_layer(net_autoenc.module.net_geoenc, "encoder", iteration)
                add_grad_histogram_per_layer(net_autoenc.module.net_geodec, "decoder", iteration)

                add_grad_histogram(get_network_named_params(net_autoenc), "all_params", iteration)
                add_grad_histogram(filter(lambda x: x[1].requires_grad,
                                          itertools.chain(
                                              net_autoenc.module.net_geodec.named_parameters(),
                                              net_autoenc.module.mcvcoeffsdec.named_parameters(),
                                          )), "decoder", iteration)
                add_grad_histogram(filter(lambda x: x[1].requires_grad,
                                          itertools.chain(
                                              net_autoenc.module.net_geoenc.named_parameters(),
                                              net_autoenc.module.mcvcoeffsenc.named_parameters(),
                                          )), "encoder", iteration)

            #just for testing
            if iteration%param.evaluate_iter == 0:
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

if __name__ == "__main__":
    train()
