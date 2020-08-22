"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import os
from plyfile import PlyData, PlyElement
import torch
#from multiprocessing.dummy import Pool as ThreadPool 

import transforms3d.euler as euler
from os import mkdir
from os.path import join, exists



SCALE = 1

def get_ply_fn_lst(folder_lst):
    ply_fn_lst =[]
    for folder in folder_lst:
        name_lst = os.listdir(folder)
        for name in name_lst:
            if(".ply" in name):
                fn=folder + "/"+name
                ply_fn_lst+=[fn]
    return ply_fn_lst



#pc p_num*3 np
def get_augmented_pc(pc):
    size= pc.shape[0]
    new_pc = pc
    
    axis = np.random.rand(3)
    axis = axis / np.sqrt(pow(axis,2).sum())    
    
    theta= (np.random.rand()-0.5)*np.pi*2
    
    Rorgmat = euler.axangle2mat(axis,theta)
    R = Rorgmat.reshape((1,3,3)).repeat(size,0)    
    Torg = (np.random.rand(1,3,1)-0.5)*0.2 #-10cm to 10cm    
    T=Torg.repeat(size,0)
    
    new_pc = np.matmul(R, new_pc.reshape((size,3,1))) +T
    
    return new_pc.reshape((size,3))

def get_augmented_pc_ret(pc):
    size= pc.shape[0]
    new_pc = pc

    axis = np.random.rand(3)
    axis = axis / np.sqrt(pow(axis,2).sum())
    
    theta= (np.random.rand()-0.5)*np.pi*2
    
    Rorgmat = euler.axangle2mat(axis,theta)
    R = Rorgmat.reshape((1,3,3)).repeat(size,0)
    Torg = (np.random.rand(1,3,1)-0.5)*0.2 #-10cm to 10cm
    T=Torg.repeat(size,0)
    
    new_pc = np.matmul(R, new_pc.reshape((size,3,1))) +T
    
    return new_pc.reshape((size,3)), Rorgmat, Torg[...,0]


##return batch*p_num*3 torch
def get_random_pc_batch_from_ply_fn_lst_torch(ply_fn_lst , batch, augmented=False):
    
    ply_fn_batch = []
    
    for b in range(batch):
        index = np.random.randint(0, len(ply_fn_lst))
        ply_fn_batch+=[ply_fn_lst[index]]
      
    pc_batch=[]
    for ply_fn in ply_fn_batch:
        pc = get_pc_from_ply_fn(ply_fn)
        if(augmented==True):
            pc = get_augmented_pc(pc)
        pc_batch +=[pc]
    
    pc_batch_torch = torch.FloatTensor(pc_batch).cuda()
    
    return pc_batch_torch

#num*p_num*3 numpy
def get_all_pcs_from_ply_fn_lst_np(ply_fn_lst):
    pc_lst=[]
    n=0
    for ply_fn in ply_fn_lst:
        pc = get_pc_from_ply_fn(ply_fn)
        pc_lst +=[pc]
        if(n%100==0):
            print (n)
        n=n+1
    print ("load", n, "pcs")
    
    return pc_lst


##return batch*p_num*3 torch  # batch*p_num*3 torch
def get_random_pc_batch_from_pc_lst_torch(pc_lst , neighbor_lst, neighbor_num_lst, batch, augmented=False):
    
    #pc_colors_batch = []
    weights_batch=[]
    pc_batch = []
    for b in range(batch):
        index = np.random.randint(0, len(pc_lst))
        pc_weights = pc_lst[index]
        pc = pc_weights[:,0:3]
        weights = pc_weights[:,3]
        #colors = pc_weights[:,3:6]

        if(augmented==True):
            pc=get_augmented_pc(pc)
        pc_batch +=[pc]
        #pc_colors = np.concatenate((pc, colors),1)
        #pc_colors_batch+=[pc_colors]
        weights_batch+=[weights]
    #pc_colors_batch = np.array(pc_colors_batch)
    weights_batch = np.array(weights_batch)
    #smoothed_pc_batch = get_smoothed_pc_batch_iter(pc_batch,neighbor_lst, neighbor_num_lst)
        
    weights_batch_torch = torch.FloatTensor(weights_batch).cuda()
    #pc_colors_batch_torch = torch.FloatTensor(pc_colors_batch).cuda()
    pc_batch_torch = torch.FloatTensor(pc_batch).cuda()

    #smoothed_pc_batch_torch = torch.FloatTensor(smoothed_pc_batch).cuda()
    return pc_batch_torch , weights_batch_torch #, smoothed_pc_batch_torch


##return batch*p_num*3 torch  # batch*p_num*3 torch
def get_indexed_pc_from_pc_lst_torch(pc_lst , index, augmented=False):
    
    pc_weights = pc_lst[index]
    pc = pc_weights[:,0:3]
    weights = pc_weights[:,3]
    
    if(augmented==True):
        pc=get_augmented_pc(pc)

    weights_torch = torch.from_numpy(weights).float()
    pc_torch = torch.from_numpy(pc).float()
        
    return pc_torch , weights_torch #, smoothed_pc_batch_torch

#point_num*3
def compute_and_save_ply_mean(folder_lst, pc_fn):
    ply_fn_lst=get_ply_fn_lst(folder_lst)
    pc_batch = []
    for ply_fn in ply_fn_lst:
        pc = get_pc_from_ply_fn(ply_fn)
        pc_batch +=[pc]
    pc_batch = np.array(pc_batch)
    pc_mean  = pc_batch.mean(0)
    pc_std = pc_batch.std(0)
    np.save(pc_fn+"mean", pc_mean)
    np.save(pc_fn+"std", pc_std)
    return pc_mean ,pc_std
    
    
    
    

#pc p_num np*3
#template_ply Plydata
def save_pc_into_ply(template_ply, pc, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]
    plydata.write(fn)
    
#pc p_num np*3
#color p_num np*3 (0-255)
#template_ply Plydata
def save_pc_with_color_into_ply(template_ply, pc, color, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]
    
    plydata['vertex']['red']=color[:,0]
    plydata['vertex']['green']=color[:,1]
    plydata['vertex']['blue']=color[:,2]
    
    plydata.write(fn)
    plydata['vertex']['red']=plydata['vertex']['red']*0+0.7
    plydata['vertex']['green']=plydata['vertex']['red']*0+0.7
    plydata['vertex']['blue']=plydata['vertex']['red']*0+0.7
    
    
def get_smoothed_pc_batch_iter(pc, neighbor_lst,neighbor_num_lst, iteration=10):
    smoothed_pc = get_smoothed_pc_batch(pc,neighbor_lst,neighbor_num_lst)
    for i in range(iteration):
        smoothed_pc = get_smoothed_pc_batch(smoothed_pc,neighbor_lst,neighbor_num_lst)
    return smoothed_pc

#pc batch*point_num*3
#neibhor_lst point_num*max_neighbor_num
#neibhor_num_lst point_num
def get_smoothed_pc_batch(pc, neighbor_lst, neighbor_num_lst):
    batch = pc.shape[0]
    point_num = pc.shape[1]
    pc_padded = np.concatenate((pc, np.zeros((batch, 1,3))),1) #batch*(point_num+1)*1
    smoothed_pc = pc.copy()
    for n in range(1,neighbor_lst.shape[1]):
        smoothed_pc += pc_padded[:,neighbor_lst[:,n]]
        
    smoothed_pc =smoothed_pc / neighbor_num_lst.reshape((1,point_num,1)).repeat(batch,0).repeat(3, 2)
    
    return smoothed_pc

def get_smoothed_pc_iter(pc, neighbor_lst,neighbor_num_lst, iteration=10):
    smoothed_pc = get_smoothed_pc(pc,neighbor_lst,neighbor_num_lst)
    for i in range(iteration):
        smoothed_pc = get_smoothed_pc(smoothed_pc,neighbor_lst,neighbor_num_lst)
    return smoothed_pc


#pc point_num*3
#neibhor_lst point_num*max_neighbor_num
#neibhor_num_lst point_num
def get_smoothed_pc(pc, neighbor_lst, neighbor_num_lst):
    point_num =pc.shape[0]
    pc_padded = np.concatenate((pc, np.zeros((1,3))),0) #batch*(point_num+1)*1
    smoothed_pc = pc.copy()
    for n in range(1,neighbor_lst.shape[1]):
        smoothed_pc += pc_padded[neighbor_lst[:,n]]

    smoothed_pc =smoothed_pc / neighbor_num_lst.reshape(point_num,1).repeat(3,1)
    
    return smoothed_pc


def transform_plys_to_npy(ply_folder, npy_fn):
    pcs = []
    name_lst = os.listdir(ply_folder)
    n=0
    for name in name_lst:
        if(".ply" in name):
            if(n%100==0):
                print (n)
            fn = ply_folder+"/"+name
            pc = get_pc_from_ply_fn(fn)
            pcs+=[pc]
            n+=1
            
    pcs = np.array(pcs)
    
    np.save(npy_fn, pcs)
    
    
def get_pcs_from_ply_folder(ply_folder):
    pcs = []
    name_lst = os.listdir(ply_folder)
    n=0
    for name in name_lst:
        if(".ply" in name):
            if(n%100==0):
                print (n)
            fn = ply_folder+"/"+name
            pc = get_pc_from_ply_fn(fn)
            pcs+=[pc]
            n+=1
            
    pcs = np.array(pcs)
    
    return pcs

