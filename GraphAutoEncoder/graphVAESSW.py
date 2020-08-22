"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
import torch.nn as nn
import numpy as np



def normalize_weights(weights):
    num = weights.shape[0]
    channel = weights.shape[1]
    #weights.normal_()
    weights_norm = weights.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
    weights =  weights/ weights_norm.view(num, 1).repeat(1, channel)

def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)

class LASMConvssw(nn.Module):
    def __init__(self, in_channel, out_channel, weight_num,in_point_num,  connection_info, b_Perpt_bias = True,  residual_rate = 0.0): #layer_info_lst= [(point_num, feature_dim)]
        super(LASMConvssw, self).__init__()

        self.relu = nn.ELU()        

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight_num = weight_num 
        self.in_point_num = in_point_num
        
        
        out_point_num = connection_info.shape[0]
        self.out_point_num = out_point_num 
                
        neighbor_num_lst = torch.from_numpy(connection_info[:,0].astype(np.float32)).float() #out_point_num*1
        self.register_buffer("neighbor_num_lst", neighbor_num_lst)
        
        neighbor_id_dist_lstlst = connection_info[:, 1:] #out_point_num*(max_neighbor_num*2)
        neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1,2))[:,:,0] #out_point_num*max_neighbor_num
        
        neighbor_id_lstlst = torch.from_numpy(neighbor_id_lstlst).long()
        self.register_buffer("neighbor_id_lstlst", neighbor_id_lstlst)
        
        max_neighbor_num = neighbor_id_lstlst.shape[1]
        self.max_neighbor_num = max_neighbor_num

        avg_neighbor_num= round(neighbor_num_lst.mean().item())
        self.avg_neighbor_num = avg_neighbor_num
            
            
        ####parameters for conv###############
        weights = nn.Parameter(torch.randn(weight_num, out_channel*in_channel))
        self.register_parameter("weights",weights)
            
        bias = nn.Parameter(torch.zeros(out_channel))
        if b_Perpt_bias:
            bias= nn.Parameter(torch.zeros(out_point_num, out_channel))

        self.register_parameter("bias",bias)
        
        self.residual_rate = residual_rate

        ####parameters for residual###############
        #residual_layer = ""
        if self.residual_rate > 0:
        
            if(out_point_num != in_point_num):        
                p_neighbors = nn.Parameter(torch.randn(out_point_num, max_neighbor_num)/(avg_neighbor_num))
                self.register_parameter("p_neighbors",p_neighbors)

            if(out_channel != in_channel):
                weight_res = torch.randn(1, out_channel*in_channel)                
                weight_res = weight_res/out_channel

                weight_res = nn.Parameter(weight_res)
                self.register_parameter("weight_res",weight_res)


        print ("in_channel", in_channel,\
        	"out_channel",out_channel, \
        	"in_point_num", in_point_num, \
        	"out_point_num", out_point_num, \
        	"weight_num", weight_num,\
            "max_neighbor_num", max_neighbor_num)
            
        
    # improved version which takes less mem
    def forward(self, in_pc, raw_w_weights, is_final_layer=False, b_max_pool = False):
        batch = in_pc.shape[0]
        device = in_pc.device #in_pc.device
        
        in_channel = self.in_channel
        out_channel = self.out_channel 
        in_pn = self.in_point_num
        out_pn = self.out_point_num 
        weight_num = self.weight_num  #M
        max_neighbor_num = self.max_neighbor_num #N
        neighbor_num_lst = self.neighbor_num_lst
        neighbor_id_lstlst = self.neighbor_id_lstlst
        
        
        pc_mask  = torch.ones(in_pn+1).float().to(in_pc.device)
        pc_mask[in_pn]=0                
        neighbor_mask_lst = index_selection_nd(pc_mask,neighbor_id_lstlst,0).contiguous()#out_pn*max_neighbor_num neighbor is 1 otherwise 0
              
        raw_weights = self.weights
        bias = self.bias
        
        w_weights = raw_w_weights*(neighbor_mask_lst.view(out_pn, max_neighbor_num, 1)) #out_pn*max_neighbor_num*weight_num
        
        
        in_pc_pad = torch.cat((in_pc, torch.zeros(batch, 1, in_channel).float().to(in_pc.device)), 1) #batch (in_pn+1) in_channel
        in_neighbors = index_selection_nd(in_pc_pad,neighbor_id_lstlst, 1)                
        fuse_neighbors = torch.einsum('pnm,bpni->bpmi',[w_weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel             
        
        normalized_weights = raw_weights.view(weight_num,out_channel,in_channel)        
        out_neighbors = torch.einsum('moi,bpmi->bpmo',[normalized_weights, fuse_neighbors]) #out_pn*max_neighbor_num*(out_channel*in_channel)


        out_pc = "" #batch*out_pn*out_channel        
        if b_max_pool:
            out_pc = out_neighbors.max(2)
        else:
            out_pc = out_neighbors.sum(2)
        
        out_pc = out_pc + bias
        
        if is_final_layer==False:
            out_pc = self.relu(out_pc) ##self.relu is defined in the init function
        
        
        if self.residual_rate==0:
            return out_pc
        
        if(in_channel != out_channel):
            in_pc_pad = torch.einsum('oi,bpi->bpo',[self.weight_res.view(out_channel,in_channel), in_pc_pad])

        out_pc_res = []
        if(in_pn == out_pn):
            out_pc_res = in_pc_pad[:,0:in_pn].clone()
        else:
            p_neighbors_raw = self.p_neighbors            
            in_neighbors = index_selection_nd(in_pc_pad,neighbor_id_lstlst, 1)

            #p_neighbors = torch.sigmoid(p_neighbors_raw) * neighbor_mask_lst
            p_neighbors = torch.abs(p_neighbors_raw)  * neighbor_mask_lst
            p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
            p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)        
            
            out_pc_res = torch.einsum('pn,bpno->bpo', [p_neighbors, in_neighbors])

        out_pc = out_pc*np.sqrt(1-self.residual_rate) + out_pc_res*np.sqrt(self.residual_rate)      

        return out_pc


    # original implementation, which takes more memory
    def forward2(self, in_pc, raw_w_weights, is_final_layer=False, b_max_pool = False): #layer_info
        batch = in_pc.shape[0]
        device = in_pc.device
        
        in_channel = self.in_channel
        out_channel = self.out_channel 
        in_pn = self.in_point_num
        out_pn = self.out_point_num 
        weight_num = self.weight_num  
        max_neighbor_num = self.max_neighbor_num
        neighbor_num_lst = self.neighbor_num_lst
        neighbor_id_lstlst = self.neighbor_id_lstlst
                
        pc_mask  = torch.ones(in_pn+1).float().to(device)
        pc_mask[in_pn]=0
        neighbor_mask_lst = pc_mask[neighbor_id_lstlst].contiguous() #out_pn*max_neighbor_num neighbor is 1 otherwise 0
        
        
        raw_weights = self.weights
        bias = self.bias        
        
        w_weights = raw_w_weights*neighbor_mask_lst.view(out_pn, max_neighbor_num, 1).repeat(1,1,weight_num) #out_pn*max_neighbor_num*weight_num
                
        normalized_weights = raw_weights          

        weights = torch.einsum('pmw,wc->pmc',[w_weights, normalized_weights]) #out_pn*max_neighbor_num*(out_channel*in_channel)
        weights = weights.view(out_pn, max_neighbor_num, out_channel,in_channel)
                
        in_pc_pad = torch.cat((in_pc, torch.zeros(batch, 1, in_channel).float().to(device)), 1) #batch*(in_pn+1)*in_channel
        
        in_neighbors = in_pc_pad[:, neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*in_channel
        
        out_neighbors = torch.einsum('pmoi,bpmi->bpmo',[weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel
        
        out_pc = ""
        if b_max_pool:
            out_pc = out_neighbors.max(2)
        else:
            out_pc = out_neighbors.sum(2)
        
        out_pc = out_pc + bias
        
        if is_final_layer==False:
            out_pc = self.relu(out_pc)
                
        if self.residual_rate==0:
            return out_pc        
                
        if(in_channel != out_channel):
            in_pc_pad = torch.einsum('oi,bpi->bpo',[self.weight_res.view(out_channel,in_channel), in_pc_pad])

        out_pc_res = []
        if(in_pn == out_pn):
            out_pc_res = in_pc_pad[:,0:in_pn].clone()
        else:
            p_neighbors_raw = self.p_neighbors
            in_neighbors = in_pc_pad[:,neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*out_channel
            
            p_neighbors = torch.abs(p_neighbors_raw)  * neighbor_mask_lst
            p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
            p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)
        
            out_pc_res = torch.einsum('pm,bpmo->bpo', [p_neighbors, in_neighbors])
        
        out_pc = out_pc*np.sqrt(1-self.residual_rate) + out_pc_res*np.sqrt(self.residual_rate)
        
        return out_pc



class MCEnc(nn.Module):    
    def __init__(self,structure, channel_lst,weight_num): #layer_info_lst= [(point_num, feature_dim)]
        super(MCEnc, self).__init__()

        self.point_num = structure.point_num
        self.residual_rate = structure.residual_rate
        
        self.b_max_pool = structure.b_max_pool
        self.perpoint_bias = structure.perpoint_bias
        
        self.channel_lst = channel_lst           
        self.layer_num = len(structure.connection_info_lsts)
                
        self.layer_lst = nn.ModuleList([])
        

        b_Perpt_bias = self.perpoint_bias
        for l in np.arange(0,self.layer_num):
            in_channel = self.channel_lst[l]
            out_channel = self.channel_lst[l+1]
            connection_info  = structure.connection_info_lsts[l]

            in_point_num = structure.ptnum_list[l]
            
            self.layer_lst.append(LASMConvssw(in_channel, out_channel, weight_num,in_point_num,  connection_info, b_Perpt_bias, self.residual_rate))           


            if l == self.layer_num-1:
                self.stdlayer = LASMConvssw(in_channel, out_channel, weight_num,in_point_num,  connection_info, b_Perpt_bias, self.residual_rate)


        self.out_nrpts = structure.ptnum_list[self.layer_num]
        self.out_nrchs = out_channel

        print(self.layer_num, self.out_nrpts, self.out_nrchs)        

        


    def forward_till_layer_n(self,in_pc,vcoeffs, layer_n):
        out_pc = in_pc.clone()       
        for i in range(layer_n):                        
            out_pc = self.layer_lst[i](out_pc,vcoeffs.vcoeffs_list[i], is_final_layer = False, b_max_pool = self.b_max_pool)
        return out_pc   
  
        
    def forward(self, in_pc, vcoeffs):        
        tmpcode = self.forward_till_layer_n(in_pc, vcoeffs, self.layer_num-1)        
        mu = self.layer_lst[self.layer_num-1](tmpcode,vcoeffs.vcoeffs_list[self.layer_num-1], is_final_layer = True, b_max_pool = self.b_max_pool) * 0.1        
        std = self.stdlayer(tmpcode,vcoeffs.vcoeffs_list[self.layer_num-1], is_final_layer = True, b_max_pool = self.b_max_pool) * 0.01
        
        return mu, std

    





class MCDec(nn.Module):
    def __init__(self, structure, channel_lst,weight_num): #layer_info_lst= [(point_num, feature_dim)]
        super(MCDec, self).__init__()
        
        self.point_num = structure.point_num
        self.residual_rate = structure.residual_rate
        
        self.b_max_pool = structure.b_max_pool
        self.perpoint_bias = structure.perpoint_bias
        
        self.channel_lst = channel_lst           
        self.layer_num = len(structure.connection_info_lsts)
        
        self.layer_lst = nn.ModuleList([])
        

        b_Perpt_bias = self.perpoint_bias
        for l in np.arange(0,self.layer_num):
            in_channel = self.channel_lst[l]
            out_channel = self.channel_lst[l+1]
            connection_info  = structure.connection_info_lsts[l]

            in_point_num = structure.ptnum_list[l]
            
            self.layer_lst.append(LASMConvssw(in_channel, out_channel, weight_num,in_point_num,  connection_info, b_Perpt_bias, self.residual_rate))           

        print(self.layer_num)

    
    def forward(self, latent, vcoeffs): 
        out_pc = self.forward_from_layer_n(latent,vcoeffs, 0)
        return out_pc


    def forward_dec(self, latent,vcoeffs):     
        out_pc = self.forward_from_layer_n(latent,vcoeffs, 0)
        return out_pc
    
    
    def forward_from_layer_n(self, in_pc,vcoeffs, layer_n):
        out_pc = in_pc.clone()
        for i in range(layer_n, self.layer_num):
            if(i<(self.layer_num-1)):                
                out_pc = self.layer_lst[i](out_pc,vcoeffs.vcoeffs_list[i], is_final_layer = False, b_max_pool = self.b_max_pool)
            else:                
                out_pc = self.layer_lst[i](out_pc,vcoeffs.vcoeffs_list[i], is_final_layer = True, b_max_pool = self.b_max_pool)
               
        return out_pc
    
    
class MCStructure(nn.Module):
    def __init__(self, param, inptnr, weight_num, bDec= True, b_perpoint_bias = True): #layer_info_lst= [(point_num, feature_dim)]
        super(MCStructure, self).__init__()
        
        self.point_num = inptnr
        
        self.residual_rate = param.residual_rate        
        self.b_max_pool = param.conv_max        
        self.perpoint_bias = b_perpoint_bias #param.perpoint_bias        
        
        if bDec:
            self.connection_layer_fn_lst = param.connection_layer_fn_lst_dec
        else:
            self.connection_layer_fn_lst = param.connection_layer_fn_lst_enc
        self.layer_num = len(self.connection_layer_fn_lst)

        self.ptnum_list = []
        self.ptnum_list += [inptnr]         
        self.connection_info_lsts = []  
        for l in np.arange(0,self.layer_num):           
            print ("##Layer",self.connection_layer_fn_lst[l])            

            connection_info  = np.load(self.connection_layer_fn_lst[l])
            out_point_num = connection_info.shape[0]
            self.connection_info_lsts += [connection_info]
            self.ptnum_list += [out_point_num]  

    def forward(self):
        return 


class MCVcoeffs(nn.Module):
    def __init__(self, structure, weight_num): #layer_info_lst= [(point_num, feature_dim)]
        super(MCVcoeffs, self).__init__()


        self.layer_num = len(structure.connection_layer_fn_lst)
        self.vcoeffs_list = nn.ParameterList([])
        for l in np.arange(0,self.layer_num):   
            connection_info = structure.connection_info_lsts[l]          
            out_point_num = connection_info.shape[0]
            
            neighbor_num_lst = torch.from_numpy(connection_info[:,0].astype(np.float32)).float() #out_point_num*1            
        
            neighbor_id_dist_lstlst = connection_info[:, 1:] #out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1,2))[:,:,0] #out_point_num*max_neighbor_num
            
            neighbor_id_lstlst = torch.from_numpy(neighbor_id_lstlst).long()
            max_neighbor_num = neighbor_id_lstlst.shape[1]
        
            avg_neighbor_num= round(neighbor_num_lst.mean().item())

            w_weights=torch.randn(out_point_num, max_neighbor_num, weight_num)/(avg_neighbor_num*weight_num)            
            w_weights = nn.Parameter(w_weights)                
            self.vcoeffs_list.append(w_weights) #+= [w_weights]



        
class MCLoss(nn.Module):
    def __init__(self,param): #layer_info_lst= [(point_num, feature_dim)]
        super(MCLoss, self).__init__()
        
        self.register_buffer('initial_neighbor_id_lstlst', torch.LongTensor(param.neighbor_id_lstlst))
        self.register_buffer('initial_neighbor_num_lst', torch.FloatTensor(param.neighbor_num_lst))        
        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]        

    def forward(self):
        return
    
    def compute_geometric_loss_l1(self, gt_pc, predict_pc,weights=[]):
        
        if(len(weights)==0):
            loss = torch.abs(gt_pc-predict_pc).mean()
        
            return loss

        else:
            batch =gt_pc.shape[0]
            point_num=gt_pc.shape[1]
            channel = gt_pc.shape[2]            
            pc_weights = weights.view(batch, point_num,1).repeat(1,1,channel)
            
            loss = ((gt_pc- predict_pc).abs()*weights).sum()/(weights.sum()+1e-6)

            return loss

    
    def compute_geometric_loss_l2(self, gt_pc, predict_pc,weights=[]):
        
        if(len(weights)==0):
            loss = (gt_pc-predict_pc).pow(2).sum(2).mean()
        
            return loss

        else:
            batch =gt_pc.shape[0]
            point_num=gt_pc.shape[1]
            channel = gt_pc.shape[2]
            pc_weights = weights.view(batch, point_num,1)
            
            loss = ((gt_pc- predict_pc).pow(2).sum(2,keepdim=True)*weights).sum()/(weights.sum()+1e-6)

            return loss
    
    def compute_geometric_mean_euclidean_dist_error(self, gt_pc, predict_pc,weights=[]):
        
        if(len(weights)==0):
            error = (gt_pc-predict_pc).pow(2).sum(2).pow(0.5).mean()
        
            return error

        else:
            batch =gt_pc.shape[0]
            point_num=gt_pc.shape[1]
            channel = gt_pc.shape[2]

            dists = (gt_pc-predict_pc).pow(2).sum(2).pow(0.5) * weights
            error = dists.sum()

            return error     
    
    
    def compute_laplace_loss_l1(self, gt_pc_raw, predict_pc_raw, weights=[]):
        gt_pc = gt_pc_raw*1
        predict_pc = predict_pc_raw*1
        
        batch = gt_pc.shape[0]
        point_num = gt_pc.shape[1]
        device = gt_pc_raw.device
        
        gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        
        batch = gt_pc.shape[0]
        
        gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        gt_pc_laplace = gt_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):            
            neighbor = gt_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            gt_pc_laplace -= neighbor
        
        
        
        predict_pc_laplace = predict_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        predict_pc_laplace = predict_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):            
            neighbor = predict_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            predict_pc_laplace -= neighbor
        
        
        if(len(weights)==0):
            loss_l1 = torch.abs(gt_pc_laplace - predict_pc_laplace).mean()
        else:            
            loss_l1 = ((gt_pc_laplace - predict_pc_laplace).abs() *weights).sum()/(weights.sum()+1e-6)                
        
        return loss_l1

    
    def compute_laplace_loss_l2(self, gt_pc_raw, predict_pc_raw, weights=[]):
        gt_pc = gt_pc_raw
        predict_pc = predict_pc_raw
        
        batch = gt_pc.shape[0]
        point_num = gt_pc.shape[1]
        device = gt_pc_raw.device
        
        gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        
        batch = gt_pc.shape[0]
        
        gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        gt_pc_laplace = gt_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):            
            neighbor = gt_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            gt_pc_laplace -= neighbor
        
        
        
        predict_pc_laplace = predict_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        predict_pc_laplace = predict_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):            
            neighbor = predict_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            predict_pc_laplace -= neighbor
        
        
        if(len(weights)==0):
            loss_l1 = (gt_pc_laplace - predict_pc_laplace).pow(2).sum(2).mean()
        else:            
            loss_l1 = ((gt_pc_laplace - predict_pc_laplace).pow(2) *weights).sum()/(weights.sum()+1e-6)                
        
        return loss_l1 #, loss_curv
    
        
    def compute_laplace_Mean_Euclidean_Error(self, gt_pc_raw, predict_pc_raw):
        gt_pc = gt_pc_raw*1
        predict_pc = predict_pc_raw*1
        
        batch = gt_pc.shape[0]
        point_num = gt_pc.shape[1]
        device = gt_pc_raw.device
        
        gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).float().to(device)), 1)
        
        batch = gt_pc.shape[0]
        
        gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        gt_pc_laplace = gt_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):            
            neighbor = gt_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            gt_pc_laplace -= neighbor
        
        predict_pc_laplace = predict_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        predict_pc_laplace = predict_pc_laplace*self.initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1,3)
        
        for n in range(1, self.initial_max_neighbor_num):
            neighbor = predict_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            predict_pc_laplace -= neighbor
        
            
        error  = torch.pow(torch.pow(gt_pc_laplace - predict_pc_laplace,2).sum(2), 0.5).mean()
        
        return error 

