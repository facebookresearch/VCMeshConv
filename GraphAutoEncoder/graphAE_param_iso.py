"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tensorboardX
import configparser
import os
import numpy as np

import json

class Parameters():
    def __init__(self):     
        super(Parameters, self).__init__()
      
        
    def read_config(self, fn):
        config = configparser.ConfigParser()
        config.read(fn)
        
        self.read_weight_path = config.get("Record","read_weight_path")
        self.write_weight_folder=config.get("Record","write_weight_folder")
        self.write_tmp_folder=config.get("Record","write_tmp_folder")
        if not os.path.exists(self.write_weight_folder):
            os.makedirs(self.write_weight_folder)
        if not os.path.exists(self.write_tmp_folder):
            os.makedirs(self.write_tmp_folder)
    
        logdir = config.get("Record","logdir")
        self.logger = tensorboardX.SummaryWriter(logdir)
        
        
        self.lr =float( config.get("Params", "lr"))
        self.batch=int( config.get("Params", "batch"))
        
        #self.normalize_data = int(config.get("Params", "normalize_data"))
        
        self.augmented_data = int(config.get("Params","augment_data"))
        
        self.start_iter=int(config.get("Params","start_iter"))
        self.end_iter=int( config.get("Params", "end_iter"))
        self.evaluate_iter = int( config.get("Params", "evaluate_iter"))
        self.save_weight_iter=int( config.get("Params", "save_weight_iter"))
        self.save_tmp_iter=int( config.get("Params", "save_tmp_iter"))
        
        self.w_pose = float(config.get("Params", "w_pose"))
        self.w_laplace = float(config.get("Params", "w_laplace"))
        self.w_color = float(config.get("Params", "w_color"))
        self.w_w_weights_l1 = float(config.get("Params","w_w_weights_l1" ))
        
        
        self.pcs_train = config.get("Params", "pcs_train")
        self.pcs_evaluate = config.get("Params", "pcs_evaluate")
        self.pcs_mean = config.get("Params", "pcs_mean")

        self.template_ply_fn = config.get("Params", "template_ply_fn")
        self.template_obj_fn = config.get("Params", "template_obj_fn")
        self.mesh_train = config.get("Params", "mesh_train")
        self.recon_train = config.get("Params", "recon_train")      
        self.calib_path = config.get("Params", "calib_path")     
        self.framelist = config.get("Params", "frame_list")


        self.point_num = int(config.get("Params", "point_num"))
        
        self.residual_rate = float(config.get("Params","residual_rate"))
        self.conv_max = int(config.get("Params","conv_max"))
        self.perpoint_bias = int(config.get("Params","perpoint_bias"))
        
        self.minus_smoothed = int(config.get("Params", "minus_smoothed"))



        self.freeze_decoder=0
        self.freeze_start_layer=1234567
        try:
            self.freeze_decoder = int(config.get("Params","freeze_decoder"))
            self.freeze_start_layer=int (config.get("Params","freeze_start_layer"))
        except:
            pass

    
        #self.connection_lst_fn = config.get("Params", "connection_lst_fn")
        self.initial_connection_fn = config.get("Params", "initial_connection_fn")
        data = np.load(self.initial_connection_fn)
        neighbor_id_dist_lstlst = data[:, 1:] # point_num*(1+2*neighbor_num) 
        self.point_num = data.shape[0]
        self.neighbor_id_lstlst= neighbor_id_dist_lstlst.reshape((self.point_num, -1,2))[:,:,0] #point_num*neighbor_num
        self.neighbor_num_lst = np.array(data[:,0])  #point_num
        
        self.connection_folder = config.get("Params", "connection_folder")
        self.connection_layer_lst_dec = json.loads(config.get("Params", "connection_layer_lst_dec"))
        self.connection_layer_lst_enc = json.loads(config.get("Params", "connection_layer_lst_enc"))
        ##load neighborlstlst_fn_lst 
        
        fn_lst = os.listdir(self.connection_folder)
        self.connection_layer_fn_lst_enc = []        
        for layer_name in self.connection_layer_lst_enc:
            layer_name = "_"+layer_name+"."
            
            find_fn = False
            for fn in fn_lst:
                if((layer_name in fn) and ((".npy" in fn) or (".npz" in fn))):
                    
                    self.connection_layer_fn_lst_enc +=[self.connection_folder+fn]
                    find_fn = True
                    break
            if(find_fn ==False):
                print ("!!!ERROR: cannot find the connection layer fn")
        
        self.connection_layer_fn_lst_dec = []        
        for layer_name in self.connection_layer_lst_dec:
            layer_name = "_"+layer_name+"."
            
            find_fn = False
            for fn in fn_lst:
                if((layer_name in fn) and ((".npy" in fn) or (".npz" in fn))):
                    
                    self.connection_layer_fn_lst_dec +=[self.connection_folder+fn]
                    find_fn = True
                    break
            if(find_fn ==False):
                print ("!!!ERROR: cannot find the connection layer fn")



        self.channel_lst_enc = json.loads(config.get("Params", "channel_lst_enc"))
        self.channel_lst_dec = json.loads(config.get("Params", "channel_lst_dec"))
        self.weight_num_lst_enc = json.loads(config.get("Params", "weight_num_lst_enc"))
        self.weight_num_lst_dec = json.loads(config.get("Params", "weight_num_lst_dec"))
        





































