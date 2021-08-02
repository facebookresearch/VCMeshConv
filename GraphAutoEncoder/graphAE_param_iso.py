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

    def read_config(self, file_names):
        config = configparser.ConfigParser()
        config.read(file_names)

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


        self.mesh_train = config.get("Params", "mesh_train")

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

        #self.connection_list_file_names = config.get("Params", "connection_list_file_names")
        self.initial_connection_file_names = config.get("Params", "initial_connection_file_names")
        data = np.load(self.initial_connection_file_names)
        neighbor_id_dist_list = data[:, 1:] # point_num*(1+2*neighbor_num)
        self.point_num = data.shape[0]
        self.neighbor_id_list= neighbor_id_dist_list.reshape((self.point_num, -1,2))[:,:,0] #point_num*neighbor_num
        self.neighbor_num_list = np.array(data[:,0])  #point_num

        self.connection_folder = config.get("Params", "connection_folder")
        self.connection_layer_list_dec = json.loads(config.get("Params", "connection_layer_list_dec"))
        self.connection_layer_list_enc = json.loads(config.get("Params", "connection_layer_list_enc"))
        ##load neighborlstlst_file_name_list

        file_names_list = os.listdir(self.connection_folder)
        self.connection_layer_file_name_list_enc = []
        for layer_name in self.connection_layer_list_enc:
            layer_name = "_"+layer_name+"."

            find_file_names = False
            for file_names in file_names_list:
                if((layer_name in file_names) and ((".npy" in file_names) or (".npz" in file_names))):

                    self.connection_layer_file_name_list_enc +=[self.connection_folder+file_names]
                    find_file_names = True
                    break
            if(find_file_names ==False):
                print ("!!!ERROR: cannot find the connection layer file_names")

        self.connection_layer_file_name_list_dec = []
        for layer_name in self.connection_layer_list_dec:
            layer_name = "_"+layer_name+"."

            find_file_names = False
            for file_names in file_names_list:
                if((layer_name in file_names) and ((".npy" in file_names) or (".npz" in file_names))):

                    self.connection_layer_file_name_list_dec +=[self.connection_folder+file_names]
                    find_file_names = True
                    break
            if(find_file_names == False):
                print ("!!!ERROR: cannot find the connection layer file_names")
