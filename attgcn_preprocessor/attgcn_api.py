
import torch
import torch.nn.functional as F
import torch.nn as nn

import sys
# sys.path.insert(1, 'attgcn_preprocessor')
import os

import json
import argparse
import time
import math
import numpy as np
from scipy.special import softmax

from attgcn_preprocessor.config import *
from sklearn.utils.class_weight import compute_class_weight

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from attgcn_preprocessor.data.data_generator import tsp_instance_reader

from attgcn_preprocessor.utils.process import *
from attgcn_preprocessor.utils.tsplib import read_tsplib_coor, read_tsplib_opt, write_tsplib_prob
from attgcn_preprocessor.utils.test_utils import partition_one_graph, multiprocess
from attgcn_preprocessor.utils.plot_utils import plot_predictions_cluster

from multiprocessing import Pool
from multiprocessing import cpu_count

from tqdm import tqdm

class ATTGCN_API():
    def __init__(self, config_path="../attgcn_preprocessor/configs/tsp20.json"):
        self.config = get_config(config_path)

    def init_net(self):
        # setting random seed to 1
        if torch.cuda.is_available():
            dtypeFloat = torch.cuda.FloatTensor
            dtypeLong = torch.cuda.LongTensor
            torch.cuda.manual_seed_all(1)
            print("Using CUDA!")
        else:
            dtypeFloat = torch.FloatTensor
            dtypeLong = torch.LongTensor
            torch.manual_seed(1)

        # Instantiate the network
        self.net = nn.DataParallel(ResidualGatedGCNModel(self.config, dtypeFloat, dtypeLong))
        if torch.cuda.is_available():
            self.net.cuda()
        # Define optimizer
        self.learning_rate = self.config.learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate) 

    def load_ckpt(self):
        # Load checkpoint
        log_dir = f"../attgcn_preprocessor/logs/{self.config.expt_name}"
        if torch.cuda.is_available():
            # TSP-20
            checkpoint = torch.load(f"{log_dir}/best_val_checkpoint.tar")
        else:
            checkpoint = torch.load(f"{log_dir}/best_val_checkpoint.tar", map_location='cpu')
        # Load network state
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load other training parameters
        self.epoch = checkpoint['epoch']
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
        for param_group in self.optimizer.param_groups:
            self.learning_rate = param_group['lr']  
    
    def clear_gpu_memory(self):
        torch.cuda.empty_cache()

    def switch_to_eval_mode(self):
        self.net.eval()

    def load_nx_test_set(self, graph_list, num_nodes):
        assert(graph_list[0].number_of_nodes() == num_nodes)
        # extract graph coordinates
        self.test_set = [np.array([g.nodes()[i]['coord'] for i in range(num_nodes)], dtype=np.float64) for g in graph_list]
        self.test_set_num_nodes = num_nodes
    

    def run_test(self, batch_size):
        if batch_size > len(self.test_set):
            batch_size = len(self.test_set)
        self.net.eval()
        result = []
        num_nodes = self.test_set_num_nodes
        K = num_nodes - 1
        avg_mean_rank = [] 
        top_k, cluster_center = K, 0
        if num_nodes <= 20:
            threshold = 1
        else:
            threshold = math.ceil((num_nodes / (top_k+1) ) * 5)
        
        epoch = int(len(self.test_set)/batch_size)
        start_row_num = 0

        if num_nodes <= 20:
            K_expand = K
        elif num_nodes == 50:
            K_expand = 29
        # init
        count_buff = np.zeros(shape=(batch_size*threshold, ), dtype=np.int32)
        edges = np.zeros(shape=(batch_size*threshold, K+1, K+1), dtype=np.int32)
        edges_values = np.zeros(shape=(batch_size*threshold, K+1, K+1), dtype=np.float16)
        nodes = np.zeros(shape = (batch_size*threshold, K+1), dtype=np.int32)
        nodes_coord = np.zeros(shape = (batch_size*threshold, K+1, 2), dtype=np.float16)
        edges_target = np.zeros(shape = (batch_size*threshold, K+1, K+1), dtype=np.int32)
        nodes_target = np.zeros(shape = (batch_size*threshold, K+1), dtype=np.int32)
        meshs = np.zeros(shape = (batch_size*threshold, 2, K+1, K+1), dtype=np.int32)
        Omegas = np.zeros(shape = (batch_size, num_nodes, num_nodes), dtype=np.int32)
        opts = np.zeros(shape = (batch_size, num_nodes+1), dtype=np.int32)

        #start = time.time()
        sum_time = 0
        for j in tqdm(range(epoch)):
            start = time.time()
            for i in range(batch_size):
                edge, edges_value, node, node_coord, edge_target, node_target, mesh, omega = partition_one_graph(coor=self.test_set[start_row_num+i], 
                                                                                                                 node_num=num_nodes, 
                                                                                                                 cluster_center=0, 
                                                                                                                 top_k=K, top_k_expand=K_expand)
                edges[i*threshold:(i+1)*threshold, ...] = edge
                edges_values[i*threshold:(i+1)*threshold, ...] = edges_value
                nodes[i*threshold:(i+1)*threshold, ...] = node
                nodes_coord[i*threshold:(i+1)*threshold, ...] = node_coord
                edges_target[i*threshold:(i+1)*threshold, ...] = edge_target
                nodes_target[i*threshold:(i+1)*threshold, ...] = node_target
                meshs[i*threshold:(i+1)*threshold, ...] = mesh
                Omegas[i, ...] = omega
                opts[i, ...] = 0

            with torch.no_grad():
                # Convert batch to torch Variables
                x_edges = Variable(torch.LongTensor(edges).type(dtypeLong), requires_grad=False)
                x_edges_values = Variable(torch.FloatTensor(edges_values).type(dtypeFloat), requires_grad=False)
                x_nodes = Variable(torch.LongTensor(nodes).type(dtypeLong), requires_grad=False)
                x_nodes_coord = Variable(torch.FloatTensor(nodes_coord).type(dtypeFloat), requires_grad=False)
                y_edges = Variable(torch.LongTensor(edges_target).type(dtypeLong), requires_grad=False)
                y_nodes = Variable(torch.LongTensor(nodes_target).type(dtypeLong), requires_grad=False)

                # Compute class weights
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

                # Forward pass
                y_preds, loss = self.net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
                y_preds_prob = F.softmax(y_preds, dim=3)
                y_preds_prob_numpy = y_preds_prob.cpu().numpy()

            end = time.time()
            sum_time += end - start
            # single - process
            for i in range(batch_size):
                edges_probs_norm = multiprocess(y_preds_prob_numpy[i*threshold:(i+1)*threshold, ...],
                                                meshs[i*threshold:(i+1)*threshold, ...], Omegas[i, ...],
                                                num_nodes)
                result.append(edges_probs_norm)
            start_row_num += batch_size

            del x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes, edge_labels
        return result