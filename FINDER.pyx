#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import tsplib95
import DQN_builder
import scipy.linalg as linalg
from scipy.sparse import csr_matrix
import os
from itertools import combinations

#################################### Hyper Parameters start ####################################

cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000 # when to take snapshots
cdef int EMBEDDING_SIZE = 64 # dimension p for each vector embedding of the nodes in the graph
cdef int MAX_ITERATION = 120000
cdef double LEARNING_RATE = 0.0001
cdef double eps_start = 1.0
cdef double eps_end = 0.05
cdef double eps_step = 10000.0  
cdef int MEMORY_SIZE = 150000
cdef double Alpha = 0.001 ## weight of reconstruction loss, default value 0.001
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5 # number of steps in NDQN until the reward is observed
cdef int NUM_MIN = 10 # min dim of training graphs
cdef int NUM_MAX = 20 # max dim of training graphs
cdef int save_interval = 300 # number of iterations after which the model is tested and saved
cdef int REG_HIDDEN = 32
cdef int BATCH_SIZE = 64 # Batch size for training
cdef double initialization_stddev = 0.01  # variance of weight initilization
cdef int n_valid = 200 # number of graphs in validation set (originally 200)
cdef int n_generator = 1000 # number of graphs for each training graph generation cycle
cdef int num_env = 1
cdef double inf = 2147483647/2
cdef double valid_scale_fac = 1/10000 # sets the factor of scaling that is applied to all external validation samples --> transfo into (0,1) square
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3 # number of aggregation steps in GNN = number of layers
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN; 3:edge weight based sum
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage
cdef int node_init_dim = 4 # number of initial node features
cdef int edge_init_dim = 1 # number of initial edge features
cdef int state_init_dim = 4 # number of initial state features
cdef int edge_embed = 1 # 0: no inclusion of edge weights, 1: inclusion of sum of edge weights per node, 2: inclusion of edge features/each specific edge weight
cdef int help_func = 1 # whether to use helper function during node insertion into current partial tour
cdef int ignore_covered_edges = 1 # 0: False, 1: True
cdef int include_selected_nodes = 1 # 0: No inclusion of selected nodes, 1: start and last selected node inclusion, 2: all selected nodes included
#################################### Hyper Parameters end ####################################

class FINDER:

    def __init__(self):
        # init some parameters
        self.node_embed_dim = EMBEDDING_SIZE
        self.edge_embed_dim = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'tsp_2d' #erdos_renyi, powerlaw, small-world， barabasi_albert
        self.valid_path = 'valid_sets/synthetic_nrange_10_20_200/'
        self.valid_sol = 0.23525577276
        self.TrainSet = graph.py_GSet() # initializes the training and test set object
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)

        ############----------------------------- variants of DQN(end) ------------------- ###################################
        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, edge_cnt]
        self.e2nsum_param = tf.sparse_placeholder(tf.float32, name="e2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf.placeholder(tf.float32, [BATCH_SIZE,1], name="target") # DQN target 
        # [batch_size, aux_dim]
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")
        # [node_cnt, node_init_dim], per node [node x pos, node y pos, 1]
        self.node_input = tf.placeholder(tf.float32, name="node_input")
        # [node_cnt, edge_init_dim], sum of the edgeweights of all adjacent edges per node
        self.edge_sum = tf.placeholder(tf.float32, name="edge_sum")

        self.edge_input = tf.placeholder(tf.float32, name="edge_input")

        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')

        # init Q network
        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildNet() #[loss,trainStep,q_pred, q_on_all, ...]
        #init Target Q Network
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.BuildNet()
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT, self.Q_param_list)]


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.global_variables_initializer())


################################################# New code for FINDER #################################################
###################################################### BuildNet start ######################################################    
    def BuildNet(self):
        """
        Builds the Tensorflow network, returning different options to access it
        """
        # [node_init_dim, node_embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([node_init_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)

        # [node_init_dim, node_embed_dim], state input embedding matrix
        w_s2l = tf.Variable(tf.truncated_normal([state_init_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)

        # Define weight matrices for GNN 
        # [node_embed_dim, node_embed_dim]
        p_node_conv = tf.Variable(tf.truncated_normal([self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32) 
        
        # [node_embed_dim, node_embed_dim]
        p_node_conv2 = tf.Variable(tf.truncated_normal([self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32) 
        
        if edge_embed == 1:
            # [edge_init_dim, edge_embed_dim]
            w_e2l = tf.Variable(tf.truncated_normal([edge_init_dim, self.edge_embed_dim], stddev=initialization_stddev), tf.float32)
            # [node_embed_dim, node_embed_dim]
            w_edge_final = tf.Variable(tf.truncated_normal([self.edge_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)
            # [3*node_embed_dim, node_embed_dim]
            p_node_conv3 = tf.Variable(tf.truncated_normal([3*self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)
        else:
            # [2*node_embed_dim, node_embed_dim]
            p_node_conv3 = tf.Variable(tf.truncated_normal([2*self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)
        
        # define weight matrices for state embedding
        # [node_embed_dim, node_embed_dim]
        # p_state_conv = tf.Variable(tf.truncated_normal([self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32) 
        p_state_conv = p_node_conv

        # [node_embed_dim, node_embed_dim]
        # p_state_conv2 = tf.Variable(tf.truncated_normal([self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)
        p_state_conv2 = p_node_conv2

        # [2*node_embed_dim, node_embed_dim]
        p_state_conv3 = tf.Variable(tf.truncated_normal([2*self.node_embed_dim, self.node_embed_dim], stddev=initialization_stddev), tf.float32)

        #[reg_hidden, 1]
        if self.reg_hidden > 0:
            # [node_embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([self.node_embed_dim, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            
            #[reg_hidden, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden, 1], stddev=initialization_stddev), tf.float32)
            
            #[reg_hidden, 1]
            last_w = h2_weight

        # [node_embed_dim, 1]
        cross_product = tf.Variable(tf.truncated_normal([self.node_embed_dim, 1], stddev=initialization_stddev), tf.float32)
        

        # [node_cnt, node_init_dim]
        node_input = self.node_input

        # [batch_size, node_init_dim]
        y_nodes_size = tf.shape(self.subgsum_param)[0]
        y_node_input = tf.ones((y_nodes_size, state_init_dim))
        
        # [node_cnt, node_init_dim] * [node_init_dim, node_embed_dim] = [node_cnt, node_embed_dim], not sparse
        input_message = tf.matmul(tf.cast(self.node_input, tf.float32), w_n2l)  
        
        # [batch_size, node_init_dim] * [node_init_dim, node_embed_dim] = [batch_size, node_embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input, tf.float32), w_s2l)
        
        # [node_cnt, node_embed_dim], not sparse
        input_potential_layer = tf.nn.relu(input_message)
        
        # [batch_size, node_embed_dim], not sparse
        y_input_potential_layer = tf.nn.relu(y_input_message)

        # [node_cnt, node_embed_dim], not sparse
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        # [batch_size, node_embed_dim], not sparse
        y_cur_message_layer = y_input_potential_layer
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        
        if edge_embed == 1:
            # [node_cnt, 1]
            edge_input = self.edge_sum
            
            # [node_cnt, 1] * [1, node_embed_dim] = [node_cnt, node_embed_dim], dense
            edge_input_message = tf.matmul(tf.cast(edge_input, tf.float32), w_e2l, name='edge_init')

            # [node_cnt, node_embed_dim]
            edge_input_potential_layer = tf.nn.relu(edge_input_message)
            
            # [node_cnt, node_embed_dim], not sparse
            norm_edge_input = tf.nn.l2_normalize(edge_input_potential_layer, axis=1)
            
            # [node_cnt, node_embed_dim] * [node_embed_dim, node_embed_dim] = [node_cnt, node_embed_dim], dense
            edge_message = tf.nn.relu(tf.matmul(norm_edge_input, w_edge_final))
            
        ################### GNN start ###################
        cdef int lv = 0
        while lv < max_bp_iter:
            lv = lv + 1
            
            # [node_cnt, node_cnt] * [node_cnt, node_embed_dim] = [node_cnt, node_embed_dim], dense
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float32), cur_message_layer) 

            # [node_cnt, node_embed_dim] * [node_embed_dim, node_embed_dim] = [node_cnt, node_embed_dim], dense
            node_linear = tf.nn.relu(tf.matmul(n2npool, p_node_conv))
        
            # [node_cnt, node_embed_dim] * [node_embed_dim, node_embed_dim] = [node_cnt, node_embed_dim], dense
            cur_message_layer_linear = tf.nn.relu(tf.matmul(cur_message_layer, p_node_conv2))
             
            
            if edge_embed == 1:
                # [[node_cnt, node_embed_dim] [node_cnt, node_embed_dim] [node_cnt, node_embed_dim]] = [node_cnt, 3*node_embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear, edge_message], axis=1)
            else:
                # [[node_cnt, node_embed_dim] [node_cnt, node_embed_dim]] = [node_cnt, 2*node_embed_dim], return dense matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], axis=1)
            
            
            # [batch_size, node_cnt] * [node_cnt, node_embed_dim] = [batch_size, node_embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param, tf.float32), cur_message_layer)
            
            # [batch_size, node_embed_dim] * [node_embed_dim, node_embed_dim] = [batch_size, node_embed_dim], dense
            y_node_linear = tf.nn.relu(tf.matmul(y_n2npool, p_state_conv))
            
            # [batch_size, node_embed_dim] * [node_embed_dim, node_embed_dim] = [batch_size, node_embed_dim], dense
            y_cur_message_layer_linear = tf.nn.relu(tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_state_conv2))
            
            
            # [node_cnt, 2(3)*node_embed_dim]*[2(3)*node_embed_dim, node_embed_dim] = [node_cnt, node_embed_dim]
            cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))
            
            # [node_cnt, node_embed_dim]
            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)           

            # [[batch_size, node_embed_dim] [batch_size, node_embed_dim]] = [batch_size, 2*node_embed_dim], return tensed matrix
            y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], axis=1)
            
            # [batch_size, 2(3)*node_embed_dim]*[2(3)*node_embed_dim, node_embed_dim] = [batch_size, node_embed_dim]
            y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_state_conv3))

            # [batch_size, node_embed_dim]
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        ################### GNN end ###################

        # [batch_size, node_embed_dim]
        y_potential = y_cur_message_layer
        
        # [batch_size, node_cnt] * [node_cnt, node_embed_dim] = [batch_size, node_embed_dim]
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
        
        # [batch_size, node_embed_dim, 1] * [batch_size, 1, node_embed_dim] = [batch_size, node_embed_dim, node_embed_dim]
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2), tf.expand_dims(y_potential, axis=1))
     
        # [batch_size, node_embed_dim]
        Shape = tf.shape(action_embed)
        
        # ([batch_size, node_embed_dim, node_embed_dim] * ([batch_size*node_embed_dim, 1] --> [batch_size, node_embed_dim, 1])) --> [batch_size, node_embed_dim] = [batch_size, node_embed_dim], first transform
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])), Shape) 
        
        last_output = embed_s_a

        if self.reg_hidden > 0:
            # [batch_size, node_embed_dim] * [node_embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            
            # [batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # [batch_size, reg_hidden]
        last_output = last_output
        
        # [batch_size, reg_hidden] * [reg_hidden, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        # Trace([node_embed_dim, node_cnt] * [node_cnt, node_cnt] * [node_cnt, node_embed_dim]) first order reconstruction loss
        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer)))
        
        # needs refinement
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))
        
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)
        # calculate full loss
        loss = loss_rl + Alpha * loss_recons

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        # This part only gets executed if tf.session.run([self.q_on_all]) or tf.session.run([self.q_on_allT])

        # [node_cnt, batch_size] * [batch_size, node_embed_dim] = [node_cnt, node_embed_dim] 
        # the result is a matrix where each row holds the state embedding of the corresponding graph and each row corresponds to a node of a graph in the batch
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)

        # [node_cnt, node_embed_dim, 1] * [node_cnt, 1, node_embed_dim] = [node_cnt, node_embed_dim, node_embed_dim]
        temp_1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2), tf.expand_dims(rep_y, axis=1))

        # [node_cnt node_embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        
        # [batch_size, node_embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp_1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        # [node_cnt, 2 * node_embed_dim]
        last_output = embed_s_a_all
        
        if self.reg_hidden > 0:
            # [node_cnt, node_embed_dim] * [node_embed_dim, reg_hidden] = [node_cnt, reg_hidden], dense
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            
            # [node_cnt, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # [node_cnt, reg_hidden]
        last_output = last_output

        # [node_cnt, reg_hidden] * [reg_hidden, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables()
###################################################### BuildNet end ######################################################


    def Train(self):
        self.PrepareValidData()
        self.gen_new_train_graphs(NUM_MIN, NUM_MAX)

        cdef int i, iter, idx
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        cdef int loss = 0
        cdef double frac, start, end
 
        #save_dir = './models/%s'%self.g_type
        save_dir = './models/{}'.format(self.g_type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            print("Iteration: ", iter)
            start = time.clock()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                print("generating new traning graphs")
                self.gen_new_train_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            print("Epsilon:", eps)
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                test_start = time.time()
                for idx in range(n_valid):
                    if self.valid_sol:
                        frac += self.Test(idx)/self.valid_sol
                    else:
                        frac += self.Test(idx)
                test_end = time.time()
                # print("Test finished, slepping for 5 seconds...")
                # time.sleep(5)
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush()
                print('iter %d, eps %.4f, average tour length: %.6f'%(iter, eps, frac/n_valid))
                print ('testing 200 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.2fs\n'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            # print("Fitting in 5 seconds...")
            # time.sleep(5)
            self.Fit()
        f_out.close()


    def Test(self, int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid), help_func)
        g_list.append(self.test_env.graph)
        # self.test_env.stepWithoutReward(0)
        cdef double cost = 0.0
        cdef int i
        sol = []
        while (not self.test_env.isTerminal()):
            # print("Current sol:", sol)
            # print("Num orig nodes:", g_list[0].num_nodes)
            if gid == 0:
                list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list], print_=True)
            else:
                list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            
            new_action = self.argMax(list_pred[0])
            if gid == 0:
                print("list_pred:", list_pred)
                print("new action:", new_action)
            self.test_env.stepWithoutReward(new_action)
            # sol.append(new_action)
        sol = self.test_env.action_list
        if gid == 0:
            print(sol)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        # print("sol:", sol)
        # print("nodes:", nodes)
        # print("Solution:", solution)
        Tourlength = self.utils.getTourLength(g_list[0], solution)
        return Tourlength
    

    def PlayGame(self, int n_traj, double eps):
        print("Playing game!")
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        print("Running simulator...\n")
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        # get intial sample
        for i in range(num_env):
            g_sample = TrainSet.Sample()
            self.env_list[i].s0(g_sample, help_func)
            self.g_list[i] = self.env_list[i].graph
            # self.env_list[i].step(0)
        # num_seq is the number of full sequences starting at an initial state until termination
        while n < num_seq:
            for i in range(num_env):
                # check whether terminal state is reached, if so add the graph to the graph list
                # print("Checking for terminal state")
                if self.env_list[i].isTerminal():
                    n = n + 1
                    # after reaching the terminal state add all nstep transitions to the replay memory 
                    # print("adding new experience to the Replay buffer")
                    self.nStepReplayMem.Add(self.env_list[i], n_step)
                    # print("Sampling new graph..") 
                    g_sample = TrainSet.Sample()
                    self.env_list[i].s0(g_sample, help_func)
                    self.g_list[i] = self.env_list[i].graph
                    # self.env_list[i].step(0)
                    # print("added new sample to the graph list, current length:", len(self.g_list))
            if n >= num_seq:
                break
            Random = False
            if random.uniform(0,1) >= eps:
                # print("Making prediction")
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    # print("Taking random action")
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                # print("Making step")
                self.env_list[i].step(a_t)
            # print("Action lists:", [env.action_list for env in self.env_list])
            # print("covered set:", [env.covered_set for env in self.env_list])
            # print("reward sequence:", [env.reward_seq for env in self.env_list])
            # print("Graph details:", self.g_list[0].num_nodes)


    def SetupTrain(self, idxes, g_list, covered, actions, target):
        # print("Running SetupTrain")
        self.m_y = target
        
        self.inputs['target'] = self.m_y
        # print("Targets:", self.inputs['target'])
        # print("preparing batch graph in SetupTrain...")
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, ignore_covered_edges, include_selected_nodes)
        # print("setting up train in SetupTrain...")
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
        # print("n2nsum_param:", self.inputs['n2nsum_param'])
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        
        self.inputs['node_input'] = prepareBatchGraph.node_feats
        self.inputs['edge_sum'] = prepareBatchGraph.edge_sum
        # print("Edge sum train:", self.inputs['edge_sum'])
        # print("Node input Train:", np.array(self.inputs['node_input']).shape)


    def SetupPredAll(self, idxes, g_list, covered):
        # print("Running SetupPredAll")
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, ignore_covered_edges, include_selected_nodes)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        # print("rep_global:", self.inputs['rep_global'])
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param

        self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
        # print("e2n matrix:", print(type(self.inputs['e2nsum_param'])))
        # print("n2nsum_param:", self.inputs['n2nsum_param'])
        # print("orig weights", g_list[0].edge_weights)
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        # print("subgsum_param:", self.inputs['subgsum_param'])
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        self.inputs['node_input'] = prepareBatchGraph.node_feats
        self.inputs['edge_input'] = prepareBatchGraph.edge_feats
        # print("Node input Pred:", np.array(self.inputs['node_input']), "with shape:", np.array(self.inputs['node_input']).shape)
        self.inputs['edge_sum'] = prepareBatchGraph.edge_sum
        # print("Edge sum pred:", self.inputs['edge_sum'])
        # print("Ran SetupPredAll")
        return prepareBatchGraph.idx_map_list


    def Predict(self, g_list, covered, isSnapShot, print_=False):
        # print("Running Predict")
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        # print("number of graphs for prediction:", n_graphs)
        for i in range(0, n_graphs, BATCH_SIZE):
            # makes sure that th indices start at zero for the first batch and go so on for 
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)
            
            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.e2nsum_param] = self.inputs['e2nsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.node_input] = np.array(self.inputs['node_input'])
            my_dict[self.edge_input] = np.array(self.inputs['edge_input'])
            my_dict[self.edge_sum] = np.array(self.inputs['edge_sum']).reshape((-1, edge_init_dim))
            if print_ == True:
                print("e2nsum_param:", my_dict[self.e2nsum_param])
                print("edge_input", my_dict[self.edge_input])
            if isSnapShot:
                result = self.session.run([self.q_on_allT], feed_dict = my_dict)
                # print("sucessfully ran training session")
            else:
                result = self.session.run([self.q_on_all], feed_dict = my_dict)
                # print("sucessfully ran training session")
            raw_output = result[0]
            # set Q values for all nodes that have already been chosen to negative inf
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))

        return pred

    def PredictWithCurrentQNet(self, g_list, covered, print_=False):
        # print("predicting with current QNet...")
        result = self.Predict(g_list, covered, isSnapShot=False, print_=print_)
        return result

    def PredictWithSnapshot(self, g_list, covered, print_=False):
        # print("predicting with snapshot...")
        result = self.Predict(g_list, covered, isSnapShot=True, print_=print_)
        return result
    #pass
    def TakeSnapShot(self):
        # print("Taking snapshot")
        self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        # Main function for fitting, uses fit() as sub function
        # obtain mini batch sample, can also be bigger since in the end we train only on batches of BATCHSIZE
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        # print("s prime of first sample:", sample.list_s_primes[0])
        # print("state of first sample:", sample.list_st[0])
        # print("mini batch sample obtained")
        ness = False
        cdef int i
        for i in range(BATCH_SIZE):
            # check if minibatch contains non terminal sample
            if (not sample.list_term[i]):
                ness = True
                break
        # print("checked whether mini batch is containing at least one non terminal state sample, it evaluated to:", ness)
        
        ############################## Target Calculation start ##############################
        # only if minbatch contains non terminal sample the Snapshot DQN is used to calculate the target
        if ness:
            if self.IsDoubleDQN:
                # first make prediction with current Q Net for all possible actions and graphs in batch
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                # secondly make prediction with snapshot (older) Q Net for all possible actions and graphs in batch
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                # calculate final prediction for the target term (approximation of expected future return) 
                # by using current net to calculate best action and older net to claculate the approx expected reward
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                # just use older version to 
                
                # print("predicting with snapshot..")
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                # print("sucessfully predicted with snapshot")
                
        list_target = np.zeros([BATCH_SIZE, 1])
        # calculate the target
        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs = GAMMA * list_pred[i]
                else:
                    q_rhs = GAMMA * self.Max(list_pred[i])
            # add the reward to the target
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        ############################## Target Calculation end ##############################
        # print("sucessfully calculated the target for DQN optimization")
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at, list_target)


    def fit(self, g_list, covered, actions, list_target):
        # sub function for fitting the net
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        print("Fitting in total:", n_graphs, "graphs.")
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            # print("Batch indices:", batch_idxes)
            # print("actions:", actions)
            # print("covered:", covered)
            # print("targets:", list_target)
            self.SetupTrain(batch_idxes, g_list, covered, actions, list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.e2nsum_param] = self.inputs['e2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            # my_dict[self.aux_input] = np.array(self.inputs['aux_input'])

            my_dict[self.node_input] = np.array(self.inputs['node_input'])
            my_dict[self.edge_sum] = np.array(self.inputs['edge_sum']).reshape((-1, edge_init_dim))
            my_dict[self.target] = self.inputs['target']
            
            print("running training session...")
            result = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)
            print("sucessfully ran training session")
            loss += result[0]*bsize
        return loss / len(g_list)
    
    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid), help_func)
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Tourlength = self.utils.getTourLength(g_list[0], solution)
        return Tourlength, solution

    def findModel(self, VCFile_path=None):
        if VCFile_path:
            VCFile = VCFile_path
        else:
            VCFile = './models/%s/ModelVC_%d_%d.csv'%(self.g_type, NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            try:
                vc_list.append(float(line))
            except:
                continue
        min_arg_vc = np.argmin(vc_list)
        min_vc = str(np.round(np.min(vc_list), 6))
        best_model_iter = save_interval * min_arg_vc

        best_model = './models/%s/nrange_%d_%d_iter_%d.ckpt' % (self.g_type, NUM_MIN, NUM_MAX, best_model_iter)
        return best_model, VCFile, min_vc
    
    
    def Evaluate(self, g, num_min=NUM_MIN, num_max=NUM_MAX):
        self.InsertGraph(g, is_test=True)
        t1 = time.time()
        len, sol = self.GetSol(0)
        t2 = time.time()
        sol_time = (t2 - t1)
        # reset test set
        self.ClearTestGraphs()
        return len, sol, sol_time


    def LoadbestModel(self, num_min=NUM_MIN, num_max=NUM_MAX):
        base_path = 'best_models/{}/'.format(self.g_type)
        best_tour_length = np.inf
        # file_endings = ['ckpt', 'csv', 'pyx']
        models = os.scandir(base_path)
        for model in models:
            # res = [ele for ele in file_endings if(ele in model)]
            # check whether we have file or folder, continue in case of file
            if model.is_file():
                continue
            new_base_path = base_path + model.name + '/'
            for f in os.listdir(new_base_path):
                nrange_str = 'nrange_{}_{}'.format(num_min, num_max)
                if ('ckpt' not in f) or (nrange_str not in f):
                    continue
                f_len = f.split('_')[-1].split('.')[0]
                tour_length = float(f_len)/(10**(len(f_len)-1))
                if tour_length < best_tour_length:
                    best_model_file = '.'.join(f.split('.')[0:-1])
        self.LoadModel(model_path=new_base_path+best_model_file)
        return best_model_file.split('.')[-2]
    
    
    def gen_graph(self, num_min, num_max):
        """
        Generates new graphs of different g_type--> used for training or testing
        """
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.g_type == 'tsp_2d':
            # slow code, might need optimization
            nodes = np.random.rand(cur_n,2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            feature_dict = {k: {'coord': nodes[k]} for k in range(cur_n)} 
            nx.set_node_attributes(g, feature_dict)
        elif self.g_type == 'tsp':
            # slow code, might need optimization
            nodes = np.random.rand(cur_n, 2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
        return g


    def gen_new_train_graphs(self, num_min, num_max, num_graphs=n_generator):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        cdef int i
        for i in tqdm(range(num_graphs)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)
    
    def PrepareValidData(self):
        cdef int counter = 0
        if self.valid_path:
            try:
                fnames = os.listdir(self.valid_path)
                print('\nLoading validation graphs...')
            except:
                print('\nBad validation directory!')
            
            for fname in fnames:
                try:
                    problem = tsplib95.load(self.valid_path + fname)
                    g = problem.get_graph()
                    # remove edges from nodes to itself
                    ebunch=[(k,k) for k in range(len(g.nodes))]
                    g.remove_edges_from(ebunch)
                    for node in range(len(g.nodes)):
                        g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * valid_scale_fac
                    for edge in g.edges:
                        g.edges[edge]['weight'] = g.edges[edge]['weight'] * valid_scale_fac
                    self.InsertGraph(g, is_test=True)
                    counter += 1
                except:
                    continue
            print("\nSucessfully loaded {} validation graphs!".format(counter))
        else:
            print('\nGenerating validation graphs...')
            sys.stdout.flush()
            for i in tqdm(range(n_valid)):
                g = self.gen_graph(NUM_MIN, NUM_MAX)
                self.InsertGraph(g, is_test=True)

    def GenNetwork(self, g, scale_factor=1):    #networkx2four
        # transforms the networkx graph object into C graph object using external pyx module
        nodes = g.nodes()
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges) 
            A = np.array(a)
            B = np.array(b)
            # edge weights
            W = np.array([g[n][m]['weight'] for n, m in zip(a, b)])
            W = W * scale_factor
            # node features (node position)
            try:
                F = np.array([g.nodes[k]['coord'] for k in range(len(nodes))])
                F = F * scale_factor
            except:
                F = np.ones((len(nodes), 2))
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
            F = np.array([0])
        return graph.py_Graph(len(nodes), len(edges), A, B, W, F)
             

    def InsertGraph(self, g, is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))


    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()


    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()


    def GetSolution(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid), help_func)
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        sum_sort_time = 0
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            start_time = time.time()
            batchSol = np.argsort(-list_pred[0])[:step]
            end_time = time.time()
            sum_sort_time += (end_time-start_time)
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        return sol
    
    def SaveModel(self,model_path):
        # saves the model based on tf saver
        self.saver.save(self.session, model_path)
        print('model sucessfully saved!')

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('model sucessfully restored from file')


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best
     
    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.e2nsum_param] = self.inputs['e2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            # my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.node_input] = np.array(self.inputs['node_input'])
            my_dict[self.edge_sum] = np.array(self.inputs['edge_sum']).reshape((-1, edge_init_dim))
            
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)
    
    def EvaluateSol(self, data_test, sol_file, strategyID, reInsertStep):
        sys.stdout.flush()
        g = nx.read_edgelist(data_test)
        g_inner = self.GenNetwork(g)
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            if reInsertStep > 0 and reInsertStep < 1:
                step = np.max([int(reInsertStep*nx.number_of_nodes(g)),1]) #step size
            else:
                step = reInsertStep
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, step)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution))
        Tourlength = self.utils.getTourLength(g_inner, solution)
        return Tourlength