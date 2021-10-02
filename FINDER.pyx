#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import tensorflow as tf
tf.compat.v1.set_random_seed(73)
import numpy as np
np.random.seed(42)
import random
random.seed(7)

import scipy.linalg as linalg
from scipy.sparse import csr_matrix
from itertools import combinations
import tsplib95

from distutils.util import strtobool
from tqdm import tqdm
import time
import pickle as cp
import sys
import os
import re
import gc


import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import pyx_utils
from py_utils.TSP_loader import TSP_loader
from dqn.FINDER_decoder import MLPdecoder, AttentionDecoder
from dqn.FINDER_state_encoder import MHAStateEncoder, BasicStateEncoder


np.set_printoptions(threshold=sys.maxsize)
# fix seeds for graph generation and weight init
# tf.set_random_seed(73)
# random.seed(7)
# np.random.seed(42)

cdef double inf = 1073741823.5

class FINDER:
    
    def __init__(self, config):
        print("Gpu available:", tf.test.is_gpu_available())
        print("Built with cuda:", tf.test.is_built_with_cuda())
        self.cfg = config
        
        self.print_params = True
        self.print_test_results = True
        
        # explicitly define some variables for speed up
        cdef int MEMORY_SIZE = self.cfg['MEMORY_SIZE']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        cdef int NUM_MIN = self.cfg['NUM_MIN']
        cdef int num_env = self.cfg['num_env']
        cdef int help_func = self.cfg['help_func']
        cdef int reward_sign = self.cfg['reward_sign']
        cdef int fix_start_node = self.cfg['fix_start_node']

        #Simulator
        self.TrainSet = graph.py_GSet() # initializes the training and test set object
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.utils = pyx_utils.pyx_Utils()
        self.tsp_loader = TSP_loader()

        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.pred=[]

        if self.cfg['IsPrioritizedSampling']:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(self.cfg['epsilon'], self.cfg['alpha'], self.cfg['beta'], self.cfg['beta_increment_per_sampling'], 
                                                                         self.cfg['TD_err_upper'], MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        cdef int norm
        if self.cfg['reward_normalization'] == 'max':
            norm = NUM_MAX
        elif self.cfg['reward_normalization'] == 'min':
            norm = NUM_MIN
        else:
            norm = -1
        
        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(norm, help_func, reward_sign, fix_start_node))
            self.g_list.append(graph.py_Graph())
        
        # stop tf from displaying deprecation warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        self.setup_placeholder_dict()

        if self.cfg['IsPrioritizedSampling']:
            self.ISWeights = tf.placeholder(tf.float32, [self.cfg['BATCH_SIZE'], 1], name='IS_weights')

        if self.cfg['net_type'] == 'AGNN':
            assert(self.cfg['REG_HIDDEN'] > 0)
            self.cfg['embeddingMethod'] = 3 # make sure all relevant aggregation matrices are calculated
            # set the state encoder
            if self.cfg['state_representation'] == 0:
                self.state_encoder = self.Build_BasicStateEncoder
            elif self.cfg['state_representation'] == 1:
                self.state_encoder = self.Build_MHAStateEncoder
            # set the decoder
            if self.cfg['decoder_type'] == 0:
                self.decoder = self.Build_MLPdecoder
            elif self.cfg['decoder_type'] == 1:
                self.decoder = self.Build_AttentionDecoder

            self.placeholder_dict['node_embed_input'] = tf.placeholder(tf.float32, name='node_embed_input')
            self.placeholder_dict['state_embed_input'] = tf.placeholder(tf.float32, name='state_embed_input')
            # init Q network
            with tf.compat.v1.variable_scope('train_DQN'):
                self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildAGNN() #[loss,trainStep,q_pred, q_on_all, ...]
            self.Q_param_list = tf.compat.v1.trainable_variables(scope='train_DQN')
            print("DQN params", [tensor.name for tensor in self.Q_param_list])
            # print(len(self.Q_param_list))
            # print("Train Decoder params",[tensor.name for tensor in tf.compat.v1.trainable_variables(scope='train_DQN/decoder')])
            #init Target Q Network
            with tf.compat.v1.variable_scope('target_DQN'):
                self.node_embed_target, self.edge_embed_target = self.BuildAGNN_encoder()

                # self.state_embed_target = self.Build_StateEncoder(self.node_embed_target)
                self.state_embed_target = self.state_encoder(self.node_embed_target, training=False)

                self.q_on_allT = self.decoder(self.state_embed_target, self.node_embed_target)

            self.init_node_embed = None
            
            with tf.compat.v1.variable_scope('test_DQN'):
                self.node_embed_test, self.edge_embed_test = self.BuildAGNN_encoder()

                # self.state_embed_test = self.Build_StateEncoder(self.placeholder_dict['node_embed_input'])
                self.state_embed_test = self.state_encoder(self.placeholder_dict['node_embed_input'], training=False)

                self.q_on_all_test = self.decoder(self.state_embed_test, self.placeholder_dict['node_embed_input'])
        
            self.test_DQN_params = tf.compat.v1.trainable_variables(scope='test_DQN')
            # print("Test DQN params", [tensor.name for tensor in self.test_DQN_params])
            # print(len(self.test_DQN_params))
            self.updateTestDQN = tf.group(*[a.assign(b) for a,b in zip(self.test_DQN_params, self.Q_param_list)])

            self.target_DQN_params = tf.compat.v1.trainable_variables(scope='target_DQN')
            print("Target DQN params", [tensor.name for tensor in self.target_DQN_params])
            self.UpdateTargetDQN = tf.group(*[a.assign(b) for a,b in zip(self.target_DQN_params, self.Q_param_list)])
        else:
           # overwrite config
            self.cfg['selected_nodes_inclusion'] = 2
            self.cfg['ignore_covered_edges'] = 1
            self.cfg['embeddingMethod'] = 3 # makes sure all relevant aggregation matrices are calculated
            # init Q network
            self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.Build_S2VDQN() #[loss,trainStep,q_pred, q_on_all, ...]
            #init Target Q Network
            self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.Build_S2VDQN()
            #takesnapsnot
            self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT, self.Q_param_list)]
            self.UpdateTargetDQN = tf.group(*self.copyTargetQNetworkOperation)

        # saving and loading networks
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.global_variables_initializer())

        self.writer = tf.compat.v1.summary.FileWriter('./graphs', graph=self.session.graph)

################################################# New code for FINDER #################################################
###################################################### BuildNet start ######################################################    
    
    def BuildAGNN_encoder(self):
        # some definitions for convenience
        cdef int node_init_dim = self.cfg['node_init_dim']
        cdef int edge_init_dim = self.cfg['edge_init_dim']
        cdef int state_init_dim = self.cfg['state_init_dim']
        
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef int edge_embed_dim = self.cfg['edge_embed_dim']
        cdef double initialization_stddev = self.cfg['initialization_stddev']

        cdef int max_bp_iter = self.cfg['max_bp_iter']
        assert(node_embed_dim == edge_embed_dim)
        # [node_init_dim, node_embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([node_init_dim, node_embed_dim], stddev=initialization_stddev), tf.float32)

        # [edge_init_dim, edge_embed_dim]
        w_e2l = tf.Variable(tf.truncated_normal([edge_init_dim, edge_embed_dim], stddev=initialization_stddev), tf.float32)

        U = [tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32) for i in range(max_bp_iter)]
        V = [tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32) for i in range(max_bp_iter)]
        A = [tf.Variable(tf.truncated_normal([node_embed_dim, edge_embed_dim], stddev=initialization_stddev), tf.float32) for i in range(max_bp_iter)]
        B = [tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32) for i in range(max_bp_iter)]
        C = [tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32) for i in range(max_bp_iter)]
        
        # [node_cnt, node_init_dim] * [node_init_dim, node_embed_dim] = [node_cnt, node_embed_dim], not sparse
        node_init = tf.matmul(tf.cast(self.placeholder_dict['node_input'], tf.float32), w_n2l)
        cur_node_embed = node_init
        
        # [edge_cnt, edge_dim] * [edge_dim, embed_dim] = [edge_cnt, embed_dim]
        edge_init = tf.matmul(tf.cast(self.placeholder_dict['edge_input'], tf.float32), w_e2l)
        cur_edge_embed = edge_init

        ################### GNN start ###################
        cdef int lv = 0
        while lv < max_bp_iter:
            cur_node_embed_prev = tf.identity(cur_node_embed)
            # node embed update
            node_linear_0 = tf.matmul(cur_node_embed, V[lv])
            tmp_n2e = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['n2esum_param_0'], tf.float32), node_linear_0)
            
            tmp_edge_embed = tf.math.sigmoid(cur_edge_embed)
            mult_edge_embed = tf.math.multiply(tmp_edge_embed, tmp_n2e)
            
            new_node_embed = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['e2nsum_param'], tf.float32), mult_edge_embed)
            node_linear_1 = tf.matmul(cur_node_embed, U[lv])
            new_node_embed = tf.math.add(node_linear_1, new_node_embed)
            
            norm_layer_0 = tf.keras.layers.LayerNormalization(axis=1)
            new_node_embed = tf.reshape(new_node_embed, [-1, node_embed_dim])
            new_node_embed = norm_layer_0(new_node_embed)
            new_node_embed = tf.nn.relu(new_node_embed)
            cur_node_embed = tf.math.add(cur_node_embed, new_node_embed)
            
            cur_node_embed = tf.cond(self.placeholder_dict['is_training'], lambda: tf.nn.dropout(cur_node_embed, rate=self.cfg['dropout_rate']), lambda: cur_node_embed)

            # edge embed update
            tmp_node_embed_0 = tf.matmul(cur_node_embed_prev, B[lv])
            tmp_node_embed_1 = tf.matmul(cur_node_embed_prev, C[lv])
            
            tmp_n2e_0 = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['n2esum_param_0'], tf.float32), tmp_node_embed_0)
            tmp_n2e_1 = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['n2esum_param_1'], tf.float32), tmp_node_embed_1)
            
            tmp_edge_embed = tf.matmul(cur_edge_embed, A[lv])
            
            sum_edge_embed = tf.math.add(tmp_n2e_0, tmp_n2e_1)
            sum_edge_embed = tf.math.add(sum_edge_embed, tmp_edge_embed)
            
            norm_layer_1 = tf.keras.layers.LayerNormalization(axis=1)
            new_edge_embed = tf.reshape(sum_edge_embed, [-1, edge_embed_dim])
            new_edge_embed = norm_layer_1(new_edge_embed)
            new_edge_embed = tf.nn.relu(new_edge_embed)
            cur_edge_embed = tf.math.add(cur_edge_embed, new_edge_embed)
            
            cur_edge_embed = tf.cond(self.placeholder_dict['is_training'], lambda: tf.nn.dropout(cur_edge_embed, rate=self.cfg['dropout_rate']), lambda: cur_edge_embed)

            lv = lv + 1
        return cur_node_embed, cur_edge_embed
    
    def Build_BasicStateEncoder(self, node_embed, training=False):
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef int state_embed_dim = self.cfg['state_embed_dim']
        self.cfg['state_embed_dim'] = 3 * self.cfg['node_embed_dim']
        state_encoder = BasicStateEncoder(placeholder_dict=self.placeholder_dict, state_embed_dim=state_embed_dim, 
                                          node_embed_dim=node_embed_dim)
        state_embed = state_encoder(node_embed)
        return state_embed

    def Build_MHAStateEncoder(self, node_embed, training=False):
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef int state_embed_dim = self.cfg['state_embed_dim']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        cdef double dropout_rate = self.cfg['dropout_rate']
        state_encoder = MHAStateEncoder(placeholder_dict=self.placeholder_dict, state_embed_dim=state_embed_dim, 
                                        node_embed_dim=node_embed_dim, num_heads=8, max_nodes=NUM_MAX, d_model=128,
                                        rate=dropout_rate)
        state_embed = state_encoder(node_embed, training=training)
        return state_embed
    
    def Build_MLPdecoder(self, state_embed, node_embed, q_on_all=True, training=False, return_decoder=False):
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef int state_embed_dim = self.cfg['state_embed_dim']
        cdef double dropout_rate = self.cfg['dropout_rate']
        decoder = MLPdecoder(placeholder_dict=self.placeholder_dict, state_embed_dim=state_embed_dim, 
                             node_embed_dim=node_embed_dim, weight_stddev=self.cfg['initialization_stddev'], 
                             hidden_dim=self.cfg['REG_HIDDEN'], rate=dropout_rate)
        q_pred = decoder(state_embed, node_embed, q_on_all=q_on_all, training=training) # Q(s,a)
        if return_decoder:
            return q_pred, decoder
        else:
            return q_pred

    def Build_AttentionDecoder(self, state_embed, node_embed, q_on_all=True, training=False, return_decoder=False):
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef int state_embed_dim = self.cfg['state_embed_dim']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        cdef double dropout_rate = self.cfg['dropout_rate']
        decoder = AttentionDecoder(placeholder_dict=self.placeholder_dict, state_embed_dim=state_embed_dim, 
                                   node_embed_dim=node_embed_dim, max_nodes=NUM_MAX, d_model=128, 
                                   use_biases=False, rate=dropout_rate)
        q_pred = decoder(state_embed, node_embed, q_on_all=q_on_all, training=training) # Q(s,a)
        if return_decoder:
            return q_pred, decoder
        else:
            return q_pred

    def BuildAGNN(self):
        with tf.compat.v1.variable_scope('encoder'):
            node_embed, cur_edge_embed = self.BuildAGNN_encoder()
        
        with tf.compat.v1.variable_scope('state_encoder'):
            # state_embed = self.Build_StateEncoder(node_embed)
            state_embed = self.state_encoder(node_embed, training=True)
        
        with tf.compat.v1.variable_scope('action_encoder'):
            # potential action encoding
            action_embed = node_embed
        
        with tf.compat.v1.variable_scope('decoder'):
            reduced_action_embed = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['action_select'], tf.float32), action_embed)
            q_pred, decoder = self.decoder(state_embed, reduced_action_embed, q_on_all=False, training=True, return_decoder=True)

        if self.cfg['IsPrioritizedSampling']:
            self.TD_errors = tf.reduce_sum(tf.abs(self.placeholder_dict['target'] - q_pred), axis=1)    # for updating Sumtree
            if self.cfg['IsHuberloss']:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.placeholder_dict['target'], self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.placeholder_dict['target'], q_pred))
        else:
            if self.cfg['IsHuberloss']:
                loss_rl = tf.losses.huber_loss(self.placeholder_dict['target'], q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.placeholder_dict['target'], q_pred)
        # calculate full loss
        loss = loss_rl

        trainStep = tf.compat.v1.train.AdamOptimizer(self.cfg['LEARNING_RATE']).minimize(loss)
        # repeat states to calc q_values for all actions
        q_on_all = decoder(state_embed, action_embed, q_on_all=True, training=False)

        return loss, trainStep, q_pred, q_on_all, tf.compat.v1.trainable_variables()

    
###################################################### BuildNet end ######################################################
    
    def Train(self):
        cdef int NUM_MIN = self.cfg['NUM_MIN']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        cdef int n_generator = self.cfg['n_generator']
        cdef double eps_start = self.cfg['eps_start'] 
        cdef double eps_end = self.cfg['eps_end']
        cdef double eps_step =  self.cfg['eps_step']
        cdef int MAX_ITERATION = self.cfg['MAX_ITERATION']
        cdef int n_valid = self.cfg['n_valid']
        cdef double loss = 0.0
        cdef double frac, start, end
        cdef int i, iter, idx

        self.PrepareValidData()
        try:
            valid_lengths = None
            valid_path = self.cfg['valid_path']
            with open(f'{valid_path}/lengths.txt', 'r') as f:
                lines = f.readlines()
                lines = [float(line.split(':')[-1].strip()) for line in lines]
            valid_lengths = lines
        except:
            print("Could not load validation lengths!")
        self.num_train_prep_cycles = 0
        self.prepare_new_training_graphs()

        
        for i in range(10):
            self.PlayGame(100, 1)
        
        self.TakeSnapShot()
        self.UpdateTestDQN()
        
        #save_dir = './models/%s'%self.cfg['g_type']
        save_dir = './models/{}/nrange_{}_{}'.format(self.cfg['g_type'], NUM_MIN, NUM_MAX)
        ckpt_save_dir = f'{save_dir}/checkpoints'
        architecture_save_dir = f'{save_dir}/architecture'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        valid_file = f'{save_dir}/Validation_{NUM_MIN}_{NUM_MAX}.csv'
        LossFile = f'{save_dir}/Loss_{NUM_MIN}_{NUM_MAX}.csv'
        valid_approx_out = open(valid_file, 'w')
        loss_out = open(LossFile, 'w')
        for iter in tqdm(range(MAX_ITERATION)):
            start = time.clock()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.prepare_new_training_graphs()
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % self.cfg['UPDATE_TIME'] == 0:
                self.TakeSnapShot()
            # print("Fitting in 5 seconds...")
            # time.sleep(5)
            loss = self.Fit()
            if iter % 10 == 0:
                loss_out.write(f'{iter} {loss}\n')
                loss_out.flush()
            # testing
            if iter % self.cfg['save_interval'] == 0:
                os.system('clear')
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                self.UpdateTestDQN()
                tqdm.write("Running test...")
                test_start = time.time()
                for idx in tqdm(range(n_valid)):
                    if valid_lengths:
                        frac += self.Test(idx)[0]/valid_lengths[idx]
                    else:
                        frac += self.Test(idx)[0]
                test_end = time.time()
                # print("Test finished, sleeping for 5 seconds...")
                # time.sleep(5)
                if self.cfg['valid_path']:
                    valid_approx = frac/len(valid_lengths)
                    valid_approx_out.write(f'{iter} {valid_approx}\n')
                else:
                    valid_approx_out.write('%.16f\n'%(frac/n_valid))
                valid_approx_out.flush()
                print('iter %d, eps %.4f, average tour length: %.6f'%(iter, eps, frac/n_valid))
                print ('testing %d graphs time: %.2fs'%(self.cfg['n_valid'], test_end-test_start))
                N_end = time.clock()
                print ('%d iterations total time: %.2fs\n'%(self.cfg['save_interval'], N_end-N_start))
                print(f"Loss: {loss}")
                sys.stdout.flush()
                model_path = f'{ckpt_save_dir}/nrange_{NUM_MIN}_{NUM_MAX}_iter_{iter}.ckpt'
                if not os.path.exists(ckpt_save_dir):
                    os.makedirs(ckpt_save_dir)
                self.SaveModel(model_path)
            self.writer.flush()
        valid_approx_out.close()
        loss_out.close()
        self.writer.close()


    def Test(self, int gid):
        graph = self.TestSet.Get(gid)

        if self.cfg['search_strategy'] == 'beam_search+':
            sol= self.solve_with_beam_search(graph=graph, select_true_best=True)
        elif self.cfg['search_strategy'] == 'beam_search':
            sol = self.solve_with_beam_search(graph=graph, select_true_best=False)
        else:
            # select greedy
            if self.print_test_results and gid == 0:
                verbose = True
            else:
                verbose = False
            sol = self.solve_greedy(graph=graph, verbose=verbose)
        
        if self.print_test_results and gid == 0:
            print(sol)
        nodes = list(range(graph.num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        # print("sol:", sol)
        # print("nodes:", nodes)
        # print("Solution:", solution)
        tour_length = self.utils.getTourLength(graph, solution)
        return tour_length, solution

    def solve_greedy(self, graph, verbose=False):
        # for the test env the norm is not used since no reward is calculated
        cdef int help_func = self.cfg['help_func']
        cdef int fix_start_node = self.cfg['fix_start_node']
        self.test_env = mvc_env.py_MvcEnv(1, help_func, 1, fix_start_node)
        self.test_env.s0(graph)
        num_nodes = graph.num_nodes
        cdef int step = 0
        while not self.test_env.isTerminal():
            possible_actions = [n for n in range(0,num_nodes) if n not in self.test_env.state]
            # skip prediction if only one node is left for selection
            if len(possible_actions) == 1:
                action = possible_actions[0]
                self.test_env.stepWithoutReward(action)
                continue
            if self.cfg['one_step_encoding']:
                # only encode in the first step
                if step == 0:
                    q_values = self.Predict([graph], [self.test_env.state], isSnapShot=False, initPred=True, test=True)     
                else:
                    q_values = self.Predict([graph], [self.test_env.state], isSnapShot=False, initPred=False, test=True)
            else:
                q_values = self.Predict([graph], [self.test_env.state], isSnapShot=False, initPred=False, test=False)
            if verbose:
                print(q_values)
            action = self.argMax(q_values[0])
            self.test_env.stepWithoutReward(action)
            step += 1
        return self.test_env.state

    def solve_with_beam_search(self, graph, select_true_best=False, only_use_cur_qvalue=False):
        cdef int help_func = self.cfg['help_func']
        cdef int fix_start_node = self.cfg['fix_start_node']
        self.test_env_list = [mvc_env.py_MvcEnv(1, help_func, 1, fix_start_node) for i in range(self.cfg['beam_width'])]
        # can be easily batched
        for test_env in self.test_env_list:
            test_env.s0(graph)
        beam_width = self.cfg['beam_width']
        num_nodes = graph.num_nodes
        sequences = [[self.test_env_list[0], [], 1]]
        # walk over each step in sequence
        cdef int step = 0
        while not self.test_env_list[0].isTerminal():
            all_candidates = []
            # expand each current candidate 
            env_states = [sequence[0].state for sequence in sequences]
            g_list = [graph for sequence in sequences]
            if self.cfg['one_step_encoding']:
                # only encode in the first step
                if step == 0:
                    list_pred = self.Predict(g_list, env_states, isSnapShot=False, initPred=True, test=True)     
                else:
                    list_pred = self.Predict(g_list, env_states, isSnapShot=False, initPred=False, test=True)
            else:
                list_pred = self.Predict(g_list, env_states, isSnapShot=False, initPred=False, test=False)
                
            
            probabilities_list = [self.softmax(q_values) for q_values in list_pred]
            # print("Number sequences:", len(sequences))
            # print("Number probability outputs", len(probabilities_list))
            for i in range(len(sequences)):    
                cur_env, cur_act, cur_score = sequences[i]
                possible_actions = [n for n in range(0,num_nodes) if n not in env_states[i]]
                for action in possible_actions:
                    if only_use_cur_qvalue:
                        candidate = [cur_env, cur_act + [action], probabilities_list[i][action]]
                    else:
                        candidate = [cur_env, cur_act + [action], cur_score * probabilities_list[i][action]]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[2], reverse=True)
            # select k best
            sequences = ordered[:beam_width]
            # take actions in test environments, initialize history 
            for i, sequence in enumerate(sequences):
                self.test_env_list[i] = mvc_env.copy_test_environment(sequence[0])
                self.test_env_list[i].stepWithoutReward(sequence[1][0])
                sequences[i] = [self.test_env_list[i], [], sequence[2]]
            step += 1
        # last list of sequences is ordered by probability
        if select_true_best:
            # select the shortest tour of the final candidate tours
            best_tour = sequences[0][0].state
            best_tour_length = self.utils.getTourLength(graph, best_tour)
            for env, sequence, score in sequences[1::]:
                tour = env.state
                tour_length = self.utils.getTourLength(graph, tour)
                if tour_length < best_tour_length:
                    best_tour = tour
                    best_tour_length = tour_length
        else:
            best_tour = sequences[0][0].state
        return best_tour
    
    
    def PredictWithCurrentQNet(self, g_list, covered):
        # print("predicting with current QNet...")
        result = self.Predict(g_list, covered, isSnapShot=False)
        return result

    def PredictWithSnapshot(self, g_list, covered):
        # print("predicting with snapshot...")
        result = self.Predict(g_list, covered, isSnapShot=True)
        return result
    
    def Predict(self, g_list, covered, isSnapShot, initPred=True, test=False):
        # for i in range(0, beam_width, bbsize):
        #     if i + bbsize <= beam_width:
        #         end_idx = i + bbsize
        #     else:
        #         end_idx = bbsize
        #     batch_g_list = g_list[i:end_idx]
        #     batch_env_states = env_states[i]
        
        # print("Running Predict")
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        full_pred = []
        # print("number of graphs for prediction:", n_graphs)
        for i in range(0, n_graphs, self.cfg['BATCH_SIZE']):
            # makes sure that th indices start at zero for the first batch and go so on for 
            bsize = self.cfg['BATCH_SIZE']
            if (i + self.cfg['BATCH_SIZE']) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)
            
            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            
            my_dict = {}
            for key in self.inputs:
                my_dict[self.placeholder_dict[key]] = self.inputs[key]
            
            if self.print_params:
                # print("start_param:", my_dict[self.placeholder_dict['start_param']])
                # print("end_param:", my_dict[self.placeholder_dict['end_param']])
                # if self.cfg['embeddingMethod'] in [2, 3]:
                #     print("e2nsum_param:", my_dict[self.placeholder_dict['e2nsum_param']])
                # print("edge_input", my_dict[self.placeholder_dict['edge_input']])
                # print("state_param:", my_dict[self.placeholder_dict['state_param']])
                pass
            if test:
                if initPred:
                    self.init_node_embed = self.session.run(self.node_embed_test, feed_dict=my_dict)
                    my_dict[self.placeholder_dict['node_embed_input']] = np.array(self.init_node_embed)
                    result = self.session.run([self.q_on_all_test], feed_dict=my_dict)
                    # print(self.init_node_embed)
                else:
                    my_dict[self.placeholder_dict['node_embed_input']] = np.array(self.init_node_embed)
                    result = self.session.run([self.q_on_all_test], feed_dict=my_dict)
                    # do something based on self.init_node_embed
            else:
                if isSnapShot:
                    result = self.session.run([self.q_on_allT], feed_dict=my_dict)
                    # result = self.session.run([self.q_on_allT, self.tensor_list], feed_dict = my_dict)
                    # print("sucessfully ran training session")
                else:
                    result = self.session.run([self.q_on_all], feed_dict=my_dict)
                    # result = self.session.run([self.q_on_all, self.tensor_list], feed_dict = my_dict)
                    # print("sucessfully ran training session")
            
            raw_output = result[0]
            # set Q values for all nodes that have already been chosen to negative inf
            if self.print_params:
                # print("covered embed padded:", np.array(result[1][0]))
                # print("masked_embed:", np.array(result[1][1]))
                # print("state_embed:", np.array(result[1][2]))
                # print("num_samples:", np.array(result[1][3]))
                # print("shape covered embed:", np.array(result[1][4]))
                # print("shape covered reshaped embed:", np.array(result[1][5]))
                # print("num nodes", [g.num_nodes for g in g_list])
                # print("num covered nodes:", [len(cover) for cover in covered])
                pass
            # print(np.array(tensor_list[0]))
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
            full_pred.extend(pred)
        return full_pred

    def prepare_inputs(self, prepareBatchGraph):

        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        
        if self.cfg['embeddingMethod'] == 2:
            self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
        elif self.cfg['embeddingMethod'] == 3:
            self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
            self.inputs['n2esum_param_0'] = prepareBatchGraph.n2esum_param_0
            self.inputs['n2esum_param_1'] = prepareBatchGraph.n2esum_param_1

        self.inputs['start_param'] = prepareBatchGraph.start_param
        self.inputs['end_param'] = prepareBatchGraph.end_param
        self.inputs['agg_state_param'] = prepareBatchGraph.agg_state_param
        self.inputs['state_sum_param'] = prepareBatchGraph.state_sum_param
        self.inputs['state_param'] = prepareBatchGraph.state_param
        self.inputs['mask_param'] = prepareBatchGraph.mask_param
        self.inputs['pad_node_param'] = prepareBatchGraph.pad_node_param
        self.inputs['pad_reverse_param'] = prepareBatchGraph.pad_reverse_param
        
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        # self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        self.inputs['edge_input'] = np.array(prepareBatchGraph.edge_feats)
        self.inputs['node_input'] = np.array(prepareBatchGraph.node_feats)
        self.inputs['edge_sum'] = np.array(prepareBatchGraph.edge_sum).reshape((-1, self.cfg['edge_init_dim']))

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        cdef int aggregatorID = self.cfg['aggregatorID']
        cdef int node_init_dim = self.cfg['node_init_dim']
        cdef int edge_init_dim = self.cfg['edge_init_dim']
        cdef int ignore_covered_edges = self.cfg['ignore_covered_edges']
        cdef int selected_nodes_inclusion = self.cfg['selected_nodes_inclusion']
        cdef int embeddingMethod = self.cfg['embeddingMethod']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        # clear inputs
        self.inputs = dict()
        # print("Running SetupTrain")
        self.inputs['target'] = target
        self.inputs['is_training'] = True
        # print("Targets:", self.inputs['target'])
        # print("preparing batch graph in SetupTrain...")
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, 
                                                                   ignore_covered_edges, selected_nodes_inclusion, 
                                                                   embeddingMethod, NUM_MAX)
        # print("setting up train in SetupTrain...")
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.prepare_inputs(prepareBatchGraph)
        # print("Edge sum train:", self.inputs['edge_sum'])
        # print("Node input Train:", np.array(self.inputs['node_input']).shape)


    def SetupPredAll(self, idxes, g_list, covered):
        cdef int aggregatorID = self.cfg['aggregatorID']
        cdef int node_init_dim = self.cfg['node_init_dim']
        cdef int edge_init_dim = self.cfg['edge_init_dim']
        cdef int ignore_covered_edges = self.cfg['ignore_covered_edges']
        cdef int selected_nodes_inclusion = self.cfg['selected_nodes_inclusion']
        cdef int embeddingMethod = self.cfg['embeddingMethod']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        # clear inputs
        # print("Running SetupPredAll")
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, 
                                                                   ignore_covered_edges, selected_nodes_inclusion, 
                                                                   embeddingMethod, NUM_MAX)
        # print("Initialized PrepareBatchGraph")
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        # print("Ran SetupPredAll")
        
        self.inputs = dict()
        self.inputs['is_training'] = False
        self.prepare_inputs(prepareBatchGraph)
        # print("Edge sum pred:", self.inputs['edge_sum'])
        # print("Ran SetupPredAll")
        return prepareBatchGraph.idx_map_list

    def Setup_NodeLevelInputs(self, idxes, g_list, covered):
        cdef int aggregatorID = self.cfg['aggregatorID']
        cdef int node_init_dim = self.cfg['node_init_dim']
        cdef int edge_init_dim = self.cfg['edge_init_dim']
        cdef int ignore_covered_edges = self.cfg['ignore_covered_edges']
        cdef int selected_nodes_inclusion = self.cfg['selected_nodes_inclusion']
        cdef int embeddingMethod = self.cfg['embeddingMethod']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        # clear inputs
        # print("Running SetupPredAll")
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, 
                                                                   ignore_covered_edges, selected_nodes_inclusion, 
                                                                   embeddingMethod, NUM_MAX)
                                        
        self.inputs = dict()
        self.inputs['is_training'] = False
        self.prepare_inputs(prepareBatchGraph)
        return prepareBatchGraph.idx_map_list

    def setup_placeholder_dict(self):
        self.placeholder_dict = dict()
        # encoder inputs
        # [node_cnt, node_init_dim]
        self.placeholder_dict['node_input'] = tf.placeholder(tf.float32, name="node_input")
        # [edge_cnt, edge_init_dim]
        self.placeholder_dict['edge_sum'] = tf.placeholder(tf.float32, name="edge_sum")
        # [edge_cnt, edge_init_dim]
        self.placeholder_dict['edge_input'] = tf.placeholder(tf.float32, name="edge_input")
        # [node_cnt, node_cnt]
        self.placeholder_dict['n2nsum_param'] = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, edge_cnt]
        self.placeholder_dict['e2nsum_param'] = tf.sparse_placeholder(tf.float32, name="e2nsum_param")
        # [edge_cnt, node_cnt]
        self.placeholder_dict['n2esum_param_0'] = tf.sparse_placeholder(tf.float32, name="n2esum_param_0")
        # [edge_cnt, node_cnt]
        self.placeholder_dict['n2esum_param_1'] = tf.sparse_placeholder(tf.float32, name="n2esum_param_1")

        # [batch_size, node_cnt]
        self.placeholder_dict['action_select'] = tf.sparse_placeholder(tf.float32, name="action_select")
        # [batch_size, node_cnt]
        self.placeholder_dict['start_param'] = tf.sparse_placeholder(tf.float32, name="start_param")
        # [batch_size, node_cnt]
        self.placeholder_dict['end_param'] = tf.sparse_placeholder(tf.float32, name="end_param")
        # [batch_size, node_cnt]
        self.placeholder_dict['agg_state_param'] = tf.sparse_placeholder(tf.float32, name="agg_state_param")
        # [batch_size, node_cnt]
        self.placeholder_dict['state_sum_param'] = tf.sparse_placeholder(tf.float32, name="state_sum_param")
        # [max_nodes*batch_size, node_cnt]
        self.placeholder_dict['state_param'] = tf.sparse_placeholder(tf.float32, name="state_param")
        # [node_cnt, node_cnt]
        self.placeholder_dict['mask_param'] = tf.sparse_placeholder(tf.float32, name="mask_param")
        # [num_nodes*batch_size, node_cnt]
        self.placeholder_dict['pad_node_param'] = tf.sparse_placeholder(tf.float32, name="pad_node_param")
        # [node_cnt, num_nodes*batch_size]
        self.placeholder_dict['pad_reverse_param'] = tf.sparse_placeholder(tf.float32, name="pad_reverse_param")
        # [node_cnt, node_cnt]
        self.placeholder_dict['laplacian_param'] = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt], sum over all noce embeddings for virtual node state representation
        self.placeholder_dict['subgsum_param'] = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [node_cnt, batch_size]
        self.placeholder_dict['rep_global'] = tf.sparse_placeholder(tf.float32, name="rep_global")
        # []
        self.placeholder_dict['is_training'] = tf.placeholder(tf.bool, name="is_training")
        # [batch_size, 1]
        self.placeholder_dict['target'] = tf.placeholder(tf.float32, [self.cfg['BATCH_SIZE'], 1], name="target")

    def Fit(self):
        # Main function for fitting, uses fit() as sub function
        # obtain mini batch sample, can also be bigger since in the end we train only on batches of BATCHSIZE
        cdef int BATCH_SIZE = self.cfg['BATCH_SIZE']
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
            if self.cfg['IsDoubleDQN']:
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
                if self.cfg['IsDoubleDQN']:
                    q_rhs = self.cfg['GAMMA'] * list_pred[i]
                else:
                    q_rhs = self.cfg['GAMMA'] * self.Max(list_pred[i])
            # add the reward to the target
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        ############################## Target Calculation end ##############################
        # print("sucessfully calculated the target for DQN optimization")
        if self.cfg['IsPrioritizedSampling']:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at, list_target)

    def fit(self, g_list, covered, actions, list_target):
        # sub function for fitting the net
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j
        cdef int bsize = self.cfg['BATCH_SIZE']
        # print("Fitting in total:", n_graphs, "graphs.")
        for i in range(0, n_graphs, bsize):
            if (i + bsize) > n_graphs:
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
            for key in self.inputs:
                my_dict[self.placeholder_dict[key]] = self.inputs[key]
            # print("running training session...")
            result = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)
            # result = self.session.run([self.trainStep], feed_dict=my_dict)
            # print("sucessfully ran training session")
            loss += result[0]*bsize
            
        return loss / n_graphs
    

    def PlayGame(self, int n_traj, double eps):
        # print("Playing game!")
        cdef int N_STEP = self.cfg['N_STEP']
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)

    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        # print("Running simulator...\n")
        cdef int num_env = len(self.env_list)
        cdef int n = 0
        cdef int i
        cdef int help_func = self.cfg['help_func']
        # get intial sample
        for i in range(num_env):
            g_sample = TrainSet.Sample()
            self.env_list[i].s0(g_sample)
            
            self.g_list[i] = self.env_list[i].graph
            # print("Inserted graph!")
            # print(self.g_list[i].EdgeWeight)
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
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
                    # self.env_list[i].step(0)
                    # print("added new sample to the graph list, current length:", len(self.g_list))
            if n >= num_seq:
                break
            Random = False
            if random.uniform(0,1) >= eps:
                # print("Making prediction")
                pred = self.PredictWithCurrentQNet(self.g_list, [env.state for env in self.env_list])
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
            # print("Action lists:", [env.state for env in self.env_list])
            # print("covered set:", [env.covered_set for env in self.env_list])
            # print("reward sequence:", [env.reward_seq for env in self.env_list])
            # print("Graph details:", self.g_list[0].num_nodes)


    def TakeSnapShot(self):
        # print("Taking snapshot")
        self.session.run(self.UpdateTargetDQN)
    
    def UpdateTestDQN(self):
        self.session.run(self.updateTestDQN)

    def Evaluate(self, test_dir=None, graph_list=None, scale_factor=None):
        if scale_factor is None:
            scale_factor = self.cfg['valid_scale_fac']
        if graph_list is not None:
            g_list = graph_list
        else:
            print('Loading test graphs...')
            g_list = self.tsp_loader.load_multi_tsp_as_nx(data_dir=test_dir, 
                                                          scale_factor=scale_factor, 
                                                          start_index=0, 
                                                          end_index=None)
        if self.cfg['use_edge_probs']:
            edge_probs = self.prepare_heatmaps(path=f'{test_dir}/heatmaps', 
                                               num_cycles=None, 
                                               num_samples_per_cycle=None)
        else:
            edge_probs = None
        # reset test set
        self.ClearTestGraphs()
        self.InsertGraphs(g_list, is_test=True, edge_probs=edge_probs)
        
        self.print_test_results = False
        lengths = []
        solutions = []
        sol_times = []
        for idx in tqdm(range(len(g_list))):
            t1 = time.time()
            length, sol = self.Test(idx)
            t2 = time.time()
            sol_time = (t2 - t1)
            sol_times.append(sol_time)
            lengths.append(length)
            solutions.append(sol)
        return lengths, solutions, sol_times
    
    def prepare_new_training_graphs(self):
        cdef int NUM_MIN = self.cfg['NUM_MIN']
        cdef int NUM_MAX = self.cfg['NUM_MAX']
        n_generator = self.cfg['n_generator']
        train_path = self.cfg['train_path']
        cdef int j = self.num_train_prep_cycles
        
        self.ClearTrainGraphs()
        
        if self.cfg['train_path']:
            assert NUM_MIN == NUM_MAX
            print('Loading new training graphs...')
            g_list = self.tsp_loader.load_multi_tsp_as_nx(data_dir=train_path, 
                                                          scale_factor=self.cfg['train_scale_fac'], 
                                                          start_index=j*n_generator, 
                                                          end_index=(j+1)*n_generator)
            if self.cfg['use_edge_probs']:
                edge_probs = self.prepare_heatmaps(path=f'{train_path}/heatmaps', 
                                                   num_cycles=self.num_train_prep_cycles, 
                                                   num_samples_per_cycle=n_generator)
            else:
                edge_probs = None
        else:
            print('Generating new training graphs...')
            g_list = self.generate_graphs(n_generator)
            edge_probs = None


        self.InsertGraphs(g_list, is_test=False, edge_probs=edge_probs)
        self.num_train_prep_cycles += 1

    def PrepareValidData(self):
        cdef int n_valid = self.cfg['n_valid']
        valid_path = self.cfg['valid_path']
        if self.cfg['valid_path']:
            g_list = self.tsp_loader.load_multi_tsp_as_nx(data_dir=valid_path, 
                                                          scale_factor=self.cfg['valid_scale_fac'])
            n_valid = len(g_list)
            print(f"\nSucessfully loaded {n_valid} validation graphs!")
            if self.cfg['use_edge_probs']:
                edge_probs = self.prepare_heatmaps(path=f'{valid_path}/heatmaps', num_cycles=0, num_samples_per_cycle=n_valid)
            else:
                edge_probs = None
        else:
            print('Generating validation graphs...')
            g_list = self.generate_graphs(n_valid)
            edge_probs = None
        

        self.InsertGraphs(g_list, is_test=True, edge_probs=edge_probs)

    def generate_graphs(self, int num_graphs):
        cdef int num_min = self.cfg['NUM_MIN']
        cdef int num_max = self.cfg['NUM_MAX']
        sys.stdout.flush()
        cdef int i
        g_list = []
        for i in tqdm(range(num_graphs)):
            g = pyx_utils.gen_graph(num_min, num_max, self.cfg['g_type'])
            g_list.append(g)
        return g_list

    def prepare_heatmaps(self, path, num_cycles, num_samples_per_cycle):
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        try:
            fnames = [f for f in os.listdir(path) if os.path.isfile(f'{path}/{f}')]
            fnames.sort(key=natural_keys)
        except:
            print('\nBad heatmap directory!')
            return None
        if len(fnames) == 1:
            return np.load(f'{path}/{fnames[0]}')
        heat_map = None
        for fname in fnames:
            start_index = int(fname.split('.')[0].split('-')[0].split('_')[-1])
            end_index = int(fname.split('.')[0].split('-')[-1])
            if start_index < num_cycles * num_samples_per_cycle:
                continue
            if end_index > (num_cycles + 1) * num_samples_per_cycle:
                continue
            print(fname)
            if heat_map is not None:
                heat_map = np.concatenate([heat_map, np.load(f'{path}/{fname}')], axis=0)
            else:
                heat_map = np.load(f'{path}/{fname}')
        print(heat_map.shape)
        return heat_map

    def GenNetwork(self, g, edge_prob=None):    #networkx2four
        cdef double NN_ratio = self.cfg['NN_ratio']
        # transforms the networkx graph object into C graph object using external pyx module
        nodes = g.nodes()
        edges = g.edges()
        num_nodes = len(nodes)
        num_edges = len(edges)
        if len(edges) > 0:
            a, b = zip(*edges) 
            A = np.array(a)
            B = np.array(b)
            # edge weights
            # W = np.array([g[n][m]['weight'] for n, m in zip(a, b)])
            W = np.array([[g[n][m]['weight'] if m != n else 0.0 for m in nodes] for n in nodes])
            if edge_prob is not None:
                P = edge_prob
            else:
                P = np.zeros(shape=(num_nodes, num_nodes), dtype=np.int32)
            # node features (node position)
            try:
                F = np.array([g.nodes[k]['coord'] for k in range(len(nodes))])
            except:
                F = np.ones((len(nodes), 2))
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
            F = np.array([0])
        return graph.py_Graph(num_nodes, num_edges, A, B, W, P, F, NN_ratio)
             

    def InsertGraphs(self, g_list, is_test, edge_probs=None):
        cdef int t
        # insert prbability calculation here --> precalculate instead
        for k, g in enumerate(g_list):
            if edge_probs is not None:
                edge_prob = edge_probs[k]
            else:
                edge_prob = None
            if is_test:
                t = self.ngraph_test
                self.ngraph_test += 1
                self.TestSet.InsertGraph(t, self.GenNetwork(g, edge_prob=edge_prob))
            else:
                t = self.ngraph_train
                self.ngraph_train += 1
                self.TrainSet.InsertGraph(t, self.GenNetwork(g, edge_prob=edge_prob))

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()
    
    def SaveModel(self,model_path):
        # saves the model based on tf saver
        self.saver.save(self.session, model_path)
        print('model sucessfully saved!')

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('model sucessfully restored from file')
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

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
    

    def Build_S2VDQN(self):
        # some definitions for convenience
        cdef int node_init_dim = self.cfg['node_init_dim']
        cdef int edge_init_dim = self.cfg['edge_init_dim']
        cdef int node_embed_dim = self.cfg['node_embed_dim']
        cdef double initialization_stddev = self.cfg['initialization_stddev']
        cdef int max_bp_iter = self.cfg['max_bp_iter']
        
        # [node_init_dim, node_embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([node_init_dim, node_embed_dim], stddev=initialization_stddev), tf.float32)
        # [edge_init_dim, edge_embed_dim]
        w_e2l = tf.Variable(tf.truncated_normal([edge_init_dim, node_embed_dim], stddev=initialization_stddev), tf.float32)
        # [node_embed_dim, node_embed_dim]
        p_node_conv1 = tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32) 
        # [node_embed_dim, node_embed_dim]
        trans_node_1 = tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32)
        # [node_embed_dim, node_embed_dim]
        trans_node_2 = tf.Variable(tf.truncated_normal([node_embed_dim, node_embed_dim], stddev=initialization_stddev), tf.float32)

        #[reg_hidden, 1]
        if self.cfg['REG_HIDDEN'] > 0:

            # [node_embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([node_embed_dim, self.cfg['REG_HIDDEN']], 
                                                        stddev=initialization_stddev), tf.float32)

            #[reg_hidden, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.cfg['REG_HIDDEN'], 1], stddev=initialization_stddev), tf.float32)
            
            #[reg_hidden, 1]
            last_w = h2_weight
        else:
            # [node_embed_dim, 1]
            h1_weight = tf.Variable(tf.truncated_normal([node_embed_dim, 1], 
                                                        stddev=initialization_stddev), tf.float32)
            last_w = h1_weight

        # [node_cnt, node_init_dim] * [node_init_dim, node_embed_dim] = [node_cnt, node_embed_dim], not sparse
        node_init = tf.matmul(tf.cast(self.placeholder_dict['node_input'], tf.float32), w_n2l)
        cur_node_embed = tf.nn.relu(node_init)
        
        # [edge_cnt, edge_dim] * [edge_dim, embed_dim] = [edge_cnt, embed_dim]
        edge_init = tf.matmul(tf.cast(self.placeholder_dict['edge_input'], tf.float32), w_e2l)
        ################### GNN start ###################
        cdef int lv = 0
        while lv < max_bp_iter:
            lv = lv + 1
            
            msg_linear = tf.matmul(cur_node_embed, p_node_conv1)
            n2e_linear = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['n2esum_param_0'], tf.float32), msg_linear)
            edge_rep = tf.math.add(n2e_linear, edge_init)
            edge_rep = tf.nn.relu(edge_rep)
            e2n = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['e2nsum_param'], tf.float32), edge_rep)
           
            node_linear_1 = tf.matmul(e2n, trans_node_1)
            node_linear_2 = tf.matmul(cur_node_embed, trans_node_2)
            node_linear = tf.math.add(node_linear_1, node_linear_2)
            cur_node_embed = tf.nn.relu(node_linear)
        ################### GNN end ###################
        # cur_state_embed = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['subgsum_param'], tf.float32), cur_node_embed)
        # [batch_size, node_cnt] * [node_cnt, node_embed_dim] = [batch_size, node_embed_dim]
        action_embed = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['action_select'], tf.float32), cur_node_embed)
        # embed_s_a = tf.concat([action_embed, cur_state_embed], axis=1)
        embed_s_a = action_embed
        last_output = embed_s_a
        
        if self.cfg['REG_HIDDEN'] > 0:
            # [batch_size, (2)node_embed_dim] * [(2)node_embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            
            # [batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)
        
        # [batch_size, reg_hidden] * [reg_hidden, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        loss = tf.losses.mean_squared_error(self.placeholder_dict['target'], q_pred)

        trainStep = tf.compat.v1.train.AdamOptimizer(self.cfg['LEARNING_RATE']).minimize(loss)

        # rep_state = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['rep_global'], tf.float32), cur_state_embed)
        # embed_s_a_all = tf.concat([cur_node_embed, rep_state], axis=1)
        
        # [node_cnt, node_embed_dim]
        embed_s_a_all = cur_node_embed
        last_output = embed_s_a_all
        
        if self.cfg['REG_HIDDEN'] > 0:
            # [node_cnt, node_embed_dim] * [node_embed_dim, reg_hidden] = [node_cnt, reg_hidden], dense
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            
            # [node_cnt, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # [node_cnt, reg_hidden] * [reg_hidden, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables() 
    

    def fit_with_prioritized(self, tree_idx, ISWeights, g_list, covered, actions, list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0, n_graphs, self.cfg['BATCH_SIZE']):
            bsize = self.cfg['BATCH_SIZE']
            if (i + bsize) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            for key in self.inputs:
                my_dict[self.placeholder_dict[key]] = self.inputs[key]
            
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            result = self.session.run([self.trainStep, self.TD_errors, self.loss], feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)
