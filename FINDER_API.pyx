from FINDER import FINDER
import numpy as np
import os
import time
from shutil import copy
from distutils.util import strtobool
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# test and evaluation functionalities or basically any interaction with FINDER in here

def read_config(config_path):
    with open(config_path) as f:
        data = f.read()
    data = data.replace(" ", "").split('\n')
    data = [element for element in data if not len(element) == 0]
    # delete comment lines
    data = [element for element in data if not element[0] == '#']
    # ignore inline comments
    data = [element.split('#')[0] for element in data]
    # delete string literals
    data = [element.replace("'", "") for element in data]
    data_dict = dict(substr.split('=') for substr in data)
    # conversion to correct data type
    for key in data_dict:
        if '.' in data_dict[key]:
            # conversion to float
            try:
                data_dict[key] = float(data_dict[key])
            except:
                pass
        else:
            # conversion to int or boolean int (0,1)
            try:
                data_dict[key] = int(data_dict[key])
            except:
                try:
                    data_dict[key] = strtobool(data_dict[key])
                except:
                    pass
    return data_dict


class FINDER_API:
    def __init__(self, train_config_path=None):
        # read input config
        config = read_config(train_config_path)

        # initialize FINDER config with default values
        self.cfg = dict()
        
        # Environment parameters
        self.cfg['help_func'] = 0 # whether to use helper function during node insertion process, which inserts node into best position in current partial tour
        self.cfg['reward_normalization'] = 'max'
        self.cfg['reward_sign'] = -1

        # GNN hyperparameters
        self.cfg['net_type'] = 'AGNN'
        self.cfg['max_bp_iter'] = 3 # number of aggregation steps in GNN equals number of layers
        self.cfg['aggregatorID'] = 0 # 0:sum; 1:mean; 2:GCN; 3:edge weight based sum
        self.cfg['node_init_dim'] = 4 # number of initial node features
        self.cfg['edge_init_dim'] = 4 # number of initial edge features
        self.cfg['state_init_dim'] = 4
        # self.cfg['state_embed_dim'] = 64
        self.cfg['node_embed_dim'] = 64
        self.cfg['edge_embed_dim'] = 64
        self.cfg['embeddingMethod'] = 2
        self.cfg['ignore_covered_edges'] = 0 
        self.cfg['selected_nodes_inclusion'] = 2
        self.cfg['focus_start_end_node'] = 1
        self.cfg['state_representation'] = 0
        
        # general training hyperparameters
        self.cfg['IsHuberloss'] = 0
        self.cfg['BATCH_SIZE'] = 64
        self.cfg['initialization_stddev'] = 0.01
        self.cfg['MAX_ITERATION'] = 150000
        self.cfg['LEARNING_RATE'] = 0.001
        self.cfg['Alpha'] = 0.001
        self.cfg['save_interval'] = 300
        self.cfg['num_env'] = 1
        self.cfg['dropout_rate'] = 0.2

        # training set specifications
        self.cfg['g_type'] = 'tsp_2d'
        self.cfg['NUM_MIN'] = 15
        self.cfg['NUM_MAX'] = 20
        self.cfg['NN_ratio'] = 1.0
        self.cfg['n_generator'] = 1000

        # Decoder hyperparameters
        self.cfg['decoder'] = 0
        self.cfg['REG_HIDDEN'] = 32
        
        # search startegy
        self.cfg['search_strategy'] = 'greedy'
        self.cfg['beam_width'] = 64

        # Q-learning hyperparameters
        self.cfg['IsDoubleDQN'] = 0
        self.cfg['N_STEP'] = 5
        self.cfg['GAMMA'] = 1.0
        self.cfg['UPDATE_TIME'] = 1000
        self.cfg['eps_start'] = 1.0
        self.cfg['eps_end'] = 0.05
        self.cfg['eps_step'] = 10000.0
        self.cfg['MEMORY_SIZE'] = 150000
        self.cfg['one_step_encoding'] = 0

        # validation set info
        self.cfg['valid_path'] = 'valid_sets/synthetic_nrange_15_20_200/'
        self.cfg['valid_scale_fac'] = 0.0001
        self.cfg['n_valid'] = 200

        # (hyper)parameters for prioritized replay sampling
        self.cfg['IsPrioritizedSampling'] = 0
        self.cfg['epsilon'] = 0.0000001
        self.cfg['alpha'] = 0.6
        self.cfg['beta'] = 0.4
        self.cfg['beta_increment_per_sampling'] = 0.001
        self.cfg['TD_err_upper'] = 1.0

        # overwrite each config key value with value from external config file (where possible)
        for key in self.cfg:         
            try:
                self.cfg[key] = config[key]
                # print("Sucessfully loaded key '{}' from external config file!".format(key))
            except:
                print("Error when loading key '{}' from external config file!".format(key))
                print("Using default value {} instead!".format(self.cfg[key]))

        self.DQN = FINDER(config=self.cfg)
    
    def load_test_config(self, config_path):
        # if test config provided overwrite corresponding values
        test_config = read_config(config_path)
        for key in self.cfg:         
            try:
                self.cfg[key] = test_config[key]
                print(f"Sucessfully overwrote key '{key}' with value {test_config[key]} from external test config file!")
            except:
                pass
        self.DQN.cfg = self.cfg
        
    def reinit_FINDER(self, config_path):
        config = read_config(config_path)
        for key in self.cfg:         
            try:
                self.cfg[key] = config[key]
                # print("Sucessfully loaded key '{}' from external config file!".format(key))
            except:
                print("Error when loading key '{}' from external config file!".format(key))
                print("Using default value {} instead!".format(self.cfg[key]))
        self.DQN = FINDER(config=self.cfg)

    def train(self):
        self.DQN.Train()

    def load_model(self, model_path):
        self.DQN.LoadModel(self, model_path)

    def run_test(self, dqn, graph_list):
        lengths = []
        solutions = []
        sol_times = []
        for g in tqdm(graph_list):
            len, sol, time = self.evaluate(g)
            lengths.append(len)
            solutions.append(sol)
            sol_times.append(time)
        return lengths, solutions, sol_times

    def evaluate(self, graph):
        # reset test set
        self.DQN.ClearTestGraphs()
        self.DQN.InsertGraph(graph, is_test=True)
        t1 = time.time()
        len, sol = self.DQN.Test(0, print_out=False)
        t2 = time.time()
        sol_time = (t2 - t1)
        return len, sol, sol_time
