import numpy as np
import os
import time
from shutil import copy
from distutils.util import strtobool
from tqdm import tqdm
from datetime import datetime
from distutils.dir_util import copy_tree
from FINDER import FINDER

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
    def __init__(self, config_path=None):
        # read input config
        config = read_config(config_path)
        self.config_path = config_path
        # initialize FINDER config with default values
        self.cfg = dict()
        
        # Environment parameters
        self.cfg['help_func'] = 0 # whether to use helper function during node insertion process, which inserts node into best position in current partial tour
        self.cfg['reward_normalization'] = 'max'
        self.cfg['reward_sign'] = -1
        self.cfg['fix_start_node'] = 1

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
        self.cfg['state_embed_dim'] = 64
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
        self.cfg['dropout_rate'] = 0.1

        # training set specifications
        self.cfg['g_type'] = 'tsp_2d'
        self.cfg['NUM_MIN'] = 15
        self.cfg['NUM_MAX'] = 20
        self.cfg['NN_ratio'] = 1.0
        self.cfg['n_generator'] = 1000
        self.cfg['train_path'] = None
        self.cfg['train_scale_fac'] = 0.000001

        # Decoder hyperparameters
        self.cfg['decoder_type'] = 0
        self.cfg['REG_HIDDEN'] = 32
        
        # search startegy/ testing parameters
        self.cfg['search_strategy'] = 'greedy'
        self.cfg['beam_width'] = 64
        self.cfg['sample_steps'] = 64
        self.cfg['test_batch_size'] = 256

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
        self.cfg['use_edge_probs'] = 0
        self.cfg['probability_construction'] = 0

        # validation set info
        self.cfg['valid_path'] = 'valid_sets/synthetic_nrange_20_20_100/'
        self.cfg['valid_scale_fac'] = 0.000001
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
        del self.DQN
        config = read_config(config_path)
        for key in self.cfg:         
            try:
                self.cfg[key] = config[key]
                # print("Sucessfully loaded key '{}' from external config file!".format(key))
            except:
                print("Error when loading key '{}' from external config file!".format(key))
                print("Using default value {} instead!".format(self.cfg[key]))
        self.DQN = FINDER(config=self.cfg)

    def train(self, save_config=True, save_architecture=True):
        print(self.cfg)
        if save_config:
            self.save_cur_config()
        if save_architecture:  
            self.save_architecture()
        self.DQN.Train()

    def save_cur_config(self):
        g_type = self.cfg['g_type']
        NUM_MIN = self.cfg['NUM_MIN']
        NUM_MAX = self.cfg['NUM_MAX']
        config_file = 'current_config.txt'
        config_save_dir = f'models/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}'
        try:
            copy(self.config_path, f'{config_save_dir}/{config_file}')
        except:
            print("Error when saving current config file!")
    
    def save_architecture(self):
        g_type = self.cfg['g_type']
        NUM_MIN = self.cfg['NUM_MIN']
        NUM_MAX = self.cfg['NUM_MAX']
        architecture_save_dir = f'models/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}/architecture'
        self.create_dir(architecture_save_dir)
        architecture_paths = ['.', '.', '.', 'src/lib', 'src/lib']
        architecture_files = ['FINDER.pyx', 'PrepareBatchGraph.pyx', 'PrepareBatchGraph.pxd', 'PrepareBatchGraph.cpp', 'PrepareBatchGraph.h']
        for k, a_file in enumerate(architecture_files):
            a_path = architecture_paths[k]
            try:
                copy(f'{a_path}/{a_file}', f'{architecture_save_dir}/{a_file}')
            except:
                print(f"Error when saving current architecture file {a_file}!")
        # copy dqn files
        copy_tree('dqn', f'{architecture_save_dir}/dqn')

    def load_model(self, ckpt_path):
        self.DQN.LoadModel(ckpt_path)

    def run_test(self, test_dir=None, graph_list=None, scale_factor=0.000001):
        print(self.cfg)
        lengths, solutions, sol_times = self.DQN.Evaluate(test_dir=test_dir, g_list=graph_list, scale_factor=scale_factor)
        return lengths, solutions, sol_times
    
    def save_train_results(self, model_name='', save_architecture=True, save_all_ckpts=True, num_best=1):
        g_type = self.cfg['g_type']
        NUM_MIN = self.cfg['NUM_MIN']
        NUM_MAX = self.cfg['NUM_MAX']
        dt_string = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        checkpoint_suffixes = ['data-00000-of-00001', 'index', 'meta']
        base_path = f'./models/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}'
        valid_file = f'Validation_{NUM_MIN}_{NUM_MAX}.csv'
        valid_file_path = f'{base_path}/{valid_file}'
        
        valid_ratios = [float(line.split(' ')[1]) for line in open(valid_file_path)]
        iterations = [int(line.split(' ')[0]) for line in open(valid_file_path)]

        min_tour_length = ''.join(str(np.round(min(valid_ratios), 5)).split('.'))
        
        save_dir = f'./saved_models/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}/{model_name}_{dt_string}_len_{min_tour_length}'
        self.create_dir(save_dir)

        print("Saving validation file...")
        copy(valid_file_path, f'{save_dir}/{valid_file}')
        
        print("Saving Loss file...")
        loss_file = f'Loss_{NUM_MIN}_{NUM_MAX}.csv'
        copy(f'{base_path}/{loss_file}', f'{save_dir}/{loss_file}')
        
        if save_all_ckpts:
            print("Saving all checkpoint files...")
            ckpt_save_dir = f'{save_dir}/checkpoints'
            self.create_dir(ckpt_save_dir)
            for iter in iterations:
                self.save_checkpoint_files(ckpt_save_dir, iter)
            
        print("Saving best checkpoint files seperately...")
        best_ckpt_save_dir = f'{save_dir}/best_checkpoint'
        self.create_dir(best_ckpt_save_dir)
        ordered_ckpts = sorted(zip(iterations, valid_ratios), key=lambda tup:tup[1], reverse=False)
        for k in range(min(num_best,len(ordered_ckpts))):
            self.save_checkpoint_files(best_ckpt_save_dir, ordered_ckpts[k][0], rank=k+1)
        
        print("Saving config file...")
        config_file = 'current_config.txt'
        copy(f'{base_path}/{config_file}', f'{save_dir}/config.txt')

        if save_architecture:
            architecture_save_dir = f'{save_dir}/architecture'
            self.create_dir(architecture_save_dir)
            print('Saving FINDER.pyx, PrepareBatchGraph.pyx, PrepareBatchGraph.pxd, PrepareBatchGraph.h, PrepareBatchGraph.cpp...')
            architecture_files = ['FINDER.pyx', 'PrepareBatchGraph.pyx', 'PrepareBatchGraph.pxd', 'PrepareBatchGraph.cpp', 'PrepareBatchGraph.h']
            for a_file in architecture_files:
                copy(f'{base_path}/architecture/{a_file}', f'{architecture_save_dir}/{a_file}')
            # copy dqn files
            try:
                copy_tree(f'{base_path}/architecture/dqn', f'{architecture_save_dir}/dqn')
            except:
                print("Couldn't save dqn files!")


    def save_checkpoint_files(self, ckpt_save_dir, iter, rank=''):
        g_type = self.cfg['g_type']
        NUM_MIN = self.cfg['NUM_MIN']
        NUM_MAX = self.cfg['NUM_MAX']
        checkpoint_suffixes = ['data-00000-of-00001', 'index', 'meta']
        for suffix in checkpoint_suffixes:
            old_ckpt_name = f'nrange_{NUM_MIN}_{NUM_MAX}_iter_{iter}.ckpt.{suffix}'
            new_ckpt_name = f'nrange_{NUM_MIN}_{NUM_MAX}_iter_{iter}_rank_{rank}.ckpt.{suffix}'
            ckpt_path = f'./models/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}/checkpoints/{old_ckpt_name}'
            copy(ckpt_path, f'{ckpt_save_dir}/{new_ckpt_name}')
    
    def create_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    