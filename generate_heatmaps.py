import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import torch
import argparse
print(torch.cuda.is_available())
from attgcn_preprocessor.attgcn_api import ATTGCN_API
import numpy as np
from py_utils.TSP_loader import TSP_loader

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_NODES", type=int, default=50)
    parser.add_argument("--data_path", type=str, default='./data/test_sets/synthetic_n_50_1000')
    parser.add_argument("--config_path", type=str, default='./attgcn_preprocessor/configs/tsp20.json')
    opts = parser.parse_known_args()[0]

NUM_NODES = opts.NUM_NODES
data_name = opts.data_path.split('/')[-1]

gcn_api = ATTGCN_API(config_path=opts.config_path)
gcn_api.clear_gpu_memory()
gcn_api.init_net()
gcn_api.load_ckpt()
gcn_api.clear_gpu_memory()

tsp_loader = TSP_loader()
g_list = tsp_loader.load_multi_tsp_as_nx(data_dir=f'{opts.data_path}', scale_factor=0.000001, start_index=0)
gcn_api.load_nx_test_set(graph_list=g_list, num_nodes=NUM_NODES)

result = gcn_api.run_test(batch_size=100)
save_path = f'{opts.data_path}/heatmaps'
create_dir(save_path)
file_name = f'{data_name}_heatmaps.npy'
np.save(file=f'{save_path}/{file_name}', arr=np.array(result))