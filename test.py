
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from FINDER_API import FINDER_API
from FINDER_test_utils import *
import os
import tqdm
import numpy as np

# insert function which automatically searches for best model with specified name
model_name = 'AGNN_default_len_10113392076640768'
model_path = f'saved_models/tsp_2d/nrange_15_20/{model_name}'

config_path = f'{model_path}/config.txt'
api = FINDER_API(config_path=config_path)

# get best checkpoint path
best_ckpt_file_path = get_best_ckpt(model_path)
# load checkpoint into Finder
api.load_model(ckpt_path=best_ckpt_file_path)

data_dir = 'data/test_sets/tsp_min-n=15_max-n=20_num-graph=1000_type=random/'
data_dir_0 = 'data/test_sets/tsp_min-n=15_max-n=20_num-graph=1000_type=random/'
data_dir_1 = 'data/test_sets/tsp_min-n=40_max-n=50_num-graph=1000_type=random/'
data_dir_2 = 'data/test_sets/tsp_min-n=50_max-n=100_num-graph=1000_type=random/'
data_dir_3 = 'data/test_sets/tsp_min-n=100_max-n=200_num-graph=1000_type=random/'

test_dirs = [data_dir_0, data_dir_1, data_dir_2]

# use beam search during testing
search_strategy = 'beam_search+'
beam_width = 64

api.DQN.cfg['search_strategy'] = search_strategy
api.DQN.cfg['beam_width'] = beam_width
api.DQN.cfg['BATCH_SIZE'] = 64

suffix = f'{search_strategy}_{beam_width}'

mean_approx_ratios = []
for data_dir in test_dirs[0::]:
    # load test data
    graph_list, fnames = prepare_testset_S2VDQN(data_dir)
    # run test
    lengths, solutions, sol_times = api.run_test(graph_list)
    approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, fnames, lengths)
    print(mean_approx_ratio)
    # save test results
    save_approx_ratios(data_dir, fnames, approx_ratios, model_name, suffix=suffix)
    save_solutions(data_dir, fnames, solutions, model_name, suffix=suffix)
    save_lengths(data_dir, fnames, lengths, model_name, suffix=suffix)