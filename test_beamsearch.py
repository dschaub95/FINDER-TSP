import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from FINDER_API import FINDER_API
from py_utils.FINDER_test_utils import *
import sys
import numpy as np

if len(sys.argv) >= 2:
    model_name = sys.argv[1]
    beam_width = int(sys.argv[2])
elif len(sys.argv) == 1:
    model_name = sys.argv[1]
else:
    model_name = None
# model_path = f'saved_models/tsp_2d/nrange_20_20/{model_name}'
model_path = f'test_models/{model_name}'

config_path = f'{model_path}/config.txt'
api = FINDER_API(config_path=config_path)

# get best checkpoint path
best_ckpt_file_path = get_best_ckpt(model_path)
# load checkpoint into Finder
api.load_model(ckpt_path=best_ckpt_file_path)

data_dir_valid = 'data/valid_sets/synthetic_nrange_20_20_100'
data_dir_0 = 'data/test_sets/synthetic_n_20_1000'
test_dirs = [data_dir_0]

# use beam search during testing
search_strategy = 'beam_search+'
batch_size = 512

api.DQN.cfg['search_strategy'] = search_strategy
api.DQN.cfg['beam_width'] = beam_width
api.DQN.cfg['test_batch_size'] = batch_size

suffix = f'{search_strategy}_{beam_width}'

# sanity check
lengths, solutions, sol_times = api.run_test(test_dir=data_dir_valid, scale_factor=0.000001)
approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir_valid, lengths)
print(mean_approx_ratio)

mean_approx_ratios = []
for data_dir in test_dirs[0:1]:
    # run test
    lengths, solutions, sol_times = api.run_test(test_dir=data_dir, scale_factor=0.000001)
    approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, lengths)
    mean_approx_ratios.append(mean_approx_ratio)
    print(mean_approx_ratio)
    # save test results
    save_approx_ratios(data_dir, approx_ratios, model_name, suffix=suffix)
    save_solutions(data_dir, solutions, model_name, suffix=suffix)
    save_lengths(data_dir, lengths, model_name, suffix=suffix)