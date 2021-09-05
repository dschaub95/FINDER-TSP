
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from FINDER_API import FINDER_API
from py_utils.FINDER_test_utils import *
import os
import tqdm
import numpy as np

# insert function which automatically searches for best model with specified name
model_name = 'AGNN_no_help_only_start_end_+_edge_prob_2021-09-01_08:31:56_len_100233'
model_name = 'AGNN_no_help_only_start_end_2021-08-30_07:58:58_len_102569'
model_path = f'saved_models/tsp_2d/nrange_20_20/{model_name}'

config_path = f'{model_path}/config.txt'
api = FINDER_API(config_path=config_path)

# get best checkpoint path
best_ckpt_file_path = get_best_ckpt(model_path)
# load checkpoint into Finder
api.load_model(ckpt_path=best_ckpt_file_path)

data_dir = 'data/valid_sets/synthetic_nrange_20_20_100'
data_dir_0 = 'data/test_sets/synthetic_n_20_1000'
data_dir_1 = 'data/test_sets/synthetic_n_50_1000'
test_dirs = [data_dir_0, data_dir_1]

lengths, solutions, sol_times = api.run_test(data_dir, scale_factor=0.000001)
approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, lengths)
print(mean_approx_ratio)