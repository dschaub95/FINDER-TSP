import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.insert(1, 'model/')
from shutil import copy

from model.FINDER_API import FINDER_API

from py_utils.FINDER_train_utils import *



cur_config_path = './logs/tsp_2d/nrange_20_20'
cur_config_file_path = f'{cur_config_path}/config.txt'
api = FINDER_API(cur_config_file_path)
if len(sys.argv) == 2:
    api.save_train_results(model_name=sys.argv[1], num_best=10, save_all_ckpts=False)
elif len(sys.argv) > 2:
    api.save_train_results(model_name=sys.argv[1], num_best=10, save_all_ckpts=True)
else:
    api.save_train_results(model_name='model', num_best=10, save_all_ckpts=False)
