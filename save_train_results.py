import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import re
import tqdm
from shutil import copy

from FINDER_API import FINDER_API
from py_utils.FINDER_train_utils import *



cur_config_path = './models/tsp_2d/nrange_20_20'
cur_config_file_path = f'{cur_config_path}/current_config.txt'
api = FINDER_API(cur_config_file_path)
if len(sys.argv) > 1:
    api.save_train_results(model_name=sys.argv[1], num_best=10)
else:
    api.save_train_results(model_name='model', num_best=10)
