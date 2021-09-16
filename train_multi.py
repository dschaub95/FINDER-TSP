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



atoi = lambda text : int(text) if text.isdigit() else text
natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]

multi_train_configs_path = './train_configs/multi_train'
fnames = [f for f in os.listdir(multi_train_configs_path) if os.path.isfile(f'{multi_train_configs_path}/{f}')]
fnames.sort(key=natural_keys)
for fname in fnames:
    train_config_file_path = f'{multi_train_configs_path}/{fname}'
    api = FINDER_API(train_config_file_path)
    try:
        api.train()
        api.save_train_results(model_name=fname.split('.')[0])
    except:
        print("Error")