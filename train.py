
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf
# import random
# import numpy as np
# # tf.compat.v1.set_random_seed(73)
# random.seed(7)
# np.random.seed(42)
import sys

import tqdm
from shutil import copy

from FINDER_API import FINDER_API
from py_utils.FINDER_train_utils import *

train_configs_path = './train_configs'
train_config_file = 'train_config.txt'
train_config_file_path = f'{train_configs_path}/{train_config_file}'
api = FINDER_API(train_config_file_path)
api.train()