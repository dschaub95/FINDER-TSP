
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from FINDER_API import FINDER_API
from py_utils.FINDER_train_utils import *

if len(sys.argv) > 1:
    train_config_file_path = sys.argv[1]
    train_config_file_name = train_config_file_path.split('/')[-1].split('.')[0]
else:
    train_configs_path = './train_configs'
    train_config_file = 'train_config.txt'
    train_config_file_path = f'{train_configs_path}/{train_config_file}'

api = FINDER_API(train_config_file_path)
api.train()

if len(sys.argv) > 1:
    api.save_train_results(model_name=train_config_file_name, num_best=100, save_all_ckpts=False)
else:
    model_name = 'model'
    api.save_train_results(model_name=model_name, num_best=100, save_all_ckpts=False)
