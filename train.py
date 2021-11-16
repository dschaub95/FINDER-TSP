import argparse
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.insert(1, 'model/')
from model.FINDER_API import FINDER_API
from py_utils.FINDER_train_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/train_config.txt')
    parser.add_argument("--model_name", type=str, default='model')
    parser.add_argument("--log_path", type=str, default='.logs/')
    parser.add_argument("--save_path", type=str, default='')
    opts = parser.parse_args()

    api = FINDER_API(opts.config_file)
    api.train()
    api.save_train_results(model_name=opts.model_name, num_best=10, save_all_ckpts=False)

