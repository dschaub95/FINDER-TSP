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
    opts = parser.parse_known_args()[0]

    api = FINDER_API(opts.config_file)
    api.train()
    api.save_train_results(model_name=opts.model_name, num_best=10, save_all_ckpts=False)
