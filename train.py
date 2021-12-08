import argparse
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import sys
sys.path.insert(1, 'model/')
from model.FINDER_API import FINDER_API
from py_utils.FINDER_train_utils import *
from py_utils.FINDER_test_utils import *
from test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/train_config.txt')
    parser.add_argument("--model_name", type=str, default='model')
    parser.add_argument("--logging", type=bool, default=True)
    opts = parser.parse_known_args()[0]
    
    if not opts.logging:
        os.environ["WANDB_MODE"] = 'disabled'
    
    api = FINDER_API(opts.config_file)
    api.train()
    save_dir = api.save_train_results(model_name=opts.model_name, num_best=10, save_all_ckpts=False)
    # specify test sets
    test_sets = ['data/test_sets/synthetic_n_20_1000', 'data/test_sets/synthetic_n_50_1000']
    test_set_names = ['tsp20_data', 'tsp50_data']
    # restart session
    api.DQN.start_tf_session()
    test(api=api, model_path=save_dir, 
         test_sets=test_sets, 
         test_set_names=test_set_names, 
         search_strategy='greedy')