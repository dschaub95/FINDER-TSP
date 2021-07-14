# -*- coding: utf-8 -*-

from FINDER import FINDER
from FINDER_train_utils import *


if __name__=="__main__":
    print("Starting FINDER...")
    config_path = 'train_configs/default_config.txt'
    dqn = FINDER(config_path=config_path)
    save_best_model(dqn, config_path=config_path)


    
