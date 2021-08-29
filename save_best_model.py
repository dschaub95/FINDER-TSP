# -*- coding: utf-8 -*-

from FINDER import FINDER
from FINDER_train_utils import *


if __name__=="__main__":
    print("Starting FINDER...")
    final_config_path = 'models/tsp_2d/current_config.txt'
    dqn = FINDER(train_config_path=final_config_path)
    save_best_model(dqn, config_path=final_config_path)


    
