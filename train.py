# -*- coding: utf-8 -*-

from FINDER import FINDER
import sys
from shutil import copy
from FINDER_train_utils import *

def main():
    print("Starting FINDER...")
    start_config_path = 'train_configs/train_config.txt'
    final_config_path = 'models/tsp_2d/current_config.txt'
    copy(start_config_path, final_config_path)
    dqn = FINDER(config_path=final_config_path)
    dqn.Train()
    if str(sys.argv[1]) == 'save':
        save_best_model(dqn, config_path=final_config_path)

if __name__=="__main__":
    main()

