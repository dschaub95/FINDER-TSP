# -*- coding: utf-8 -*-

from FINDER import FINDER
import sys
from shutil import copy
from FINDER_train_utils import *

def main():
    print("Starting FINDER...")
    config_path = 'train_configs/train_config.txt'
    copy(config_path, 'models/tsp_2d/current_config.txt')
    dqn = FINDER(config_path=config_path)
    dqn.Train()
    if str(sys.argv[1]) == 'save':
        save_best_model(dqn, config_path=config_path)

if __name__=="__main__":
    main()

