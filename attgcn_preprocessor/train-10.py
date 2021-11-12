import os

import numpy as np

import torch
print(torch.cuda.is_available())
from config import *
from utils.process import *

notebook_mode = True
viz_mode = False

# model-parameter
# config_path = "configs/tsp20.json"
config_path = "configs/tsp10.json"
config = get_config(config_path)


if viz_mode==False:

    # tsp20--model
    net = main(config, pretrained=False, patience=50, lr_scale=0.001, random_neighbor=False)