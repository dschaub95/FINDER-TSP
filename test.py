import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.insert(1, 'model/')
from model.FINDER_API import FINDER_API
from py_utils.FINDER_test_utils import *
import argparse
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='model')
    parser.add_argument("--valid_data", type=str, default='data/valid_sets/synthetic_nrange_20_20_100')
    parser.add_argument("--tsp20_data", type=str, default='data/test_sets/synthetic_n_20_1000')
    parser.add_argument("--tsp50_data", type=str, default='')
    parser.add_argument("--tsp100_data", type=str, default='')
    parser.add_argument("--tsp200_data", type=str, default='')
    parser.add_argument("--search_strategy", type=str, default='greedy')
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--sample_steps", type=int, default=32)
    parser.add_argument("--beam_width", type=int, default=64)
    parser.add_argument("--logging", type=bool, default=False)
    opts = parser.parse_known_args()[0]
    # make dataset list
    datasets = [opts.__dict__[key] for key in opts.__dict__ if 'data' in key and opts.__dict__[key] != '']
    dataset_names = [key for key in opts.__dict__ if 'data' in key and opts.__dict__[key] != '']
    # model_path = f'saved_models/tsp_2d/nrange_20_20/{opts.model_name}'
    model_path = f'test_models/{opts.model_name}'
    config_path = f'{model_path}/config.txt'
    if not opts.logging:
        os.environ["WANDB_MODE"] = 'disabled'
    api = FINDER_API(config_path=config_path)

    # get best checkpoint path
    best_ckpt_file_path = get_best_ckpt(model_path)
    # load checkpoint into Finder
    api.load_model(ckpt_path=best_ckpt_file_path)

    api.DQN.cfg.update({'search_strategy': opts.search_strategy}, allow_val_change=True)
    api.DQN.cfg.update({'sample_steps': opts.sample_steps}, allow_val_change=True)
    api.DQN.cfg.update({'beam_width': opts.beam_width}, allow_val_change=True)
    api.DQN.cfg.update({'test_batch_size': opts.test_batch_size}, allow_val_change=True)
    
    # create nice name
    if opts.search_strategy == "greedy":
        suffix = f'{opts.search_strategy}'
    elif opts.search_strategy == "beam_search" or opts.search_strategy == "beam_search+":
        suffix = f'{opts.search_strategy}_{opts.beam_width}'
    elif opts.search_strategy == "sampling":
        suffix = f'{opts.search_strategy}_{opts.sample_steps}'
    else:
        suffix = f'{opts.search_strategy}_{opts.sample_steps}'
    
    mean_approx_ratios = []
    for k, data_dir in enumerate(datasets):
        # run test
        lengths, solutions, sol_times = api.run_test(test_dir=data_dir, scale_factor=0.000001)
        approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, lengths)
        mean_approx_ratios.append(mean_approx_ratio)
        print(mean_approx_ratio)
        wandb.log({f"performance/{dataset_names[k]}/approximation_ratio": mean_approx_ratio})
        # save test results
        save_approx_ratios(data_dir, approx_ratios, opts.model_name, suffix=suffix)
        save_solutions(data_dir, solutions, opts.model_name, suffix=suffix)
        save_lengths(data_dir, lengths, opts.model_name, suffix=suffix)