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


def test(api, model_path, test_sets, test_set_names, search_strategy='greedy', sample_steps=64, beam_width=64, test_batch_size=512):
    model_name = model_path.split('/')[-1]
    # get best checkpoint path
    best_ckpt_file_path = get_best_ckpt(model_path)
    # load checkpoint into Finder
    api.load_model(ckpt_path=best_ckpt_file_path)
    api.DQN.cfg.update({'search_strategy': search_strategy}, allow_val_change=True)
    api.DQN.cfg.update({'sample_steps': sample_steps}, allow_val_change=True)
    api.DQN.cfg.update({'beam_width': beam_width}, allow_val_change=True)
    api.DQN.cfg.update({'test_batch_size': test_batch_size}, allow_val_change=True)
    
    # create nice name
    if search_strategy == "greedy":
        suffix = f'{search_strategy}'
    elif search_strategy == "beam_search" or search_strategy == "beam_search+":
        suffix = f'{search_strategy}_{beam_width}'
    elif search_strategy == "sampling":
        suffix = f'{search_strategy}_{sample_steps}'
    else:
        suffix = f'{search_strategy}_{sample_steps}'
    
    for k, data_dir in enumerate(test_sets):
        # run test
        lengths, solutions, sol_times = api.run_test(test_dir=data_dir, scale_factor=0.000001)
        mean_sol_time = np.mean(sol_times)
        approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, lengths)
        print(mean_approx_ratio)
        wandb.log({f"performance/{test_set_names[k]}/mean_approximation_ratio": mean_approx_ratio})
        wandb.log({f"performance/{test_set_names[k]}/mean_solution_time": mean_sol_time})
        # save test results
        save_approx_ratios(data_dir, approx_ratios, model_name, suffix=suffix)
        save_solutions(data_dir, solutions, model_name, suffix=suffix)
        save_lengths(data_dir, lengths, model_name, suffix=suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='')
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
    # make test set list
    test_sets = [opts.__dict__[key] for key in opts.__dict__ if 'data' in key and opts.__dict__[key] != '']
    test_set_names = [key for key in opts.__dict__ if 'data' in key and opts.__dict__[key] != '']
    config_path = f'{opts.model_path}/config.txt'
    if not opts.logging:
        os.environ["WANDB_MODE"] = 'disabled'
    api = FINDER_API(config_path=config_path)

    test(api=api, 
         model_path=opts.model_path, 
         search_strategy=opts.search_strategy, 
         sample_steps=opts.sample_steps, 
         beam_width=opts.beam_width, 
         test_batch_size=opts.test_batch_size, 
         test_sets=test_sets, 
         test_set_names=test_set_names)