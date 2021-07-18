import os
import numpy as np
import pandas as pd
import tsplib95
import networkx as nx
from tqdm import tqdm
import re

def prepare_testset_FINDER(data_dir, scale_factor=0.000001):
    graph_list = []

    atoi = lambda text : int(text) if text.isdigit() else text
    natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]

    fnames = os.listdir(data_dir)
    fnames.sort(key=natural_keys)
    print("Loading test graphs...")
    for fname in tqdm(fnames):
        try:
            if not '.tsp' in fname or '.sol' in fname:
                continue
            problem = tsplib95.load(data_dir + fname)
            g = problem.get_graph()
            
        except:
            print('Error, while loading file {}'.format(fname))
        # remove edges from one node to itself
        ebunch=[(k,k) for k in g.nodes]
        g.remove_edges_from(ebunch)
        # reset node index to start at zero
        mapping = {k:i for i,k in enumerate(g.nodes)}
        g = nx.relabel_nodes(g, mapping)
        # scale size of the graphs such that it fits into 0,1 square
        for node in g.nodes:
            g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * scale_factor
        for edge in g.edges:
            g.edges[edge]['weight'] = g.edges[edge]['weight'] * scale_factor
        graph_list.append(g)
    print("Number of loaded test graphs:",len(graph_list))
    return graph_list, fnames

def find_best_model(dqn):
    num_min = dqn.cfg['NUM_MIN']
    num_max = dqn.cfg['NUM_MAX']
    g_type = dqn.cfg['g_type']
    base_path = f'best_models/{g_type}/'
    best_tour_length = np.inf
    # file_endings = ['ckpt', 'csv', 'pyx']
    models = os.scandir(base_path)
    for model in models:
        # print(model.name)
        # res = [ele for ele in file_endings if(ele in model)]
        # check whether we have file or folder, continue in case of file
        if model.is_file():
            print(model.name)
            continue
        new_base_path = base_path + model.name + '/'
        for f in os.listdir(new_base_path):
            # print(f)
            nrange_str = 'nrange_{}_{}'.format(num_min, num_max)
            if ('ckpt' not in f):# or (nrange_str not in f):
                continue
            # print(f)
            f_len = f.split('_')[-1].split('.')[0]
            tour_length = float(f_len)/(10**(len(f_len)-1))
            if f_len[0] != '1':
                continue
                # norm_tour_length = tour_length/float(config['valid_sol'])
            else:
                norm_tour_length = tour_length
            if norm_tour_length < best_tour_length:
                best_model_file = '.'.join(f.split('.')[0:-1])
                best_model_base_path = new_base_path
                best_tour_length = tour_length
    print("Best model file:", best_model_file)
    return best_model_file, best_model_base_path, best_tour_length

def run_test(dqn, graph_list):
    lengths = []
    solutions = []
    sol_times = []
    for g in tqdm(graph_list):
        len, sol, time = dqn.Evaluate(g)
        lengths.append(len)
        solutions.append(sol)
        sol_times.append(time)
    return lengths, solutions, sol_times

def get_approx_ratios(data_dir, fnames, test_lengths):
    true_lengths = []
    len_dict = get_len_dict(data_dir)
    for fname in fnames:
        true_lengths.append(len_dict[fname])
    approx_ratios = [length[0]/length[1] for length in zip(test_lengths, true_lengths)]
    mean_approx_ratio = np.mean([length[0]/length[1] for length in zip(test_lengths, true_lengths)])
    return approx_ratios, mean_approx_ratio

def get_len_dict(folder):
    # get lengths
    with open(folder+'lengths.txt', 'r') as f:
        lines = f.readlines()
        file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
        test_lens = [float(line.split(':')[-1].strip()) for k, line in enumerate(lines)]
        len_dict = dict(zip(file_names, test_lens))
    return len_dict

def save_solutions(data_dir, fnames, solutions, best_model_file):
    sol_df = pd.DataFrame()
    idx = 0
    print("Saving solutions...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = solutions[idx]
        sol_df = pd.concat([sol_df,tmp_df.astype(int)], ignore_index=False, axis=1)
        idx += 1
    best_model = best_model_file.split('.')[0]
    result_folder = data_dir.split("/")[-2]
    try:
        os.mkdir(f'results/{result_folder}')
    except:
        pass
    sol_df.to_csv('results/{}/solution_{}.csv'.format(result_folder, best_model))

def save_lengths(data_dir, fnames, lengths, best_model_file):
    lens_df = pd.DataFrame()
    idx = 0
    print("Saving solution lengths...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = [lengths[idx]]
        lens_df = pd.concat([lens_df,tmp_df], ignore_index=False, axis=1)
        idx += 1
    best_model = best_model_file.split('.')[0]
    result_folder = data_dir.split("/")[-2]
    try:
        os.mkdir(f'results/{result_folder}')
    except:
        pass
    lens_df.to_csv('results/{}/tour_lengths_{}.csv'.format(result_folder, best_model))

def save_approx_ratios(data_dir, fnames, approx_ratios, best_model_file):
    lens_df = pd.DataFrame()
    idx = 0
    print("Saving approximation ratios...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = [approx_ratios[idx]]
        lens_df = pd.concat([lens_df,tmp_df], ignore_index=False, axis=1)
        idx += 1
    best_model = best_model_file.split('.')[0]
    result_folder = data_dir.split("/")[-2]
    try:
        os.mkdir(f'results/{result_folder}')
    except:
        pass
    lens_df.to_csv('results/{}/approx_ratios_{}.csv'.format(result_folder, best_model))

def prepare_testset_S2VDQN(folder, scale_factor=0.000001):
    if folder[-1] == '/':
        folder = folder[0:-1]
    graph_list = []
    fnames = []
    print("Loading test graphs...")
    with open(f'{folder}/paths.txt', 'r') as f:
        for line in tqdm(f):
            fname = line.split('/')[-1].strip()
            file_path = '%s/%s' % (folder, fname)
            try:
                if not '.tsp' in fname or '.sol' in fname:
                    continue
                problem = tsplib95.load(file_path)
                g = problem.get_graph()
                
            except:
                print('Error, while loading file {}'.format(fname))
        
            # remove edges from one node to itself
            ebunch=[(k,k) for k in g.nodes]
            g.remove_edges_from(ebunch)
            # reset node index to start at zero
            mapping = {k:i for i,k in enumerate(g.nodes)}
            g = nx.relabel_nodes(g, mapping)
            # scale size of the graphs such that it fits into 0,1 square
            for node in g.nodes:
                g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * scale_factor
            for edge in g.edges:
                g.edges[edge]['weight'] = g.edges[edge]['weight'] * scale_factor
            graph_list.append(g)
            fnames.append(fname)
    # print("Number of loaded test graphs:",len(graph_list))
    return graph_list, fnames

def get_data_from_result_files(data_dir, result_dir):
    for f in os.listdir(result_dir):
        if 'solution' in f:
            sol_df = pd.read_csv(result_dir + f, index_col=0)
        elif 'tour' in f:
            len_df = pd.read_csv(result_dir + f, index_col=0)
        elif 'approx' in f:
            approx_df = pd.read_csv(result_dir + f, index_col=0)
    try:
        approx_ratios = list(approx_df.iloc[0])
        fnames = [fname+'.tsp' for fname in len_df.columns if not '.tsp' in fname]
        test_lengths = list(len_df.iloc[0])
        
    except:
        print("Generating approx ratios")
        fnames = [fname+'.tsp' for fname in len_df.columns if not '.tsp' in fname]
        test_lengths = list(len_df.iloc[0])
        approx_ratios, mean_approx_ratio = get_approx_ratios(data_dir, fnames=fnames, test_lengths=test_lengths)
    solutions = []
    for column in sol_df.columns:
            raw_list = list(sol_df[column])
            processed_list = [int(k) for k in raw_list if not np.isnan(k)]
            solutions.append(processed_list)
    return fnames, approx_ratios, test_lengths, solutions

def get_model_file(model_path):

    for f in os.listdir(model_path):
        if ('ckpt' not in f):# or (nrange_str not in f):
            continue
        # print(f)
        f_len = f.split('_')[-1].split('.')[0]
        tour_length = float(f_len)/(10**(len(f_len)-1))
        if f_len[0] != '1':
            continue
            # norm_tour_length = tour_length/float(config['valid_sol'])
        else:
            model_file = '.'.join(f.split('.')[0:-1])
            model_base_path = model_path
    print("Best model file:", model_file)
    return model_file, model_base_path, tour_length