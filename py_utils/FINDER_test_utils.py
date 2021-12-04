import os
import numpy as np
import pandas as pd
import tsplib95
import networkx as nx
from tqdm import tqdm
import sys
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

def get_approx_ratios(data_dir, test_lengths):
    fnames = get_fnames(data_dir)
    true_lengths = []
    len_dict = get_len_dict(data_dir)
    for fname in fnames:
        true_lengths.append(len_dict[fname])
    approx_ratios = [length[0]/length[1] for length in zip(test_lengths, true_lengths)]
    mean_approx_ratio = np.mean([length[0]/length[1] for length in zip(test_lengths, true_lengths)])
    return approx_ratios, mean_approx_ratio

def get_fnames(dir, search_phrase='tsp'):
    atoi = lambda text : int(text) if text.isdigit() else text
    natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
    try:
        fnames = [f for f in os.listdir(dir) if os.path.isfile(f'{dir}/{f}')]
        fnames.sort(key=natural_keys)
    except:
        print('\nBad directory!')
    fnames = [fname for fname in fnames if search_phrase in fname]
    return fnames

def get_len_dict(folder):
    # get lengths
    with open(f'{folder}/lengths.txt', 'r') as f:
        lines = f.readlines()
        file_names = [line.split(':')[0].strip() for k, line in enumerate(lines)]
        test_lens = [float(line.split(':')[-1].strip()) for k, line in enumerate(lines)]
        len_dict = dict(zip(file_names, test_lens))
    return len_dict

def save_solutions(data_dir, solutions, model_name, suffix=''):
    fnames = get_fnames(data_dir)
    sol_df = pd.DataFrame()
    idx = 0
    tqdm.write("Saving solutions...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = solutions[idx]
        sol_df = pd.concat([sol_df,tmp_df.astype(int)], ignore_index=False, axis=1)
        idx += 1
    test_set_folder = data_dir.split("/")[-2]
    test_set_name = data_dir.split("/")[-1]
    result_path = f'results/{model_name}/{test_set_folder}/{test_set_name}'
    model_name_short = '_'.join(model_name.split('_')[0:-4])
    create_dir(result_path)
    if suffix:
        sol_df.to_csv(f'{result_path}/solutions_{model_name_short}_{suffix}.csv')
    else:
        sol_df.to_csv(f'{result_path}/solutions_{model_name_short}.csv')

def save_lengths(data_dir, lengths, model_name, suffix=''):
    fnames = get_fnames(data_dir)
    lens_df = pd.DataFrame()
    idx = 0
    tqdm.write("Saving solution lengths...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = [lengths[idx]]
        lens_df = pd.concat([lens_df,tmp_df], ignore_index=False, axis=1)
        idx += 1
    test_set_folder = data_dir.split("/")[-2]
    test_set_name = data_dir.split("/")[-1]
    result_path = f'results/{model_name}/{test_set_folder}/{test_set_name}'
    model_name_short = '_'.join(model_name.split('_')[0:-4])
    create_dir(result_path)
    if suffix:
        lens_df.to_csv(f'{result_path}/tour_lengths_{model_name_short}_{suffix}.csv')
    else:
        lens_df.to_csv(f'{result_path}/tour_lengths_{model_name_short}.csv')

def save_approx_ratios(data_dir, approx_ratios, model_name, suffix=''):
    fnames = get_fnames(data_dir)
    approx_df = pd.DataFrame()
    idx = 0
    tqdm.write("Saving approximation ratios...")
    for fname in tqdm(fnames):
        if not '.tsp' in fname or '.sol' in fname:
            continue
        tmp_df = pd.DataFrame()
        tmp_df[fname] = [approx_ratios[idx]]
        approx_df = pd.concat([approx_df,tmp_df], ignore_index=False, axis=1)
        idx += 1
    test_set_folder = data_dir.split("/")[-2]
    test_set_name = data_dir.split("/")[-1]
    result_path = f'results/{model_name}/{test_set_folder}/{test_set_name}'
    model_name_short = '_'.join(model_name.split('_')[0:-4])
    create_dir(result_path)
    if suffix:
        approx_df.to_csv(f'{result_path}/approx_ratios_{model_name_short}_{suffix}.csv')
    else:
        approx_df.to_csv(f'{result_path}/approx_ratios_{model_name_short}.csv')

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def get_test_approx_ratios_for_model(test_set_names, model_name, search_strategy='greedy'):
    mean_approx_ratios = []
    std_approx_ratios = []
    for test_set in test_set_names:
        result_dir = f'../results/{model_name}/test_sets/{test_set}'
        try:
            fnames, approx_ratios, test_lengths, solutions = get_data_from_result_files(result_dir, search_strategy=search_strategy)
        except: 
            # print(search_strategy)
            # print('Using placeholders!')
            approx_ratios = [np.nan]
        mean_approx_ratios.append(np.mean(approx_ratios))
        std_approx_ratios.append(np.std(approx_ratios))
    return mean_approx_ratios, std_approx_ratios

def get_data_from_result_files(result_dir, search_strategy='greedy'):
    # print(result_dir)
    for f in os.listdir(result_dir):
        if not search_strategy in f:
            continue
        if 'solution' in f:
            sol_df = pd.read_csv(f'{result_dir}/{f}', index_col=0)
        elif 'tour' in f:
            len_df = pd.read_csv(f'{result_dir}/{f}', index_col=0)
        elif 'approx' in f:
            approx_df = pd.read_csv(f'{result_dir}/{f}', index_col=0)

    approx_ratios = list(approx_df.iloc[0])
    fnames = [f'{fname}.tsp' if not '.tsp' in fname else fname for fname in len_df.columns ]
    test_lengths = list(len_df.iloc[0])

    solutions = []
    for column in sol_df.columns:
        raw_list = list(sol_df[column])
        processed_list = [int(k) for k in raw_list if not np.isnan(k)]
        solutions.append(processed_list)
    return fnames, approx_ratios, test_lengths, solutions

def get_best_ckpt(model_path, rank=1):
    best_ckpt_path = f'{model_path}/best_checkpoint'
    fnames = get_fnames(best_ckpt_path, search_phrase='ckpt')
    for fname in fnames:
        if 'rank' in fname:
            if f'rank_{rank}.' in fname:
                best_ckpt_file = '.'.join(fname.split('.')[0:-1])
                break
        else:
            best_ckpt_file = '.'.join(fnames[0].split('.')[0:-1])
            break
    best_ckpt_file_path = f'{best_ckpt_path}/{best_ckpt_file}'
    return best_ckpt_file_path

def get_model_file(model_path):
    k = 0
    for f in os.listdir(model_path):
        if not 'ckpt' in f:# or (nrange_str not in f):
            continue
        # print(f)
        f_len = f.split('_')[-1].split('.')[0]
        tour_length = float(f_len)/(10**(len(f_len)-1))
        if f_len[0] != '1':
            continue
            # norm_tour_length = tour_length/float(config['valid_sol'])
        
        else:
            k += 1
            model_file = '.'.join(f.split('.')[0:-1])
            model_base_path = model_path
    if k > 0:
        print("Best model file:", model_file)
    else:
        print("Could not find any checkpoint file in the specified folder!")
    return model_file, model_base_path, tour_length

def prepare_real_samples(data_dir):
    if not data_dir[-1] == '/':
        data_dir = data_dir + '/'
    prepared_graphs = []
    raw_graphs = []
    fnames = []
    for fname in os.listdir(data_dir):
        if not '.tsp' in fname:
            continue
        try:
            problem = tsplib95.load(data_dir + fname)
            g = problem.get_graph()
        except:
            print('Error loading tsp file!')
            continue
        #try:
        # remove edges from nodes to itself
        ebunch=[(k,k) for k in g.nodes()]
        g.remove_edges_from(ebunch)
        # mapping
        mapping = {k:i for i,k in enumerate(g.nodes)}
        g = nx.relabel_nodes(g, mapping)
        # save raw graph
        raw_graphs.append(g.copy())
        # make sure every coordinate is positive
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for node in g.nodes:
            x = g.nodes[node]['coord'][0]
            y = g.nodes[node]['coord'][1]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        if min_x <= 0:
            x_offset = -min_x + 1
        else:
            x_offset = 0
        if min_y <= 0:
            y_offset = -min_y + 1
        else:
            y_offset = 0
        # change node positions into 0,1 square
        for node in g.nodes():
            g.nodes[node]['coord'] = np.array(g.nodes[node]['coord'])
            g.nodes[node]['coord'][0] += x_offset
            g.nodes[node]['coord'][1] += y_offset
            if max_x > max_y:
                scale_factor = 1 / (1.05 * max_x)
            else:
                scale_factor = 1 / (1.05 * max_y)
            g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * scale_factor
        for edge in g.edges:
            g.edges[edge]['weight'] = g.edges[edge]['weight'] * scale_factor
        #except:
        #    g = nx.Graph()
        #    print("Error altering graph!")
        #    continue
        prepared_graphs.append(g)
        
        fnames.append(fname)
    return raw_graphs, prepared_graphs, fnames

