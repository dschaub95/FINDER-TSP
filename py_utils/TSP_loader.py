import os
import networkx as nx
import tsplib95
import re
from tqdm import tqdm
import numpy as np

class TSP_loader:
    def __init__(self) -> None:
        pass
    
    def load_multi_tsp_as_nx(self, data_dir, scale_factor=0.000001, start_index=0, end_index=np.inf):
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        graph_list = []
        try:
            fnames = [f for f in os.listdir(data_dir) if os.path.isfile(f'{data_dir}/{f}')]
            fnames.sort(key=natural_keys)
        except:
            print('\nBad TSP directory!')
            return graph_list
        fnames = [fname for fname in fnames if 'tsp' in fname]
        if start_index is None:
            start_index = 0
        if (end_index is None) or (end_index > len(fnames)):
            end_index = len(fnames)
        for fname in tqdm(fnames[start_index:end_index]):
            index = int(fname.split('.')[0].split('_')[-1])
            if index < start_index:
                continue
            if index > end_index:
                continue
            g = self.load_tsp_as_nx(f'{data_dir}/{fname}', scale_factor=scale_factor)
            graph_list.append(g)
        return graph_list

    def load_tsp_as_nx(self, file_path, scale_factor=0.000001):
        try:
            problem = tsplib95.load(file_path)
            g = problem.get_graph()
            # remove edges from nodes to itself
            ebunch=[(k,k) for k in g.nodes()]
            g.remove_edges_from(ebunch)
            for node in g.nodes():
                g.nodes[node]['coord'] = np.array(g.nodes[node]['coord']) * scale_factor
            for edge in g.edges:
                g.edges[edge]['weight'] = g.edges[edge]['weight'] * scale_factor
        except:
            g = nx.Graph()
            print("Error!")
        return g
