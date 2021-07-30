import os
import networkx as nx
import tsplib95
import re
import numpy as np

class TSP_loader:
    def __init__(self) -> None:
        pass
    
    def load_multi_tsp_as_nx(self, data_dir, scale_factor=0.0001):
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split('(\d+)', text)]
        fnames = os.listdir(data_dir)
        fnames.sort(key=natural_keys)
        graph_list = []
        for fname in fnames:
            if not 'tsp' in fname:
                continue
            g = self.load_tsp_as_nx(data_dir+fname, scale_factor=scale_factor)
            graph_list.append(g)
        return graph_list

    def load_tsp_as_nx(self, file_path, scale_factor=0.0001):
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