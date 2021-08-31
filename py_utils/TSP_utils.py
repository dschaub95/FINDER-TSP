import numpy as np
import networkx as nx
import re
import os
import tsplib95
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib as mpl 
import time
from itertools import permutations
import sys
from py_utils.TSP_loader import TSP_loader

class TSP_plotter:
    def __init__(self) -> None:
        pass
    def plot_nx_graph(self, graph, draw_edges=True, tour_length=None, solution=None, title=''):
        plt.style.use('seaborn-paper')
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
        num_nodes = graph.number_of_nodes()
        labels = dict()
        if solution:
            labels = {solution[i]:i for i in graph.nodes}
            tour_edges = list(zip(solution, solution[1:]))
            tour_edges.append((solution[-1], solution[0]))
        else:
            labels = {i:i for i in graph.nodes}
        pos = {i:graph.nodes[i]['coord'] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='y', node_size=200)
        if draw_edges:
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='y', width=1, alpha=0.2)
        if solution:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=tour_edges, edge_color='r', width=2)
        # Draw labels
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=9)
        # ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.set_xlabel('x-coordinate')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_ylabel('y-coordinate')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig('plots/tour_plot.png', dpi=400)
        plt.show()

class TSP_solver:
    def __init__(self) -> None:
        self.loader = TSP_loader()
    
    def brute_solve_tsp(self, graph):
        perms = permutations(list(graph.nodes))
        opt_tour_length = np.inf
        opt_tour = []
        for perm in perms:
            tmp_len = self.calc_tour_length(graph, perm)
            if tmp_len < opt_tour_length:
                opt_tour_length = tmp_len
                opt_tour = list(perm)
            else:
                continue
        return opt_tour_length, opt_tour
    
    def calc_tour_length(self, graph, solution):
        tot_len = 0
        for i in range(np.array(solution).shape[0]):
            if i == np.array(solution).shape[0] - 1:
                tot_len += graph[solution[i]][solution[0]]['weight']
            else:
                tot_len += graph[solution[i]][solution[i + 1]]['weight']
        return tot_len
class TSP_generator:
    def __init__(self, g_type, num_min, num_max) -> None:
        self.g_type = g_type
        self.num_min = num_min
        self.num_max = num_max
        self.solver = TSP_solver()
    
    def save_nx_as_tsp(self, graph_list, save_path, scale=6, start_index=0, init_pos=None, goal_pos=None):
        # make sure everything is saved in the save dir
        if save_path[-1] != '/':
            save_path = save_path + '/'
        # create save dir if needed
        if not os.path.isdir(save_path):
            try: 
                os.mkdir(save_path) 
            except OSError as error: 
                print(error) 
        for k, graph in enumerate(graph_list):
            problem = tsplib95.models.StandardProblem()
            problem.name = 'TSP_Problem_{}'.format(start_index + k)
            problem.type = 'TSP'
            problem.dimension = graph.number_of_nodes()
            problem.edge_weight_type = 'EUC_2D'
            # problem.node_coord_type = 'TWOD_COORDS'
            if init_pos == '0,1' and goal_pos == '-0.5,0.5':
                problem.node_coords = {'{}'.format(node[0]): list(np.round((10**scale)*(node[1]['coord']-0.5), 0)) for node in graph.nodes.items()}
            else:
                problem.node_coords = {'{}'.format(node[0]): list(np.round((10**scale)*node[1]['coord'], 0)) for node in graph.nodes.items()}
            # save_path = 'valid_sets/synthetic_nrange_10_20_200/TSP_Problem_{}.tsp'.format(k)
            file_path = save_path + 'TSP_Problem_{}.tsp'.format(start_index + k)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            problem.save(file_path)
            with open(file_path, 'a') as f:
                f.write('\n')
    
    def gen_graphs(self, num_graphs=1000, **args):
        graph_list = []
        for i in range(0, num_graphs):
            graph_list.append(self.gen_graph(**args))
        return graph_list

    def gen_graph(self, pos='0,1'):
        """
        Generates new graphs of different g_type--> used for training or testing
        """
        max_n = self.num_max
        min_n = self.num_min
        g_type = self.g_type
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if g_type == 'tsp_2d':
            # slow code, might need optimization
            if pos == '0,1':
                node_postions = np.random.rand(cur_n, 2)
            elif pos == '-0.5,0.5':
                node_postions = np.random.rand(cur_n, 2) - 0.5
            else:
                print("Unknown position input!")
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(node_postions),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            feature_dict = {k: {'coord': node_postions[k]} for k in range(cur_n)} 
            nx.set_node_attributes(g, feature_dict)
        elif g_type == 'tsp':
            # slow code, might need optimization
            node_postions = np.random.rand(cur_n, 2)
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(node_postions),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
        return g