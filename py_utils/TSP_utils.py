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

    def plot_nx_graph(self, graph, draw_edges=True, opt_solution=None, partial_solution=[], title='', edge_probs=None, save_path=None, 
                      node_values=None, only_draw_relevant_edges=False, dpi=300, show=True):
        plt.style.use('seaborn-paper')
        cmap = plt.cm.plasma.reversed()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True, sharey=True, dpi=dpi)
        if node_values is not None:
            nodes_sizes = (np.exp(node_values)/np.sum(np.exp(node_values)) + 0.2) * 500
        else:
            nodes_sizes = 200
        edge_list = [edge for edge in graph.edges]
        if opt_solution is not None:
            opt_tour_edges = list(zip(opt_solution, opt_solution[1:]))
            opt_tour_edges.append((opt_solution[-1], opt_solution[0]))
        if len(partial_solution) > 1:
            partial_tour_edges = list(zip(partial_solution, partial_solution[1:]))

        labels = {i:i for i in graph.nodes}
        pos = {i:graph.nodes[i]['coord'] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='y', node_size=nodes_sizes)
        if len(partial_solution) > 0:
            # mark last selected node
            nx.draw_networkx_nodes(graph, pos, nodelist=[partial_solution[-1]], ax=ax, node_color='r', node_size=200, label="Last selected node")
        if only_draw_relevant_edges and len(partial_solution) > 1:
            # keep edges of partial tour
            remaining_edges_partial = [edge for edge in edge_list if edge in partial_tour_edges or edge[::-1] in partial_tour_edges]
            # keep edges of optimal solution
            remaining_edges_opt = [edge for edge in edge_list if edge in opt_tour_edges or edge[::-1] in opt_tour_edges]
            # delete all edges that are not relevant during the next selection step
            last_node = partial_solution[-1]
            remaining_edges_select = [edge for edge in edge_list if (edge[0] == last_node and edge[1] not in partial_solution) or (edge[1] == last_node and edge[0] not in partial_solution)]
            edge_list = remaining_edges_partial + remaining_edges_opt + remaining_edges_select
            if edge_probs is not None:
                
                probabilties = [edge_probs[edge[0], edge[1]] for edge in edge_list]
                edge_colors = ((np.array(probabilties) + 0.05) * 50)
                edge_alphas = [0.1 if prob == 0.0 else 0.5 for prob in probabilties]
                edges = nx.draw_networkx_edges(graph, pos, edgelist=edge_list, ax=ax, edge_color=edge_colors, width=1, alpha=1.0, edge_cmap=cmap)
            else:
                nx.draw_networkx_edges(graph, pos, edgelist=edge_list, ax=ax, edge_color='y', width=1, alpha=1.0)
        else:
            if edge_probs is not None:
                # edge_list = np.transpose(np.array(np.where(np.triu(edge_probs) > 0)))
                # edge_list = np.transpose(np.asarray(np.triu(edge_probs) > 0).nonzero())
                probabilties = [edge_probs[edge[0], edge[1]] for edge in edge_list]
                edge_colors = ((np.array(probabilties) + 0.05) * 50)
                edge_alphas = [0.1 if prob == 0.0 else 0.5 for prob in probabilties]
                edges = nx.draw_networkx_edges(graph, pos, edgelist=edge_list, ax=ax, edge_color=edge_colors, width=1, alpha=0.2, edge_cmap=cmap)
            else:
                nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='y', width=1, alpha=0.2)
        if opt_solution is not None:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=opt_tour_edges, edge_color='r', label='Optimal solution', width=1, style=(0, (5, 10)))
        if len(partial_solution) > 1:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=partial_tour_edges, edge_color='black', label='Current solution', width=1, style=(0, (1, 10)))
        # Draw labels
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=9)
        # ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

        # set alpha value for each edge
        # for i in range(len(edge_list)):
        #     edges[i].set_alpha(edge_alphas[i])
        
        # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        # pc.set_array(edge_colors)
        # plt.colorbar(pc)

        ax.legend()
        ax.set_xlabel('x-coordinate')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_ylabel('y-coordinate')
        ax.set_title(title)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, format="raw", dpi=dpi)
        if show:
            plt.show()
        return fig
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