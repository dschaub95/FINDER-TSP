from itertools import permutations
import numpy as np
from py_utils.TSP_loader import TSP_loader

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