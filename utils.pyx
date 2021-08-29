from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free
from distutils.util import strtobool
import networkx as nx
import numpy as np
from itertools import combinations

cdef class pyx_Utils:
    cdef shared_ptr[Utils] inner_Utils
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())

    def getTourLength(self, _g, solution):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).NN_ratio = _g.NN_ratio
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).node_feats = _g.node_feats
        deref(self.inner_Graph).EdgeWeight = _g.EdgeWeight
        return deref(self.inner_Utils).getTourLength(self.inner_Graph, solution)

def gen_graph(num_min, num_max, g_type):
    """
    Generates new graphs of different g_type--> used for training or testing
    """
    cdef int max_n = num_max
    cdef int min_n = num_min
    cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=4)
    elif g_type == 'tsp_2d':
        # slow code, might need optimization
        nodes = np.random.rand(cur_n,2)
        edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        feature_dict = {k: {'coord': nodes[k]} for k in range(cur_n)} 
        nx.set_node_attributes(g, feature_dict)
    elif g_type == 'tsp':
        # slow code, might need optimization
        nodes = np.random.rand(cur_n, 2)
        edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(nodes),2)]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
    return g
