﻿from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_Utils:
    cdef shared_ptr[Utils] inner_Utils
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())

    def getTourLength(self,_g,solution):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_weights = _g.edge_weights
        deref(self.inner_Graph).node_feats = _g.node_feats
        deref(self.inner_Graph).EdgeWeight = _g.EdgeWeight
        return deref(self.inner_Utils).getTourLength(self.inner_Graph, solution)
