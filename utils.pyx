from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free
from distutils.util import strtobool

cdef class py_Utils:
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

def read_config(config_path):
    with open(config_path) as f:
        data = f.read()
    data = data.replace(" ", "").split('\n')
    data = [element for element in data if not len(element) == 0]
    # delete comment lines
    data = [element for element in data if not element[0] == '#']
    # ignore inline comments
    data = [element.split('#')[0] for element in data]
    # delete string literals
    data = [element.replace("'", "") for element in data]
    data_dict = dict(substr.split('=') for substr in data)
    # conversion to correct data type
    for key in data_dict:
        if '.' in data_dict[key]:
            # conversion to float
            try:
                data_dict[key] = float(data_dict[key])
            except:
                pass
        else:
            # conversion to int or boolean int (0,1)
            try:
                data_dict[key] = int(data_dict[key])
            except:
                try:
                    data_dict[key] = strtobool(data_dict[key])
                except:
                    pass
    return data_dict