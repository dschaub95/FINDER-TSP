﻿from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_MvcEnv:
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self, double _norm):
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm))
        self.inner_Graph =shared_ptr[Graph](new Graph())
    # def __dealloc__(self):
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    def s0(self, _g, int _help_func):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_weights = _g.edge_weights
        deref(self.inner_Graph).node_feats = _g.node_feats
        deref(self.inner_Graph).EdgeWeight = _g.EdgeWeight
        deref(self.inner_MvcEnv).s0(self.inner_Graph, _help_func)

    def step(self, int a):
        return deref(self.inner_MvcEnv).step(a)

    def stepWithoutReward(self, int a):
        deref(self.inner_MvcEnv).stepWithoutReward(a)

    def randomAction(self):
        return deref(self.inner_MvcEnv).randomAction()

    # def betweenAction(self):
    #     return deref(self.inner_MvcEnv).betweenAction()

    def isTerminal(self):
        return deref(self.inner_MvcEnv).isTerminal()

    def add_node(self, int new_node):
        return deref(self.inner_MvcEnv).add_node(new_node)

    def getReward(self):
        return deref(self.inner_MvcEnv).getReward()

    def getLastTourDifference(self):
        return deref(self.inner_MvcEnv).getLastTourDifference()

    def getTourDifference(self, int new_node):
        return deref(self.inner_MvcEnv).getTourDifference(new_node)

    @property
    def norm(self):
        return deref(self.inner_MvcEnv).norm

    @property
    def graph(self):
        # temp_innerGraph=deref(self.inner_Graph)   #The Graph object is obtained
        return self.G2P(deref(self.inner_Graph))

    @property
    def state_seq(self):
        return deref(self.inner_MvcEnv).state_seq

    @property
    def act_seq(self):
        return deref(self.inner_MvcEnv).act_seq

    @property
    def action_list(self):
        return deref(self.inner_MvcEnv).action_list

    @property
    def reward_seq(self):
        return deref(self.inner_MvcEnv).reward_seq

    @property
    def sum_rewards(self):
        return deref(self.inner_MvcEnv).sum_rewards

    @property
    def numCoveredEdges(self):
        return deref(self.inner_MvcEnv).numCoveredEdges

    @property
    def covered_set(self):
        return deref(self.inner_MvcEnv).covered_set

    @property
    def avail_list(self):
        return deref(self.inner_MvcEnv).avail_list

    # TSP changes!
    cdef G2P(self, Graph graph1):
        num_nodes = graph1.num_nodes     #得到Graph对象的节点个数
        num_edges = graph1.num_edges    #得到Graph对象的连边个数
        edge_list = graph1.edge_list
        edge_weights = graph1.edge_weights
        node_feats = graph1.node_feats

        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        cdouble_edge_weights = np.zeros([num_edges], dtype=np.double)
        cdouble_vec_node_feats = np.zeros([num_nodes, 2], dtype=np.double)
        # print(num_nodes)
        cdef int i
        for i in range(num_edges):
            cint_edges_from[i] = edge_list[i].first
            cint_edges_to[i] = edge_list[i].second
            cdouble_edge_weights[i] = edge_weights[i]
        cdef int j
        cdef int k
        for j in range(num_nodes):
             cdouble_vec_node_feats[j,:] = node_feats[j]
        # print("test:", cdouble_vec_node_feats)
        return graph.py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_edge_weights, cdouble_vec_node_feats)


    # cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
    #     cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int i
    #     for i in range(_num_edges):
    #         cint_edges_from[i] = edges_from[i]
    #     for i in range(_num_edges):
    #         cint_edges_to[i] = edges_to[i]
    #     free(cint_edges_from)
    #     free(cint_edges_to)
    #     return  new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0])