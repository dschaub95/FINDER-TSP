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
    
    # def __cinit__(self, double _norm, int _help_func, int _sign, int _fix_start_node):
    #     self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm, _help_func, _sign, _fix_start_node))
    #     self.inner_Graph = shared_ptr[Graph](new Graph())
    
    def __cinit__(self, *args):
        cdef int _norm
        cdef int _help_func
        cdef int _sign
        cdef int _fix_start_node

        self.inner_Graph = shared_ptr[Graph](new Graph())
        if len(args) == 1:
            _norm = args[0].norm
            _help_func = args[0].help_func
            _sign =  args[0].sign
            _fix_start_node = args[0].fix_start_node
            self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm, _help_func, _sign, _fix_start_node))
            
            self.insert_graph(args[0].graph)
            deref(self.inner_MvcEnv).graph = self.inner_Graph
            
            deref(self.inner_MvcEnv).numCoveredEdges = args[0].numCoveredEdges
            deref(self.inner_MvcEnv).state_seq = args[0].state_seq
            deref(self.inner_MvcEnv).act_seq = args[0].act_seq
            deref(self.inner_MvcEnv).state = args[0].state
            deref(self.inner_MvcEnv).reward_seq = args[0].reward_seq
            deref(self.inner_MvcEnv).sum_rewards = args[0].sum_rewards
            deref(self.inner_MvcEnv).covered_set = args[0].covered_set
            deref(self.inner_MvcEnv).avail_list = args[0].avail_list
            # print("orig act seq:", args[0].act_seq)
            # print("copied act seq:", deref(self.inner_MvcEnv).act_seq)
        else:
            _norm = args[0]
            _help_func = args[1]
            _sign =  args[2]
            _fix_start_node = args[3]
            self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm, _help_func, _sign, _fix_start_node))
    
    def s0(self, _g):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).NN_ratio = _g.NN_ratio
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).node_feats = _g.node_feats
        deref(self.inner_Graph).EdgeWeight = _g.EdgeWeight
        deref(self.inner_Graph).edge_probs = _g.edge_probs
        deref(self.inner_MvcEnv).s0(self.inner_Graph)

    def insert_graph(self, _g):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).NN_ratio = _g.NN_ratio
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).node_feats = _g.node_feats
        deref(self.inner_Graph).EdgeWeight = _g.EdgeWeight
        deref(self.inner_Graph).edge_probs = _g.edge_probs
        deref(self.inner_MvcEnv).s0(self.inner_Graph)

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
    def help_func(self):
        return deref(self.inner_MvcEnv).help_func

    @property
    def sign(self):
        return deref(self.inner_MvcEnv).sign
    
    @property
    def fix_start_node(self):
        return deref(self.inner_MvcEnv).fix_start_node

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
    def state(self):
        return deref(self.inner_MvcEnv).state

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
        NN_ratio = graph1.NN_ratio
        edge_list = graph1.edge_list
        node_feats = graph1.node_feats
        EdgeWeight = graph1.EdgeWeight
        edge_probs = graph1.edge_probs

        cint_edges_from = np.zeros([num_edges],dtype=np.int32)
        cint_edges_to = np.zeros([num_edges],dtype=np.int32)
        cdouble_vec_node_feats = np.zeros([num_nodes, 2], dtype=np.double)
        cdouble_EdgeWeight = np.zeros([num_nodes, num_nodes], dtype=np.double)
        cdouble_edge_probs = np.zeros([num_nodes, num_nodes], dtype=np.double)
        # print(num_nodes)
        cdef int i
        for i in range(num_edges):
            cint_edges_from[i] = edge_list[i].first
            cint_edges_to[i] = edge_list[i].second
        cdef int j
        cdef int k
        for j in range(num_nodes):
             cdouble_vec_node_feats[j,:] = node_feats[j]
             cdouble_EdgeWeight[j,:] = EdgeWeight[j]
             cdouble_edge_probs[j,:] = edge_probs[j]
        # print("test:", cdouble_vec_node_feats)
        return graph.py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_EdgeWeight, cdouble_edge_probs,
                              cdouble_vec_node_feats, NN_ratio)

def copy_test_environment(test_env):
    # needs to be updated by using a specific constructor which allows to set all properties directly (apart from graph)
    cdef int NUM_MAX = test_env.norm
    cdef int help_func = test_env.help_func
    cdef int reward_sign = test_env.sign
    cdef int fix_start_node = test_env.fix_start_node
    
    copied_test_env = py_MvcEnv(test_env)
    # copied_test_env = py_MvcEnv(NUM_MAX, help_func, reward_sign, fix_start_node)
    # copied_test_env.s0(test_env.graph)
    # for action in test_env.act_seq:
    #     copied_test_env.stepWithoutReward(action)
    
    assert(test_env.act_seq == copied_test_env.act_seq)
    # assert(test_env.graph == copied_test_env.graph)
    assert(test_env.state == copied_test_env.state)
    return copied_test_env