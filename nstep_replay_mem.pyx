from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp cimport bool
import  graph
import  numpy as np
# import gc


cdef class py_ReplaySample:
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    def __cinit__(self,int batch_size):
        self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample(batch_size))
    # def __dealloc__(self):
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()
    @property
    def g_list(self):
        result = []
        for graphPtr in deref(self.inner_ReplaySample).g_list:
            result.append(self.G2P(deref(graphPtr)))
            # print("sucessfully appended graph")
        return  result
    @property
    def list_st(self):
        return deref(self.inner_ReplaySample).list_st
    @property
    def list_s_primes(self):
        return deref(self.inner_ReplaySample).list_s_primes
    @property
    def list_at(self):
        return deref(self.inner_ReplaySample).list_at
    @property
    def list_rt(self):
        return deref(self.inner_ReplaySample).list_rt
    @property
    def list_term(self):
        return deref(self.inner_ReplaySample).list_term #boolean whether terminal state is reached for that sample (after n or less steps)

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
        
        cdef int i
        # print("saving data to cvectors..")
        for i in range(num_edges):
            cint_edges_from[i]= edge_list[i].first
            cint_edges_to[i] = edge_list[i].second
            #print("saving", i, "th edge weight")
            cdouble_edge_weights[i] = edge_weights[i]
        cdef int j
        for j in range(num_nodes):
            cdouble_vec_node_feats[j,:] = node_feats[j]
        # print("saved data to cvectors")
        return graph.py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_edge_weights, cdouble_vec_node_feats)

cdef class py_NStepReplayMem:
    cdef shared_ptr[NStepReplayMem] inner_NStepReplayMem
    cdef shared_ptr[Graph] inner_Graph
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    #__cinit__会在__init__之前被调用
    def __cinit__(self,int memory_size):
        '''default constructor, without calling Graph's default constructor for now.
        The memory allocated on the stack by the default constructor is faster to read and write.
        But in practice, once the structure of the network changes, you have to recreate the object on the heap, so basically the memory allocated on the stack will not be used
        Unless the class implementation file is rewritten to include the python call interface, there is no way to avoid creating objects on the heap.'''
        #print('默认构造函数。')
        self.inner_NStepReplayMem = shared_ptr[NStepReplayMem](new NStepReplayMem(memory_size))
    # def __dealloc__(self):
    #     if self.inner_NStepReplayMem != NULL:
    #         self.inner_NStepReplayMem.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()


    def Add(self, mvcenv, int nstep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        g = mvcenv.graph
        deref(self.inner_Graph).num_nodes= g.num_nodes
        deref(self.inner_Graph).num_edges=g.num_edges
        deref(self.inner_Graph).edge_list = g.edge_list
        deref(self.inner_Graph).adj_list = g.adj_list
        deref(self.inner_Graph).edge_weights = g.edge_weights
        deref(self.inner_Graph).node_feats = g.node_feats
        deref(self.inner_Graph).EdgeWeight = g.EdgeWeight
        
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(mvcenv.norm, mvcenv.help_func, mvcenv.sign))
        deref(self.inner_MvcEnv).norm = mvcenv.norm
        deref(self.inner_MvcEnv).help_func = mvcenv.help_func
        deref(self.inner_MvcEnv).sign = mvcenv.sign
        deref(self.inner_MvcEnv).graph = self.inner_Graph
        deref(self.inner_MvcEnv).state_seq = mvcenv.state_seq
        deref(self.inner_MvcEnv).act_seq = mvcenv.act_seq
        deref(self.inner_MvcEnv).action_list = mvcenv.action_list
        deref(self.inner_MvcEnv).reward_seq = mvcenv.reward_seq
        deref(self.inner_MvcEnv).sum_rewards = mvcenv.sum_rewards
        deref(self.inner_MvcEnv).numCoveredEdges = mvcenv.numCoveredEdges
        deref(self.inner_MvcEnv).covered_set = mvcenv.covered_set
        deref(self.inner_MvcEnv).avail_list = mvcenv.avail_list
        deref(self.inner_NStepReplayMem).Add(self.inner_MvcEnv,nstep)

    def Sampling(self,int batch_size):
        # self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample(batch_size))
        self.inner_ReplaySample =  deref(self.inner_NStepReplayMem).Sampling(batch_size)
        # print("Step 1 of sampling from replay memory sucessful")
        result = py_ReplaySample(batch_size)
        result.inner_ReplaySample = self.inner_ReplaySample
        return result

    @property
    def graphs(self):
        result = []
        for graphPtr in deref(self.inner_NStepReplayMem).graphs:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def actions(self):
        return deref(self.inner_NStepReplayMem).actions
    @property
    def rewards(self):
        return deref(self.inner_NStepReplayMem).rewards
    @property
    def states(self):
        return deref(self.inner_NStepReplayMem).states
    @property
    def s_primes(self):
        return deref(self.inner_NStepReplayMem).s_primes
    @property
    def terminals(self):
        return deref(self.inner_NStepReplayMem).terminals
    @property
    def current(self):
        return deref(self.inner_NStepReplayMem).current
    @property
    def count(self):
        return deref(self.inner_NStepReplayMem).count
    @property
    def memory_size(self):
        return deref(self.inner_NStepReplayMem).memory_size
    
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

        cdef int i
        for i in range(num_edges):
            cint_edges_from[i] = edge_list[i].first
            cint_edges_to[i] = edge_list[i].second
            cdouble_edge_weights[i] = edge_weights[i]
        cdef int j
        for j in range(num_nodes):
            cdouble_vec_node_feats[j,:] = node_feats[j]
        return graph.py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_edge_weights, cdouble_vec_node_feats)
