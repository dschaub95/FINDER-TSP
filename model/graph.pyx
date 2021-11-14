'''
#file:graph.pyx类graph的实现文件
#可以自动导入相同路径下相同名称的.pxd的文件
#可以省略cimport graph命令
#需要重新设计python调用的接口，此文件
'''
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np


cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph

    def __cinit__(self, *args):
        '''doing something before python calls the __init__.
        C/C++ objects of cdef must be initialized inside __cinit__, otherwise no memory is allocated for them
        Can take arguments and use python's variable argument model to implement function overloading-like functionality.。'''

        self.inner_graph = shared_ptr[Graph](new Graph())
        
        cdef int _num_nodes
        cdef int _num_edges
        cdef double _NN_ratio
        
        cdef int[:] edges_from
        cdef int[:] edges_to
        cdef double[:,:] node_feats
        cdef double[:,:] EdgeWeight
        cdef double[:,:] edge_probs
        
        if len(args) == 0:
            deref(self.inner_graph).num_edges = 0
            deref(self.inner_graph).num_nodes = 0
        elif len(args) == 8:
            _num_nodes = args[0]
            _num_edges = args[1]
            edges_from = np.array([int(x) for x in args[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in args[3]], dtype=np.int32)
            EdgeWeight = np.array([x for x in args[4]], dtype=np.double)
            edge_probs = np.array([x for x in args[5]], dtype=np.double)
            node_feats = np.array([x for x in args[6]], dtype=np.double)
            _NN_ratio = args[7]
            # print("True weights\n", args[4])
            self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, EdgeWeight, edge_probs, node_feats, _NN_ratio)   
        else:
            print('Error: py_Graph class was not initialized successfully because the number of parameters provided did not match, and the number of parameters was not 0 or 6.')

    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def NN_ratio(self):
        return deref(self.inner_graph).NN_ratio

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list

    @property
    def node_feats(self):
        return deref(self.inner_graph).node_feats
    
    @property
    def EdgeWeight(self):
        return deref(self.inner_graph).EdgeWeight

    @property
    def edge_probs(self):
        return deref(self.inner_graph).edge_probs

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, 
                       double[:,:] EdgeWeight, double[:,:] edge_probs, double[:,:] node_feats, double _NN_ratio):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        # tsp change
        cdef double **cdouble_vec_node_feats = <double**>malloc(_num_nodes*sizeof(double*))
        cdef double **cdouble_EdgeWeight = <double**>malloc(_num_nodes*sizeof(double*))
        cdef double **cdouble_edge_probs = <double**>malloc(_num_nodes*sizeof(double*))
        cdef int i
        # print("node feats 0", node_feats[0][0], node_feats[0][1])
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        # tsp change
        for i in range(_num_nodes):
            cdouble_vec_node_feats[i] = &node_feats[i, 0]
            cdouble_EdgeWeight[i] = &EdgeWeight[i, 0]
            cdouble_edge_probs[i] = &edge_probs[i, 0]
        # print("sucessfully saved features")
        # graph constructor changed
        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes, _num_edges, &cint_edges_from[0], &cint_edges_to[0], 
                                                       &cdouble_EdgeWeight[0], &cdouble_edge_probs[0],
                                                       &cdouble_vec_node_feats[0], _NN_ratio))
        
        free(cint_edges_from)
        free(cint_edges_to)
        free(cdouble_vec_node_feats)
        free(cdouble_EdgeWeight)
        free(cdouble_edge_probs)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())

    def InsertGraph(self, int gid, py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid, graph.inner_graph)

    def Sample(self):
        temp_innerGraph = deref(deref(self.inner_gset).Sample())   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Get(self,int gid):
        temp_innerGraph = deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self, Graph graph):
        num_nodes = graph.num_nodes     #得到Graph对象的节点个数
        num_edges = graph.num_edges    #得到Graph对象的连边个数
        NN_ratio = graph.NN_ratio
        edge_list = graph.edge_list
        node_feats = graph.node_feats
        EdgeWeight = graph.EdgeWeight
        edge_probs = graph.edge_probs

        cint_edges_from = np.zeros([num_edges], dtype=np.int32)
        cint_edges_to = np.zeros([num_edges], dtype=np.int32)
        cdouble_vec_node_feats = np.zeros([num_nodes, 2], dtype=np.double)
        cdouble_EdgeWeight = np.zeros([num_nodes, num_nodes], dtype=np.double)
        cdouble_edge_probs = np.zeros([num_nodes, num_nodes], dtype=np.double)

        cdef int i
        for i in range(num_edges):
            cint_edges_from[i]= edge_list[i].first
            cint_edges_to[i] = edge_list[i].second
        cdef int j
        for j in range(num_nodes):
            cdouble_vec_node_feats[j,:] = node_feats[j]
            cdouble_EdgeWeight[j,:] = EdgeWeight[j]
            cdouble_edge_probs[j,:] = edge_probs[j]
        return py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_EdgeWeight, cdouble_edge_probs,
                        cdouble_vec_node_feats, NN_ratio)


