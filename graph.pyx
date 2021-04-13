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
# import gc

cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph#使用unique_ptr优于shared_ptr
    #__cinit__会在__init__之前被调用
    def __cinit__(self,*arg):
        '''doing something before python calls the __init__.
        C/C++ objects of cdef must be initialized inside __cinit__, otherwise no memory is allocated for them
        Can take arguments and use python's variable argument model to implement function overloading-like functionality.。'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        self.inner_graph = shared_ptr[Graph](new Graph())
        cdef int _num_nodes
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        cdef double[:] edge_weights
        if len(arg)==0:
            #这两行代码为了防止内存没有初始化，没有实际意义
            deref(self.inner_graph).num_edges=0
            deref(self.inner_graph).num_nodes=0
        # elif len(arg)==4:
        #     _num_nodes=arg[0]
        #     _num_edges=arg[1]
        #     edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
        #     edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
        #     self.reshape_Graph(_num_nodes,  _num_edges,  edges_from,  edges_to)
        elif len(arg) == 5:
            _num_nodes=arg[0]
            _num_edges=arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            edge_weights = np.array([x for x in arg[4]], dtype=np.double)
            self.reshape_Graph(_num_nodes,  _num_edges,  edges_from,  edges_to, edge_weights)   
        else:
            print('Error: py_Graph class was not initialized successfully because the number of parameters provided did not match, and the number of parameters was not 0 or 5.')
    # def __dealloc__(self):
    #     if self.inner_graph != NULL:
    #         self.inner_graph.reset()
    #         gc.collect()
    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    # @num_nodes.setter
    # def num_nodes(self):
    #     def __set__(self,num_nodes):
    #         self.setadj(adj_list)

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list
    # Tsp change
    @property
    def edge_weights(self):
        return deref(self.inner_graph).edge_weights

    def getEdgeWeight(self, int start_node, int end_node):
        return deref(self.inner_graph).getEdgeWeight(start_node, end_node)

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:] edge_weights):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        # tsp change
        cdef double *cdouble_edge_weights = <double*>malloc(_num_edges*sizeof(double))
        cdef int i
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        # tsp change
        for i in range(_num_edges):
            cdouble_edge_weights[i] = edge_weights[i]
        # graph constructor changed
        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes, _num_edges, &cint_edges_from[0], &cint_edges_to[0], &cdouble_edge_weights[0]))
        
        free(cint_edges_from)
        free(cint_edges_to)
        free(cdouble_edge_weights)

    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:] edge_weights):
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, edge_weights)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())
    # def __dealloc__(self):
    #     if self.inner_gset != NULL:
    #         self.inner_gset.reset()
    #         gc.collect()
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid, graph.inner_graph)
        #self.InsertGraph(gid,graph.inner_graph)

        # deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
         #self.Inner_InsertGraph(gid,graph.inner_graph)

    def Sample(self):
        temp_innerGraph=deref(deref(self.inner_gset).Sample())   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self, Graph graph):
        num_nodes = graph.num_nodes     #得到Graph对象的节点个数
        num_edges = graph.num_edges    #得到Graph对象的连边个数
        edge_list = graph.edge_list
        edge_weights = graph.edge_weights
        
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        cdouble_edge_weights = np.zeros([num_edges], dtype=np.double)
        
        cdef int i
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
            cdouble_edge_weights[i] = edge_weights[i]
        return py_Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to, cdouble_edge_weights)


