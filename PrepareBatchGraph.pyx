from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from graph cimport Graph
import tensorflow as tf
from scipy.sparse import coo_matrix
# import gc


cdef class py_sparseMatrix:
    cdef shared_ptr[sparseMatrix] inner_sparseMatrix
    def __cinit__(self):
        self.inner_sparseMatrix =shared_ptr[sparseMatrix](new sparseMatrix())
    # def __dealloc__(self):
    #     if self.inner_sparseMatrix != NULL:
    #         self.inner_sparseMatrix.reset()
    #         gc.collect()
    @property
    def rowIndex(self):
        return deref(self.inner_sparseMatrix).rowIndex
    @property
    def colIndex(self):
        return deref(self.inner_sparseMatrix).colIndex
    @property
    def value(self):
        return deref(self.inner_sparseMatrix).value
    @property
    def rowNum(self):
        return deref(self.inner_sparseMatrix).rowNum
    @property
    def colNum(self):
        return deref(self.inner_sparseMatrix).colNum


cdef class py_PrepareBatchGraph:
    cdef shared_ptr[PrepareBatchGraph] inner_PrepareBatchGraph
    cdef sparseMatrix matrix
    def __cinit__(self, aggregatorID, node_init_dim, edge_init_dim, ignore_covered_edges, include_selected_nodes, embeddingMethod):
        self.inner_PrepareBatchGraph = shared_ptr[PrepareBatchGraph](new PrepareBatchGraph(aggregatorID, node_init_dim, edge_init_dim, ignore_covered_edges, 
                                                                                           include_selected_nodes, embeddingMethod))

    def SetupTrain(self, idxes, g_list, covered, list actions):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).NN_ratio = _g.NN_ratio
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            deref(inner_Graph).node_feats = _g.node_feats
            deref(inner_Graph).EdgeWeight = _g.EdgeWeight
            inner_glist.push_back(inner_Graph)

        cdef int *refint = <int*>malloc(len(actions)*sizeof(int))
        cdef int i
        for i in range(len(actions)):
            refint[i] = actions[i]
        deref(self.inner_PrepareBatchGraph).SetupTrain(idxes, inner_glist, covered, refint)
        free(refint)

    def SetupPredAll(self, idxes, g_list, covered):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).NN_ratio = _g.NN_ratio
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            deref(inner_Graph).node_feats = _g.node_feats
            deref(inner_Graph).EdgeWeight = _g.EdgeWeight
            inner_glist.push_back(inner_Graph)
        deref(self.inner_PrepareBatchGraph).SetupPredAll(idxes,inner_glist,covered)

    @property
    def act_select(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).act_select)
        return self.ConvertSparseToTensor(self.matrix)
    @property
    def rep_global(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).rep_global)
        return self.ConvertSparseToTensor(matrix)
    @property
    def n2nsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2nsum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def laplacian_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).laplacian_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def subgsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).subgsum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def e2nsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).e2nsum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def n2esum_param_0(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2esum_param_0)
        return self.ConvertSparseToTensor(matrix)
    @property
    def n2esum_param_1(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2esum_param_1)
        return self.ConvertSparseToTensor(matrix)
    @property
    def start_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).start_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def end_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).end_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def state_sum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).state_sum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def state_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).state_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def mask_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).mask_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def idx_map_list(self):
        return deref(self.inner_PrepareBatchGraph).idx_map_list
    @property
    def subgraph_id_span(self):
        return deref(self.inner_PrepareBatchGraph).subgraph_id_span
    @property
    def aux_feat(self):
        return deref(self.inner_PrepareBatchGraph).aux_feat
    @property
    def node_feats(self):
        return deref(self.inner_PrepareBatchGraph).node_feats
    @property
    def edge_feats(self):
        return deref(self.inner_PrepareBatchGraph).edge_feats
    @property
    def edge_sum(self):
        return deref(self.inner_PrepareBatchGraph).edge_sum
    @property
    def aggregatorID(self):
        return deref(self.inner_PrepareBatchGraph).aggregatorID
    @property
    def avail_node_cnt(self):
        return deref(self.inner_PrepareBatchGraph).avail_node_cnt
    @property
    def avail_edge_cnt(self):
        return deref(self.inner_PrepareBatchGraph).avail_edge_cnt

    cdef ConvertSparseToTensor(self, sparseMatrix matrix):

        rowIndex = matrix.rowIndex
        colIndex = matrix.colIndex
        data = matrix.value
        rowNum = matrix.rowNum
        colNum = matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()
        return tf.SparseTensorValue(indices, data, (rowNum,colNum))




