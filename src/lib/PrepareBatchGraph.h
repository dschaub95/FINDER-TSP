#ifndef PREPAREBATCHGRAPH_H_
#define PREPAREBATCHGRAPH_H_

#include "graph.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <iterator>
#include <math.h>

class sparseMatrix
{
public:
    sparseMatrix();
    ~sparseMatrix();
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class PrepareBatchGraph{
public:
    PrepareBatchGraph(int _aggregatorID, int _node_init_dim, int _edge_init_dim, int _ignore_covered_edges, 
                      int _include_selected_nodes, int embeddingMethod, int _max_nodes);
    
    ~PrepareBatchGraph();
    
    void SetupGraphInput(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions);
    void SetupTrain(std::vector<int> idxes,
                    std::vector< std::shared_ptr<Graph> > g_list,
                    std::vector< std::vector<int> > covered,
                    const int* actions);
    void SetupPredAll(std::vector<int> idxes,
                      std::vector< std::shared_ptr<Graph> > g_list,
                      std::vector< std::vector<int> > covered);
    void AddEdgetoBatchgraph(std::shared_ptr<Graph> &g, int local_first, int local_second, int global_first, 
                             int global_second, std::set<int> c, int &edge_cnt);

    void PrepareNodeInputs(std::vector<int> idxes,
                        std::vector< std::shared_ptr<Graph> > g_list,
                        std::vector< std::vector<int> > covered);
    void SetupNodeLevelInput(std::vector<int> idxes,
                        std::vector< std::shared_ptr<Graph> > g_list,
                        std::vector< std::vector<int> > covered);

    std::vector<int> GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, std::vector<int>& idx_map);
    int GetNodeStatus(std::shared_ptr<Graph> g, int num, const int* covered, std::vector<int>& idx_map, std::set<int> to_be_deleted_nodes);
    int GetEdgeStatus(std::shared_ptr<Graph> g, int num, const int* covered, std::set<int> to_be_deleted_nodes);
    std::set<int> GetToBeDeletedNodes(std::shared_ptr<Graph> g, int num, const int* covered);

    std::shared_ptr<sparseMatrix> act_select;
    std::shared_ptr<sparseMatrix> rep_global;
    std::shared_ptr<sparseMatrix> n2nsum_param;
    std::shared_ptr<sparseMatrix> e2nsum_param;
    std::shared_ptr<sparseMatrix> n2esum_param_0;
    std::shared_ptr<sparseMatrix> n2esum_param_1;
    std::shared_ptr<sparseMatrix> laplacian_param;
    std::shared_ptr<sparseMatrix> subgsum_param;
    std::shared_ptr<sparseMatrix> start_param;
    std::shared_ptr<sparseMatrix> end_param;
    std::shared_ptr<sparseMatrix> agg_state_param;
    std::shared_ptr<sparseMatrix> state_sum_param;
    std::shared_ptr<sparseMatrix> state_param;
    std::shared_ptr<sparseMatrix> mask_param;
    std::shared_ptr<sparseMatrix> pad_node_param;
    std::shared_ptr<sparseMatrix> pad_reverse_param;

    std::vector< std::vector< int > > idx_map_list;
    std::vector< std::pair< int,int > > subgraph_id_span;
    std::vector< std::vector< double > > aux_feat;
    std::vector< std::vector< double > > node_feats;
    std::vector< std::vector< double > > edge_feats;
    // add edge sums for all availabe nodes, the last chosen node and the starting(end) node
    std::vector< std::vector< double > > edge_sum; 
    std::vector< int > avail_node_cnt;
    std::vector< int > avail_edge_cnt;

    GraphStruct graph;
    
    int aggregatorID;
    int ignore_covered_edges;
    int include_selected_nodes;
    int embeddingMethod;
    int node_init_dim; 
    int edge_init_dim;
    int max_nodes;

    std::vector<std::shared_ptr<sparseMatrix>> n2n_construct(GraphStruct* graph, int aggregatorID);
};

// std::vector<std::shared_ptr<sparseMatrix>> n2n_construct(GraphStruct* graph, int aggregatorID);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span, int aggregatorID);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph, int aggregatorID);

std::vector< std::shared_ptr<sparseMatrix> > n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

#endif