#include "PrepareBatchGraph.h"
#include <math.h>

sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
}

sparseMatrix::~sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
    rowIndex.clear();
    colIndex.clear();
    value.clear();
}

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID, int _node_init_dim, int _edge_init_dim, int _ignore_covered_edges, int _include_selected_nodes, int _embeddingMethod)
{
    aggregatorID = _aggregatorID;
    node_init_dim = _node_init_dim;
    edge_init_dim = _edge_init_dim;
    include_selected_nodes = _include_selected_nodes;
    ignore_covered_edges = _ignore_covered_edges;
    embeddingMethod = _embeddingMethod;
    
    idx_map_list.clear();
    node_feats.clear();
    edge_feats.clear();
    edge_sum.clear();
    aux_feat.clear();
    avail_node_cnt.clear();
    avail_edge_cnt.clear();
}

PrepareBatchGraph::~PrepareBatchGraph()
{
    aggregatorID = -1;
    node_init_dim = -1;
    edge_init_dim = -1;
    ignore_covered_edges = -1;
    include_selected_nodes = -1;
    embeddingMethod = -1;
    
    act_select = nullptr;
    rep_global = nullptr;
    n2nsum_param = nullptr;
    subgsum_param = nullptr;
    laplacian_param = nullptr;
    e2nsum_param = nullptr;
    n2esum_param_0 = nullptr;
    n2esum_param_1 = nullptr;
    start_param = nullptr;
    end_param = nullptr;
    state_sum_param = nullptr;
    state_param = nullptr;
    mask_param = nullptr;

    idx_map_list.clear();
    node_feats.clear();
    edge_feats.clear();
    edge_sum.clear();
    aux_feat.clear();
    avail_node_cnt.clear();
    avail_edge_cnt.clear();
}

std::set<int> PrepareBatchGraph::GetToBeDeletedNodes(std::shared_ptr<Graph> g, int num, const int* covered)
{
    std::set<int> to_be_deleted_nodes;
    switch(include_selected_nodes)
    {
        case 0:
        {
            // delete all nodes that have been selected until this point 
            for (int i = 0; i < num; ++i)
            {
                to_be_deleted_nodes.insert(covered[i]);
            }
            break;
        }
        case 1:
        {
            // include first and last selected node so it is available during network embedding
            for (int i = 1; i < num-1; ++i)
            {
                to_be_deleted_nodes.insert(covered[i]);
            }
            break;
        }
        case 2:
        {
            // keep all nodes and just change node features + edge features later depending on whether node was already chosen
            // also possibly delete links between selected nodes
            break;
        }
        default:
            break;
    }
    return to_be_deleted_nodes;    
}

int PrepareBatchGraph::GetNodeStatus(std::shared_ptr<Graph> g, int num, const int* covered, std::vector<int>& idx_map, std::set<int> to_be_deleted_nodes)
{
    idx_map.resize(g->num_nodes);
    for (int i = 0; i < g->num_nodes; ++i)
    {
        idx_map[i] = -1;
    }
    // number of nodes which are completely uncovered
    int node_cnt = 0;
    for (int j = 0; j < (int)g->num_nodes; ++j)
    {
        if (to_be_deleted_nodes.count(j) == false)
        {
            idx_map[j] = 0;
            node_cnt++;
        }
    }
    return node_cnt;
}

int PrepareBatchGraph::GetEdgeStatus(std::shared_ptr<Graph> g, int num, const int* covered, std::set<int> to_be_deleted_nodes)
{
    std::set<int> covered_set;
    for (int i = 0; i < num; ++i)
    {
        covered_set.insert(covered[i]);
    }

    int edge_cnt = 0;
    for (auto& p : g->edge_list)
    {
        edge_cnt += 2;
        if (ignore_covered_edges == 1)
        {
            if (to_be_deleted_nodes.count(p.first) || to_be_deleted_nodes.count(p.second))
            {
                edge_cnt -= 2;
                continue;
            }
            if (covered_set.count(p.first) && covered_set.count(p.second))
            {
                edge_cnt -= 2;
            }
        }
        else
        {
            if (to_be_deleted_nodes.count(p.first) || to_be_deleted_nodes.count(p.second))
            {
                edge_cnt -= 2;
            }
        }
    }
    // add links between subsequent nodes in the current tour
    if (ignore_covered_edges == 1)
    {
        if (num > 1)
        {
            edge_cnt += 2 * (num - to_be_deleted_nodes.size());
        }
        if (include_selected_nodes == 1 && num >= 2)
        {
            edge_cnt -= 4;
        }
    }
    // ignore inner edges between nodes that have already been selected
    return edge_cnt;
}

std::vector<int> PrepareBatchGraph::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, 
                                                  std::vector<int>& idx_map)
// calculates number of available actions
{
    std::set<int> to_be_deleted_nodes;
    std::set<int> covered_set;
    std::vector<int> resultList;
    resultList.resize(2);
    idx_map.resize(g->num_nodes);

    for (int i = 0; i < g->num_nodes; ++i)
    {
        idx_map[i] = -1;
    }
    
    for (int i = 0; i < num; ++i)
    {
        covered_set.insert(covered[i]);
    }

    switch(include_selected_nodes)
    {
        case 0:
        {
            // delete all nodes that have been selected until this point 
            for (int i = 0; i < num; ++i)
            {
                to_be_deleted_nodes.insert(covered[i]);
            }
            break;
        }
        case 1:
        {
            // include first and last selected node so it is available during network embedding
            for (int i = 1; i < num-1; ++i)
            {
                to_be_deleted_nodes.insert(covered[i]);
            }
            break;
        }
        case 2:
        {
            // keep all nodes and just change node features + edge features later depending on whether node was already chosen
            // also possibly delete links between selected nodes
            break;
        }
        default:
            break;
    }    
    // number of nodes which are completely uncovered
    int node_cnt = 0;
    int edge_cnt = 0;
    // iterate over all edges
    for (int j = 0; j < (int)g->num_nodes; ++j)
    {
        if (to_be_deleted_nodes.count(j) == false)
        {
            idx_map[j] = 0;
            node_cnt++;
        }
    }
    for (auto& p : g->edge_list)
    {
        edge_cnt += 2;
        if (ignore_covered_edges == 1)
        {
            if (to_be_deleted_nodes.count(p.first) || to_be_deleted_nodes.count(p.second))
            {
                edge_cnt -= 2;
                continue;
            }
            if (covered_set.count(p.first) && covered_set.count(p.second))
            {
                edge_cnt -= 2;
            }
        }
        else
        {
            if (to_be_deleted_nodes.count(p.first) || to_be_deleted_nodes.count(p.second))
            {
                edge_cnt -= 2;
            }
        }
    }
    // add links between subsequent nodes in the current tour
    if (ignore_covered_edges == 1)
    {
        if (num > 1)
        {
            edge_cnt += 2 * (num - to_be_deleted_nodes.size());
        }
        if (include_selected_nodes == 1 && num >= 2)
        {
            edge_cnt -= 4;
        }
    }
    // ignore inner edges between nodes that have already been selected
    resultList[0] = node_cnt;
    resultList[1] = edge_cnt;
    return resultList;
}

void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                                        std::vector< std::shared_ptr<Graph> > g_list, 
                                        std::vector< std::vector<int> > covered, 
                                        const int* actions)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    start_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    end_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    state_sum_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    state_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    mask_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    // idxes.size() = BATCHSIZE, vector of vectors
    idx_map_list.resize(idxes.size());
    // simple vector < int >
    avail_node_cnt.resize(idxes.size());
    avail_edge_cnt.resize(idxes.size());

    int node_cnt = 0;
    int edge_cnt = 0;
    int max_nodes = 0;
    for (int i = 0; i < (int)idxes.size(); ++i)
    {   
        // possibility to include more handcrafted features
        std::vector<double> temp_feat;
        temp_feat.push_back(1.0);
        aux_feat.push_back(temp_feat);

        auto g = g_list[idxes[i]];
        // get the maximum number of nodes of a graph in the batch
        if (g->num_nodes > max_nodes)
        {
            max_nodes = g->num_nodes;
        }
        // calc counter, twohop_number and threehop_number + avail
        auto resultStatus = GetStatusInfo(g, covered[idxes[i]].size(), covered[idxes[i]].data(), idx_map_list[i]);
        avail_node_cnt[i] = resultStatus[0];
        avail_edge_cnt[i] = resultStatus[1];

        node_cnt += avail_node_cnt[i];
        // calculate edge count
        edge_cnt += avail_edge_cnt[i];
    }
    // printf("max nodes: %d\n", max_nodes);
    // (Batchsize, all available actions in all graphs)
    // prepare in and out edges size, and subgraphs(=remaining parts of the current graph)
    // printf("Edge count calc: %d\n", edge_cnt);
    // printf("Node count: %d\n", node_cnt);
    // printf("Edge count full: %d\n", node_cnt*(node_cnt-1));
    // printf("Idx count: %d\n", idxes.size());
    graph.Resize(idxes.size(), node_cnt);
    edge_sum.resize(node_cnt, std::vector<double>(edge_init_dim, 1.0));
    node_feats.resize(node_cnt, std::vector<double>(node_init_dim, 1.0));
    edge_feats.resize(edge_cnt, std::vector<double>(edge_init_dim , 1.0));
    if (actions)
    {
        act_select->rowNum = idxes.size();
        act_select->colNum = node_cnt;
    }
    
    rep_global->rowNum = node_cnt;
    rep_global->colNum = idxes.size();
    
    start_param->rowNum = idxes.size();
    start_param->colNum = node_cnt;

    end_param->rowNum = idxes.size();
    end_param->colNum = node_cnt;

    state_sum_param->rowNum = idxes.size();
    state_sum_param->colNum = node_cnt;

    state_param->rowNum = max_nodes * idxes.size();
    state_param->colNum = node_cnt;

    mask_param->rowNum = node_cnt;
    mask_param->colNum = node_cnt;

    node_cnt = 0;
    edge_cnt = 0;
    // make calculations on each graph in the batch
    for (int i = 0; i < (int)idxes.size(); ++i)
    {             
        auto g = g_list[idxes[i]];
        auto idx_map = idx_map_list[i];
        int start_node = -1;
        int last_node = -1;
        std::set<int> c;
        // get start and last node info
        for (int j = 0; j < (int)covered[idxes[i]].size(); ++j)
        {
            c.insert(covered[idxes[i]][j]);  
        }
        if (covered[idxes[i]].size() > 0)
        {
            int last_idx = covered[idxes[i]].size() - 1;
            start_node = covered[idxes[i]][0];
            last_node = covered[idxes[i]][last_idx];
        }
        // extract info per node
        int t = 0;
        for (int j = 0; j < (int)g->num_nodes; ++j)
        {   
            // masks all node indices that are unavailable
            if (idx_map[j] < 0)
            {
                continue;
            }
            idx_map[j] = t;
            // change in case of simple node weights can be adapted to capture node positions
            // Node init feature pattern for available node: [x,y,1,1,1,1], last selected node: [x,y,0,1,1,1],
            // selected nodes (between last and start): [x,y,0,0,1,1], start node: [x,y,0,0,0,1]
            for (int k = 0; k < 2; ++k)
            {
                node_feats[node_cnt + t][k] = g->node_feats[j][k];
            }
            if (c.count(j) && node_init_dim >= 5)
            {
                node_feats[node_cnt + t][4] = 0.0;
                // zero means "available for selection"
                if (j == start_node)
                {
                    node_feats[node_cnt + t][3] = 0.0;
                    node_feats[node_cnt + t][2] = 0.0;
                }
                else if (j == last_node)
                {
                    node_feats[node_cnt + t][3] = 1.0;
                }
            }
            graph.AddNode(i, node_cnt + t);
            // needed for adding last and start node embeddings, other than that only needed in case !actions 
            rep_global->rowIndex.push_back(node_cnt + t);
            rep_global->colIndex.push_back(i);
            rep_global->value.push_back(1.0);
            // create matrices to extract the embeddings of both the last selected and the start node
            if (j == start_node)
            {
                start_param->rowIndex.push_back(i);
                start_param->colIndex.push_back(node_cnt + t);
                start_param->value.push_back(1.0);
            }
            if (j == last_node)
            {
                end_param->rowIndex.push_back(i);
                end_param->colIndex.push_back(node_cnt + t);
                end_param->value.push_back(1.0);
            }
            if (c.count(j))
            {
                state_sum_param->rowIndex.push_back(i);
                state_sum_param->colIndex.push_back(node_cnt + t);
                state_sum_param->value.push_back(1/(double)c.size());
            }
            else
            {
                // mask all covered nodes, only keep uncovered ones
                mask_param->rowIndex.push_back(node_cnt + t);
                mask_param->colIndex.push_back(node_cnt + t);
                mask_param->value.push_back(1.0);
            }
            t += 1;
        }
        // iterate over all covered nodes in order to extract their corresponding node embeddings
        for (int k = 0; k < (int)covered[idxes[i]].size(); ++k)
        {
            int node = covered[idxes[i]][k];
            state_param->rowIndex.push_back(i * max_nodes + k);
            state_param->colIndex.push_back(node_cnt + idx_map[node]);
            state_param->value.push_back(1.0);
        }
        assert(t == avail_node_cnt[i]);
        if (actions)
        {   
            auto act = actions[idxes[i]];
            assert(idx_map[act] >= 0 && act >= 0 && act < g->num_nodes);
            act_select->rowIndex.push_back(i);
            act_select->colIndex.push_back(node_cnt + idx_map[act]);
            act_select->value.push_back(1.0);
        }
        for (auto p : g->edge_list)
        {   
            // check whether one node of the edge has been selected and removed from the remaining graph
            if (idx_map[p.first] < 0 || idx_map[p.second] < 0)
            {
               continue; 
            }
            // ignore inner edges between nodes that have already been selected
            if (c.count(p.first) && c.count(p.second))
            {
                if (ignore_covered_edges == 1)
                {
                    continue;
                }
            }
            auto x = idx_map[p.first] + node_cnt, y = idx_map[p.second] + node_cnt;
            // add code to determine edge weight corresponding to the current edge
            double edge_weight = g->EdgeWeight[p.first][p.second];
            // printf("Index x: %d, Index y: %d\n", x, y);
            graph.AddEdge(edge_cnt, x, y, edge_weight);
            edge_feats[edge_cnt][0] = edge_weight;
            if (edge_init_dim > 2)
            {
                edge_feats[edge_cnt][1] = (double) (c.count(p.first) ^ c.count(p.second));
                edge_feats[edge_cnt][2] = (double) c.count(p.first);
            }
            
            // edge_feats[edge_cnt][3] = y;
            // edge_feats[edge_cnt][4] = x;
            edge_cnt += 1;
            graph.AddEdge(edge_cnt, y, x, edge_weight);
            edge_feats[edge_cnt][0] = edge_weight;
            
            if (edge_init_dim > 2)
            {
                edge_feats[edge_cnt][1] = (double) (c.count(p.first) ^ c.count(p.second));
                edge_feats[edge_cnt][2] = (double) c.count(p.second);
            }
            
            // edge_feats[edge_cnt][3] = y;
            // edge_feats[edge_cnt][4] = x;
            edge_cnt += 1;
        }
        if (ignore_covered_edges == 1)
        {
            if ((int)covered[idxes[i]].size() > 1 && include_selected_nodes == 2)
            {
                for (int j = 0; j < (int)covered[idxes[i]].size(); ++j)
                {
                    int n_c = covered[idxes[i]][j];
                    int next_c = covered[idxes[i]][0];
                    if (idx_map[n_c] < 0 || idx_map[next_c] < 0)
                    {
                        continue;
                    }
                    if (j + 1 < (int)covered[idxes[i]].size())
                    {
                        next_c = covered[idxes[i]][j + 1];
                    }
                    auto x = idx_map[n_c] + node_cnt, y = idx_map[next_c] + node_cnt;
                    // add code to determine edge weight corresponding to the current edge
                    double edge_weight = g->EdgeWeight[n_c][next_c];
                    graph.AddEdge(edge_cnt, x, y, edge_weight);
                    edge_feats[edge_cnt][0] = edge_weight;
                    if (edge_init_dim > 2)
                    {
                        edge_feats[edge_cnt][1] = 0.0;
                        edge_feats[edge_cnt][2] = 1.0;
                    }
                    edge_cnt += 1;
                    graph.AddEdge(edge_cnt, y, x, edge_weight);
                    edge_feats[edge_cnt][0] = edge_weight;
                    if (edge_init_dim > 2)
                    {
                        edge_feats[edge_cnt][1] = 0.0;
                        edge_feats[edge_cnt][2] = 1.0;
                    }
                    edge_cnt += 1;
                }
            }
        }
        // add all available actions in the current state to the node count
        node_cnt += avail_node_cnt[i];
    }
    // printf("Edge count: %d\n", graph.num_edges);
    assert(node_cnt == (int)graph.num_nodes);
    auto result_list = n2n_construct(&graph, aggregatorID);
    n2nsum_param = result_list[0];
    laplacian_param = result_list[1];
    subgsum_param = subg_construct(&graph, subgraph_id_span, aggregatorID);
    
    if (embeddingMethod == 2)
    {
        e2nsum_param = e2n_construct(&graph, aggregatorID);
    }
    else if (embeddingMethod == 3)
    {
        e2nsum_param = e2n_construct(&graph, aggregatorID);
        auto n2e_result_list = n2e_construct(&graph);
        n2esum_param_0 = n2e_result_list[0];
        n2esum_param_1 = n2e_result_list[1];
    } 
}
void PrepareBatchGraph::SetupNodeInput(std::vector<int> idxes,
                                       std::vector< std::shared_ptr<Graph> > g_list,
                                       std::vector< std::vector<int> > covered)
{
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    start_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    end_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    state_sum_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    state_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    mask_param = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    idx_map_list.resize(idxes.size());
}

void PrepareBatchGraph::SetupTrain(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions)
{
    SetupGraphInput(idxes, g_list, covered, actions);
}

void PrepareBatchGraph::SetupPredAll(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered)
{
    SetupGraphInput(idxes, g_list, covered, nullptr);
}


std::vector<std::shared_ptr<sparseMatrix>> PrepareBatchGraph::n2n_construct(GraphStruct* graph, int aggregatorID)
// defines aggregation of nodefeatures
// computes both the block laplacian as well as the block aggregation matrix
// also sums edge weights of node neighborhoods and saves them into vector
{
    //aggregatorID = 0 sum
    //aggregatorID = 1 mean
    //aggregatorID = 2 GCN
    std::vector<std::shared_ptr<sparseMatrix>> resultList;
    resultList.resize(2);
    std::shared_ptr<sparseMatrix> result = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;

    std::shared_ptr<sparseMatrix> result_laplacian = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result_laplacian->rowNum = graph->num_nodes;
    result_laplacian->colNum = graph->num_nodes;

    double edge_weight_sum = 0;
	for (int i = 0; i < (int)graph->num_nodes; ++i)
	{
		// list of all incoming edges of the corresponding node
        auto& list = graph->in_edges->head[i];

        if (list.size() > 0)
        {
            // push back degree of the node
            result_laplacian->value.push_back(list.size());
		    
            result_laplacian->rowIndex.push_back(i);
		    result_laplacian->colIndex.push_back(i);
        }
        edge_weight_sum = 0;
		for (int j = 0; j < (int)list.size(); ++j)
		{
		    switch(aggregatorID)
            {
		        case 0:
		        {
                    result->value.push_back(1.0);
                    break;
		        }
		        case 1:
		        {
                    result->value.push_back(1.0/(double)list.size());
                    break;
		        }
		        case 2:
		        {
                    int neighborDegree = (int)graph->in_edges->head[list[j].second].size();
                    int selfDegree = (int)list.size();
                    double norm = sqrt((double)(neighborDegree+1))*sqrt((double)(selfDegree+1));
                    result->value.push_back(1.0/norm);
                    break;
		        }
                case 3:
		        {
                    result->value.push_back(sqrt(1/graph->edge_weights->head[i][j]));
                    break;
		        }
		        default:
		            break;
		    }
            // add edge weight to the sum for that specific node
            edge_weight_sum += graph->edge_weights->head[i][j];
            // push back the node index
            result->rowIndex.push_back(i);
            // printf("row_index: %d, col_index: %d\n", i, list[j].second);

            // and the neighbour node index corresponding to the jth incoming edge for node i
            result->colIndex.push_back(list[j].second);
            result_laplacian->value.push_back(-1.0);
		    result_laplacian->rowIndex.push_back(i);
		    result_laplacian->colIndex.push_back(list[j].second);

		}
        // add the sum of all edge weights to neighbouring nodes
        edge_sum[i][0] = edge_weight_sum;
        // node_feats[i][4] = edge_weight_sum;
	}
	resultList[0] = result;
	resultList[1] = result_laplacian;
    return resultList;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph, int aggregatorID)
{
    std::shared_ptr<sparseMatrix> result = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;
	for (int i = 0; i < (int)graph->num_nodes; ++i)
	{
        auto& list = graph->in_edges->head[i];
		for (int j = 0; j < (int)list.size(); ++j)
		{
            switch(aggregatorID)
            {
		        case 0:
		        {
                    result->value.push_back(1.0);
                    break;
		        }
		        case 1:
		        {
                    result->value.push_back(1.0/(double)list.size());
                    break;
		        }
		        default:
		            result->value.push_back(1.0);
                    break;
		    }
            // result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            // printf("Currently considered node: %d\n", i);
            result->colIndex.push_back(list[j].first);
            // printf("in edge %d first entry %d\n", j, list[j].first);
            // printf("in edge %d second entry %d\n", j, list[j].second);
		}
	}
    return result;
}

std::vector< std::shared_ptr<sparseMatrix> > n2e_construct(GraphStruct* graph)
{
    std::vector<std::shared_ptr<sparseMatrix>> resultList;
    resultList.resize(2);
    std::shared_ptr<sparseMatrix> result_first = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result_first->rowNum = graph->num_edges;
    result_first->colNum = graph->num_nodes;

    std::shared_ptr<sparseMatrix> result_sec = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result_sec->rowNum = graph->num_edges;
    result_sec->colNum = graph->num_nodes;


	for (int i = 0; i < (int)graph->num_edges; ++i)
	{
        result_first->value.push_back(1.0);
        result_first->rowIndex.push_back(i);
        result_first->colIndex.push_back(graph->edge_list[i].first);

        result_sec->value.push_back(1.0);
        result_sec->rowIndex.push_back(i);
        result_sec->colIndex.push_back(graph->edge_list[i].second);


	}
    resultList[0] = result_first;
	resultList[1] = result_sec;
    return resultList;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span, int aggregatorID)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph;
    result->colNum = graph->num_nodes;

    subgraph_id_span.clear();
    int start = 0;
    int end = 0;
	for (int i = 0; i < (int)graph->num_subgraph; ++i)
	{

		auto& list = graph->subgraph->head[i];
        end  = start + list.size() - 1;
		for (int j = 0; j < (int)list.size(); ++j)
		{
            if (aggregatorID == 1)
            {
                result->value.push_back(1.0/(double)list.size());
            }
            else
            {
                result->value.push_back(1.0);
            }
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
		if (list.size() > 0){
		    subgraph_id_span.push_back(std::make_pair(start, end));
		}
		else{
		    subgraph_id_span.push_back(std::make_pair(graph->num_nodes, graph->num_nodes));
		}
		start = end + 1 ;
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
// not used in current code
{
    std::shared_ptr<sparseMatrix> result = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (int i = 0; i < (int)graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first;
        int node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from];
        for (int j = 0; j < (int)list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
        }
    }
    return result;
}

