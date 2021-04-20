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

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID)
{
    aggregatorID = _aggregatorID;
}

PrepareBatchGraph::~PrepareBatchGraph()
{
    act_select = nullptr;
    rep_global = nullptr;
    n2nsum_param = nullptr;
    subgsum_param = nullptr;
    laplacian_param = nullptr;
    idx_map_list.clear();
    node_feats.clear();
    edge_sum.clear();
    aux_feat.clear();
    avail_act_cnt.clear();
    aggregatorID = -1;
}

int PrepareBatchGraph::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, int& counter, int& twohop_number, int& threehop_number, std::vector<int>& idx_map)
// calculates number of available actions
{
    std::set<int> c;

    idx_map.resize(g->num_nodes);

    for (int i = 0; i < g->num_nodes; ++i)
    {
        idx_map[i] = -1;
    }
        
    for (int i = 0; i < num; ++i)
    {
        c.insert(covered[i]);  
    }
        
    counter = 0;

    twohop_number = 0;
    threehop_number = 0;
    std::set<int> node_twohop_set;

    int n = 0;
 	std::map<int,int> node_twohop_counter;
    // iterate over all edges
    for (auto& p : g->edge_list)
    {
        // find all edges that are connected to at least one covered node
        // printf("Edge: (%d, %d) \n", p.first, p.second);
        if (c.count(p.first) || c.count(p.second))
        {
            counter++;         
        } 
        else 
        {
            // 
            if (idx_map[p.first] < 0)
            {
                n++;
            }
            if (idx_map[p.second] < 0)
            {
                n++;
            }
                
            idx_map[p.first] = 0;
            idx_map[p.second] = 0;

            if (node_twohop_counter.find(p.first) != node_twohop_counter.end())
            {
                twohop_number += node_twohop_counter[p.first];
                node_twohop_counter[p.first] = node_twohop_counter[p.first] + 1;
            }
            else
            {
                node_twohop_counter.insert(std::make_pair(p.first,1));
            }

            if (node_twohop_counter.find(p.second) != node_twohop_counter.end())
            {
                twohop_number += node_twohop_counter[p.second];
                node_twohop_counter[p.second] = node_twohop_counter[p.second] + 1;
            }
            else
            {
                node_twohop_counter.insert(std::make_pair(p.second,1));
            }
        }
    }
    // printf("num nodes: %d \n", g->num_nodes);
    // printf("n: %d \n", n);  
    return n;
}

void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list, 
                           std::vector< std::vector<int> > covered, 
                           const int* actions)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());

    // idxes.size() = BATCHSIZE, vector of vectors
    idx_map_list.resize(idxes.size());
    // simple vector < int >
    avail_act_cnt.resize(idxes.size());

    int node_cnt = 0;

    for (int i = 0; i < (int)idxes.size(); ++i)
    {   
        std::vector<double> temp_feat;

        auto g = g_list[idxes[i]];

        int counter;
        int twohop_number;
        int threehop_number;

        // save ratio of covered nodes compared to all nodes
        if (g->num_nodes)
        {
            temp_feat.push_back((double)covered[idxes[i]].size() / (double)g->num_nodes);
        }
        // calc counter, twohop_number and threehop_number + avail
        avail_act_cnt[i] = GetStatusInfo(g, covered[idxes[i]].size(), covered[idxes[i]].data(), counter, twohop_number, threehop_number, idx_map_list[i]);

        // save ratio of edges that are connected to at least one covered node and all edges
        if (g->edge_list.size())
        {
            temp_feat.push_back((double)counter / (double)g->edge_list.size());
        }

        temp_feat.push_back((double)twohop_number / ((double)g->num_nodes * (double)g->num_nodes));

        temp_feat.push_back(1.0);

        node_cnt += avail_act_cnt[i];
        aux_feat.push_back(temp_feat);
    }
    // (Batchsize, all available actions in all graphs)
    // prepare in and out edges size, and subgraphs(=remaining parts of the current graph)
    graph.Resize(idxes.size(), node_cnt);

    if (actions)
    {
        act_select->rowNum = idxes.size();
        act_select->colNum = node_cnt;
    } else
    {
        rep_global->rowNum = node_cnt;
        rep_global->colNum = idxes.size();
    }

    node_cnt = 0;
    int edge_cnt = 0;
    // make calculations on each graph in the batch
    for (int i = 0; i < (int)idxes.size(); ++i)
    {             
        auto g = g_list[idxes[i]];
        auto idx_map = idx_map_list[i];
    
        // extract info per node
        int t = 0;
        for (int j = 0; j < (int)g->num_nodes; ++j)
        {   
            // maskes all node indices that are unaivalable
            if (idx_map[j] < 0)
            {
                continue;
            }
            idx_map[j] = t;
            
            // change in case of simple node weights can be adapted to capture node positions
            std::vector<double> temp_node_feat;
            // printf("Size node feats: %d", (int)sizeof(g->node_feats[0]));
            for (int k = 0; k < 2; ++k)
            {
                temp_node_feat.push_back(g->node_feats[j][k]);
            }
            temp_node_feat.push_back(1.0);
            node_feats.push_back(temp_node_feat);

            graph.AddNode(i, node_cnt + t);
            if (!actions)
            {
                rep_global->rowIndex.push_back(node_cnt + t);
                rep_global->colIndex.push_back(i);
                rep_global->value.push_back(1.0);
            }
            t += 1;
        }
        assert(t == avail_act_cnt[i]);

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
            if (idx_map[p.first] < 0 || idx_map[p.second] < 0)
            {
               continue; 
            }
                
            auto x = idx_map[p.first] + node_cnt, y = idx_map[p.second] + node_cnt;
            // add code to determine edge weight corresponding to the current edge
            double edge_weight = g->getEdgeWeight(p.first, p.second);
            // printf("Index x: %d, Index y: %d\n", x, y);
            graph.AddEdge(edge_cnt, x, y, edge_weight);
            edge_cnt += 1;
            graph.AddEdge(edge_cnt, y, x, edge_weight);
            edge_cnt += 1;
        }
        // add all available actions in the current state to the node count
        node_cnt += avail_act_cnt[i];
    }
    assert(node_cnt == (int)graph.num_nodes);

    auto result_list = n2n_construct(&graph, aggregatorID);
    n2nsum_param = result_list[0];
    laplacian_param = result_list[1];
    subgsum_param = subg_construct(&graph, subgraph_id_span);

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
        edge_sum.push_back(edge_weight_sum);
	}
	resultList[0] = result;
	resultList[1] = result_laplacian;
    return resultList;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span)
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
            result->value.push_back(1.0);
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
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (int i = 0; i < (int)graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
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

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
// not used in current code
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;

	for (int i = 0; i < (int)graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
// not used in the current code
{
    std::shared_ptr<sparseMatrix> result = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;
	for (int i = 0; i < (int)graph->num_nodes; ++i)
	{
        auto& list = graph->in_edges->head[i];
		for (int j = 0; j < (int)list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}
