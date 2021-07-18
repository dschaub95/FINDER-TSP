#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "stdio.h"


Graph::Graph() : num_nodes(0), num_edges(0), NN_percent(0)
{
    edge_list.clear();
    adj_list.clear();
    edge_weights.clear();
    EdgeWeight.clear();
    node_feats.clear();
}

Graph::~Graph()
{
    edge_list.clear();
    adj_list.clear();
    edge_weights.clear();
    EdgeWeight.clear();
    node_feats.clear();
    num_nodes = 0;
    num_edges = 0;
    NN_percent = 0;
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, 
             const double* _edge_weights, double** _node_feats, const double _NN_percent)
        : num_nodes(_num_nodes), num_edges(_num_edges), NN_percent(_NN_percent)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    node_feats.resize(num_nodes);
    EdgeWeight.resize(num_nodes, std::vector<double>(num_nodes, 0.0));
    
    // printf("Clearing edge weights list\n");
    // calc number of allowed neighbors based on input percent value
    int knn = std::ceil(NN_percent * (num_nodes - 1));
    
    // get number of node features per node
    // save node features, hard coded 2 for now
    for (int i = 0; i < num_nodes; ++i)
    {
        adj_list[i].clear();
        node_feats[i].clear(); 
        for (int j = 0; j < 2; ++j)
        {
            node_feats[i].push_back(_node_feats[i][j]);
        }
    }
    // check whether the graph has already been truncated (number edges is at least as high as given by the kNN)
    
    if (num_edges < num_nodes * (num_nodes - 1) / 2 || knn == num_nodes - 1 || knn == 0)
    {
        // printf("knn: %d \n", knn);
        // printf("num nodes: %d \n", num_nodes);
        for (int i = 0; i < num_edges; ++i)
        {
            int x = edges_from[i], y = edges_to[i];
            adj_list[x].push_back(y);
            adj_list[y].push_back(x);
            edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
            edge_weights.push_back(_edge_weights[i]);
            EdgeWeight[x][y] = _edge_weights[i];
            EdgeWeight[y][x] = _edge_weights[i];
        }
    }
    else
    {
        // make sure 
        assert(num_edges == num_nodes * (num_nodes - 1) / 2);
        // printf("Hello: %d\n", knn);
        std::vector< std::pair<int, double> > neighbors;    
        int k = 0;
        for (int i = 0; i < num_nodes; ++i)
        {
            neighbors.clear();
            for (int j = 0; j < num_nodes; ++j)
            {
                if (j == i)
                {
                    continue;
                }
                if (k < num_edges)
                {
                    int x = edges_from[k], y = edges_to[k];
                    EdgeWeight[x][y] = _edge_weights[k];
                    EdgeWeight[y][x] = _edge_weights[k];
                }
                neighbors.push_back( std::make_pair(j, EdgeWeight[i][j]) );
                ++k;
            }
            // select only the x% nearest neighbors of each node
            std::sort(neighbors.begin(), neighbors.end(), []
                (const std::pair<int, double>& x, const std::pair<int, double>& y){
                return x.second < y.second;
            });

            int n = neighbors.size();
            if (knn >= 0 && n > knn)
            {
                n = knn;
            }
            for (int j = 0; j < n; ++j)
            {
                int neighbor = neighbors[j].first;
                double edge_weight = neighbors[j].second;
                std::pair< int, int > cur_edge =  std::make_pair(i, neighbor);
                std::pair< int, int > counter_edge =  std::make_pair(neighbor, i);

                
                // check whether the edge is already in the edge list and only add it if thats not the case
                if (std::find(edge_list.begin(), edge_list.end(), counter_edge) == edge_list.end()) {
		            edge_list.push_back(cur_edge);
                    edge_weights.push_back(edge_weight);

                    adj_list[i].push_back(neighbor);
                    adj_list[neighbor].push_back(i);
	            }
            }
            
        }
        num_edges = 0;
        for (int i = 0; i < num_nodes; ++i)
            num_edges += adj_list[i].size();
    }
    
    // printf("node_feats: %f, %f \n", node_feats[0][0], node_feats[0][1]);
    // printf("Sucessfully added %d elements to edge weights list for a graph of size %d\n", (int)edge_weights.size(), num_nodes);
    // printf("edge_weight: %f\n", edge_weights[0]);
}

GSet::GSet()
{
    graph_pool.clear();
}

GSet::~GSet()
{
    graph_pool.clear();
}

void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
//    printf("graph_pool_size:%d",graph_pool.size());
    assert(graph_pool.size());
//    printf("graph_pool_size:%d",graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;