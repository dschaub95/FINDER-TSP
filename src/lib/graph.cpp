#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "stdio.h"
#include <math.h>
#include <limits>

typedef std::numeric_limits< double > dbl;

Graph::Graph() : num_nodes(0), num_edges(0), NN_ratio(0)
{
    edge_list.clear();
    adj_list.clear();
    EdgeWeight.clear();
    edge_probs.clear();
    node_feats.clear();
}

Graph::~Graph()
{
    edge_list.clear();
    adj_list.clear();
    EdgeWeight.clear();
    edge_probs.clear();
    node_feats.clear();
    num_nodes = 0;
    num_edges = 0;
    NN_ratio = 0;
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, 
             double** _EdgeWeight, double** _edge_probs, double** _node_feats, const double _NN_ratio)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    NN_ratio = _NN_ratio;
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    node_feats.resize(num_nodes);
    EdgeWeight.resize(num_nodes, std::vector<double>(num_nodes, 0.0));
    edge_probs.resize(num_nodes, std::vector<double>(num_nodes, 0.0));
    
    
    double min_prob = 1.0;
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
        for (int j = 0; j < num_nodes; ++j)
        {
            EdgeWeight[i][j] = _EdgeWeight[i][j];
            // cut off edge probs below cut-off value
            if (_edge_probs[i][j] < pow(10,-16))
            {
                edge_probs[i][j] = 0.0;
            }
            else
            {
                edge_probs[i][j] = _edge_probs[i][j];
            }
            
        
            // if (edge_probs[i][j] < min_prob && edge_probs[i][j] > 0.0)
            // {
            //     min_prob = edge_probs[i][j];
            // }
        }
    }
    // std::cout.precision(dbl::max_digits10);
    // std::cout << "Min prob: " << min_prob << std::endl;
    // check whether the graph has already been truncated (number edges is at least as high as given by the kNN)
    // calc number of allowed neighbors based on input percent value
    if (NN_ratio < 1.0)
    {
        SparsifyWithKNN();
    }
    else if (1)
    {
        SparsifyWithProbabilities(edges_from, edges_to);
    }
    else
    {
        // printf("knn: %d \n", knn);
        // printf("num nodes: %d \n", num_nodes);
        for (int i = 0; i < num_edges; ++i)
        {
            int x = edges_from[i], y = edges_to[i];
            adj_list[x].push_back(y);
            adj_list[y].push_back(x);
            edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
        }
    }
}

void Graph::SparsifyWithProbabilities(const int* edges_from, const int* edges_to)
{
    assert(num_edges == num_nodes * (num_nodes - 1) / 2);
    edge_list.clear();
    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        // printf("x: %d \n", x);
        // printf("y: %d \n", y);
        // check if edge prob > 0.0
        // printf("edge prob: %f \n", edge_probs[x][y]);
        if (edge_probs[x][y] > 0.0)
        {
            adj_list[x].push_back(y);
            adj_list[y].push_back(x);
            edge_list.push_back(std::make_pair(edges_from[i], edges_to[i]));
        }
        else
        {
            continue;
        }
    }
    num_edges = edge_list.size();
    // printf("num edges: %d \n", num_edges);
}

void Graph::SparsifyWithKNN()
{
    // check whether the graph has already been truncated (number edges is at least as high as given by the kNN)
    int knn = std::ceil(NN_ratio * (num_nodes - 1));
    // make sure 
    assert(num_edges == num_nodes * (num_nodes - 1) / 2);
    //printf("knn: %d\n", knn);
    //printf("nn percent: %f\n", NN_ratio);
    //printf("num edges: %d\n", num_edges);
    //printf("num nodes: %d\n", num_nodes);
    edge_list.clear();
    std::vector< std::pair< int, double > > neighbors;    
    for (int i = 0; i < num_nodes; ++i)
    {
        // get extract neighborhood info
        neighbors.clear();
        for (int j = 0; j < num_nodes; ++j)
        {
            // printf("%f ", EdgeWeight[i][j]);
            if (j == i)
            {
                continue;
            }
            neighbors.push_back( std::make_pair(j, EdgeWeight[i][j]) ); 
            
            //printf("EdgeWeight[%d][%d]: %f\n", i, j, EdgeWeight[i][j]);
            
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
            std::pair< int, int > cur_edge =  std::make_pair(i, neighbor);
            std::pair< int, int > counter_edge =  std::make_pair(neighbor, i);

            // check whether the edge is already in the edge list and only add it if thats not the case
            if (std::find(edge_list.begin(), edge_list.end(), counter_edge) == edge_list.end()) {
                //printf("edge first: %d\n", cur_edge.first);
                //printf("edge second: %d\n", cur_edge.second);
                edge_list.push_back(cur_edge);

                adj_list[i].push_back(neighbor);
                adj_list[neighbor].push_back(i);
            }
        }
    }
    num_edges = edge_list.size();
    //printf("num edges: %d\n", num_edges);
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