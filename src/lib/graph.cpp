#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "stdio.h"


Graph::Graph() : num_nodes(0), num_edges(0)
{
    edge_list.clear();
    adj_list.clear();
    edge_weights.clear();
    node_feats.clear();
}

Graph::~Graph()
{
    edge_list.clear();
    adj_list.clear();
    edge_weights.clear();
    node_feats.clear();
    num_nodes = 0;
    num_edges = 0;
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const double* _edge_weights, double** _node_feats)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    edge_weights.resize(num_edges);
   
    // get number of node features per node
    
    node_feats.resize(num_nodes);
    //tsp change
    // printf("Clearing edge weights list\n");
    edge_weights.clear();

    for (int i = 0; i < num_nodes; ++i)
    {
        adj_list[i].clear();
        // save node features, hard coded 2 for now
        node_feats[i].resize(2);
        node_feats[i].clear();
        for (int j = 0; j < 2; ++j)
        {
            node_feats[i].push_back(_node_feats[i][j]);
        }
    }
        
    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);
        adj_list[y].push_back(x);
        edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
        edge_weights.push_back(_edge_weights[i]);
    }
    // printf("node_feats: %f, %f \n", node_feats[0][0], node_feats[0][1]);
    // printf("Sucessfully added %d elements to edge weights list for a graph of size %d\n", (int)edge_weights.size(), num_nodes);
    //printf("edge_weight: %f\n", edge_weights[0]);
}

double Graph::getEdgeWeight(int start_node, int end_node)
// order of the nodes in undirected complete case irelevant
{
    int high_node, low_node;
    // check which node has higher index to determine the corresponding edge weight
    if (start_node > end_node)
    {
        high_node = start_node;
        low_node = end_node;
    }
    else
    {
        high_node = end_node;
        low_node = start_node;
    }
    // calculate index..
    int start_idx = low_node*(num_nodes) - (int)(low_node*(low_node + 1)/2);
    // printf("Edge (%d, %d) \n", start_node, end_node);
    // printf("Number of nodes: %d\n", num_nodes);
    // printf("Result edge weight index: %d\n", start_idx + high_node - low_node - 1);
    // printf("Result edge weight: %f\n", edge_weights[start_idx + high_node - low_node - 1]);
    return edge_weights[start_idx + high_node - low_node - 1];
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