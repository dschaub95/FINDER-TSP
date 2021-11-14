#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, 
          double** _EdgeWeight, double** _edge_probs, double** _node_feats, const double _NN_ratio);
    ~Graph();
    void SparsifyWithKNN();
    void SparsifyWithProbabilities(const int* edges_from, const int* edges_to);
    int num_nodes;
    int num_edges;
    double NN_ratio;
    std::vector< std::vector< int > > adj_list;
    std::vector< std::pair< int, int > > edge_list;
    std::vector< std::vector< double > > EdgeWeight;
    std::vector< std::vector< double > > edge_probs;
    std::vector< std::vector< double > > node_feats;
};

class GSet
{
public:
    GSet();
    ~GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    std::shared_ptr<Graph> Sample();
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif