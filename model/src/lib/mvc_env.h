#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"

class MvcEnv
{
public:
    MvcEnv();

    MvcEnv(double _norm, int _help_func, int _sign, int _fix_start_node);

    MvcEnv(std::shared_ptr<MvcEnv> mvc_env);

    ~MvcEnv();

    void s0(std::shared_ptr<Graph> _g);

    double step(int a);

    void stepWithoutReward(int a);

    int randomAction();

    bool isTerminal();
    
    double add_node(int new_node);

    double getReward();

    double getCurrentTourLength();

    double getTourDifference(int new_node);

    double getLastTourDifference();

    void printGraph();

    int help_func;

    double norm;

    int sign;

    int fix_start_node;

    std::shared_ptr<Graph> graph;

    std::vector< std::vector<int> > state_seq;

    std::vector<int> act_seq, state;

    std::vector<double> reward_seq, sum_rewards;

    int numCoveredEdges;

    std::set<int> covered_set;

    std::vector<int> avail_list;

    std::vector<int > node_degrees;

    int total_degrees;
};

#endif