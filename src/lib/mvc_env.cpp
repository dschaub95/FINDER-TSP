#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include <queue>
#include <stack>
#include <cstdio>

MvcEnv::MvcEnv()
{
    norm = 0;
    graph = nullptr;
    sign = 0;
    help_func = 0;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    state.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

MvcEnv::MvcEnv(double _norm, int _help_func, int _sign)
{
    norm = _norm;
    graph = nullptr;
    sign = _sign;
    help_func = _help_func;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    state.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

MvcEnv::MvcEnv(std::shared_ptr<MvcEnv> mvc_env)
{

}

MvcEnv::~MvcEnv()
{
    norm = 0;
    help_func = 0;
    sign = 0;
    graph = nullptr;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    state.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    // help_func = _help_func;
    // make norm depend on specific graph if selected
    if ((int)norm == -1) { norm = _g->num_nodes; }
    covered_set.clear();
    state.clear();
    state.push_back(0);
    covered_set.insert(0);
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MvcEnv::step(int a)
// takes node as action
{
    assert(graph);
    // make sure node was not yet visited
    assert(covered_set.count(a) == 0);
    // assert(a > 0 && a < graph->num_nodes);
    state_seq.push_back(state);
    act_seq.push_back(a);
    double r_t;
    if (help_func == 1)
    {
        // adds node at best postition and also updates covered set and action list
        r_t = add_node(a);
    }
    else
    {
        // printf("Calculating tour difference without helper function..\n");
        r_t = getTourDifference(a);
        state.push_back(a);
        covered_set.insert(a);
        // state.push_back(a);
        // covered_set.insert(a);
        // r_t = getReward();
    }
    reward_seq.push_back(r_t);
    // initialize sum of rewards with the standard reward values
    sum_rewards.push_back(r_t);  

    return r_t;
}


void MvcEnv::stepWithoutReward(int a)
// used to generate solutions after training, where the the reward is not relevant
{
    assert(graph);
    assert(covered_set.count(a) == 0);
    act_seq.push_back(a);
    if (help_func == 1)
    {
        double tmp_reward = add_node(a);
    }
    else 
    {
        state.push_back(a);
        covered_set.insert(a);
    }
    for (auto neigh : graph->adj_list[a])
    {
        if (covered_set.count(neigh) == 0)
        {
            numCoveredEdges++;
        }
    }
        
            
}

// random
int MvcEnv::randomAction()
// return random node that is adjacent to one of the currently selected ones
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
        {
            avail_list.push_back(i);
        }

    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool MvcEnv::isTerminal()
{
    assert(graph);
    // printf("Size Action list: %d\n", (int)state.size());
    // printf("Graph Nodes: %d\n", graph->num_nodes);
    // printf("Bool: %d\n", ((int)state.size() == (int)graph->num_nodes));
    // return (int)state.size() == (int)graph->num_nodes - 1;
    return (int)state.size() == (int)graph->num_nodes;
}

double MvcEnv::add_node(int new_node)
{
    double cur_dist = 10000000.0;
    int pos = -1;
    for (size_t i = 0; i < state.size(); ++i)
    {
        int adj;
        if (i + 1 == state.size())
        {
            adj = state[0];
        }    
        else
        {
            adj = state[i + 1]; 
        }
        double cost = graph->EdgeWeight[new_node][state[i]]
                      + graph->EdgeWeight[new_node][adj]
                      - graph->EdgeWeight[state[i]][adj];
        if (cost < cur_dist)
        {
            cur_dist = cost;
            pos = i;
        }
    }
    assert(pos >= 0);
    assert(cur_dist >= -1e-8);
    state.insert(state.begin() + pos + 1, new_node);
    covered_set.insert(new_node);
    return sign * cur_dist / norm;  	
}


double MvcEnv::getTourDifference(int new_node)
{
    assert(graph);
    
    int adj = state[0];
    int last_node = state[state.size()-1];
    double cost = graph->EdgeWeight[last_node][new_node]
                  + graph->EdgeWeight[new_node][adj]
                  - graph->EdgeWeight[last_node][adj]; 
    return sign * cost / norm;
}

void MvcEnv::printGraph()
{
    printf("edge_list:\n");
    printf("[");
    for (int i = 0; i < (int)graph->edge_list.size();i++)
    {
    printf("[%d,%d],",graph->edge_list[i].first,graph->edge_list[i].second);
    }
    printf("]\n");


    printf("covered_set:\n");

    std::set<int>::iterator it;
    printf("[");
    for (it=covered_set.begin();it!=covered_set.end();it++)
    {
        printf("%d,",*it);
    }
    printf("]\n");
}

double MvcEnv::getReward()
{
    double orig_node_num = (double) graph->num_nodes;
    // TSP reward
    double reward = 0;
    // the last set of available nbodes consists of two nodes, selecting one determines the entire tour length --> no further action needed
    if ((int)state.size() == graph->num_nodes - 1)
    {
        reward = -(double)getLastTourDifference();
    }
    else
    {
        reward = -(double)getTourDifference(0);
    }
    double norm_reward = reward/orig_node_num;
    // printf("reward: %f \n", reward);
    // printf("number of nodes: %f \n", orig_node_num);
    // printf("self calculated reward: %f \n", (orig_node_num-covered_set.size())*(orig_node_num-covered_set.size()-1)/(orig_node_num*(orig_node_num-1)));
    return norm_reward;
}

double MvcEnv::getLastTourDifference()
{
    double previousLength = 0.0;
    double currentLength = 0.0;
    assert((int)state.size() > 2);
    
    // get name of the last remaining node
    int last_node = randomAction();
    int size = state.size();
    // calc length of the last part of the current tour, ranging from third last node to the start node
    currentLength += graph->EdgeWeight[state[size-2]][state[size-1]];
    currentLength += graph->EdgeWeight[state[size-1]][last_node];
    currentLength += graph->EdgeWeight[last_node][state[0]];

    // calc length of the last part of the previous tour
    previousLength += graph->EdgeWeight[state[size-2]][state[0]];
  
    return currentLength - previousLength;
}