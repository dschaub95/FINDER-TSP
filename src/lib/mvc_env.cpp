#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include <queue>
#include <stack>
#include <cstdio>


MvcEnv::MvcEnv(double _norm)
{
norm = _norm;
graph = nullptr;
numCoveredEdges = 0;
CcNum = 1.0;
state_seq.clear();
act_seq.clear();
action_list.clear();
reward_seq.clear();
sum_rewards.clear();
covered_set.clear();
avail_list.clear();
}

MvcEnv::~MvcEnv()
{
    norm = 0;
    graph = nullptr;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0;
    CcNum = 1.0;
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
    state_seq.push_back(action_list);
    act_seq.push_back(a);
    covered_set.insert(a);
    action_list.push_back(a);


    // for (auto neigh : graph->adj_list[a])
    //     if (covered_set.count(neigh) == 0)
    //         numCoveredEdges++;
    
    double r_t = getReward();
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
    covered_set.insert(a);
    action_list.push_back(a);
    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;
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
            // bool useful = false;
            // for (auto neigh : graph->adj_list[i])
            //     if (covered_set.count(neigh) == 0)
            //     {
            //         useful = true;
            //         break;
            //     }
            // if (useful)
            avail_list.push_back(i);
        }

    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    // printf("Availabe nodes %d\n", avail_list.size());
    return avail_list[idx];
}


bool MvcEnv::isTerminal()
{
    assert(graph);
    // printf("Size Action list: %d\n", (int)action_list.size());
    // printf("Graph Nodes: %d\n", graph->num_nodes);
    // printf("Bool: %d\n", ((int)action_list.size() == (int)graph->num_nodes));
    return (int)action_list.size() == (int)graph->num_nodes - 1;
}


double MvcEnv::getReward()
{
    double orig_node_num = (double) graph->num_nodes;
    // TSP reward
    double reward = 0;
    if ((int)action_list.size() == graph->num_nodes - 1)
    {
        reward = -(double)getLastTourDifference();
    }
    else
    {
        reward = -(double)getTourDifference();
    }
    double norm_reward = reward/orig_node_num;
    // printf("reward: %f \n", reward);
    // printf("number of nodes: %f \n", orig_node_num);
    // printf("self calculated reward: %f \n", (orig_node_num-covered_set.size())*(orig_node_num-covered_set.size()-1)/(orig_node_num*(orig_node_num-1)));
    return norm_reward;
}

double MvcEnv::getTourDifference()
{
    double previousLength = 0.0;
    double currentLength = 0.0;
    
    if ((int)action_list.size() > 2)
    {
        // calc length of the last part of the current tour, ranging from second last node to the start node
        int idx_1 = getEdgeWeightIndex(action_list[action_list.size()-1], action_list[0]);
        int idx_2 = getEdgeWeightIndex(action_list[action_list.size()-2], action_list[action_list.size()-1]);
        currentLength += graph->edge_weights[idx_1];
        currentLength += graph->edge_weights[idx_2];

        // calc length of the last part of the previous tour
        int idx = getEdgeWeightIndex(action_list[action_list.size()-2], action_list[0]);
        previousLength += graph->edge_weights[idx];
    }
    else if ((int)action_list.size() == 2)
    {
        // previous tour contains only one node
        int idx_1 = getEdgeWeightIndex(action_list[action_list.size()-1], action_list[0]);
        int idx_2 = getEdgeWeightIndex(action_list[action_list.size()-2], action_list[action_list.size()-1]);
        currentLength += graph->edge_weights[idx_1];
        currentLength += graph->edge_weights[idx_2];
    }
    return currentLength - previousLength;
}

double MvcEnv::getLastTourDifference()
{
    double previousLength = 0.0;
    double currentLength = 0.0;
    assert((int)action_list.size() > 2);
    
    // get name of the last node
    int last_node = randomAction();
    
    // calc length of the last part of the current tour, ranging from third last node to the start node
    int idx_1 = getEdgeWeightIndex(action_list[action_list.size()-2], action_list[action_list.size()-1]);
    int idx_2 = getEdgeWeightIndex(action_list[action_list.size()-1], last_node);
    int idx_3 = getEdgeWeightIndex(last_node, action_list[0]);
    currentLength += graph->edge_weights[idx_1];
    currentLength += graph->edge_weights[idx_2];
    currentLength += graph->edge_weights[idx_3];

    // calc length of the last part of the previous tour
    int idx = getEdgeWeightIndex(action_list[action_list.size()-2], action_list[0]);
    previousLength += graph->edge_weights[idx];
  
    return currentLength - previousLength;
}

double MvcEnv::getCurrentTourLength()
{
    // check that graph object is non emtpy
    assert(graph);
    double tourLength = 0.0;
    // make sure there are at least two nodes in the current tour
    if ((int)action_list.size() <= 1)
    {
        return tourLength;
    }
    // more than two nodes in current tour
    
    for (int i = 0; i < (int)action_list.size(); ++i)
    {   
        if (i == 0)
        {
            continue;
        }
        int idx = getEdgeWeightIndex(action_list[i-1], action_list[i]);
        tourLength += graph->edge_weights[idx];
    }
    // add path from last to first node
    int last_idx = getEdgeWeightIndex(action_list[action_list.size()-1], action_list[0]);
    tourLength += graph->edge_weights[last_idx];
    return tourLength;
}

int MvcEnv::getEdgeWeightIndex(int start_node, int end_node)
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
    int start_idx = low_node*(graph->num_nodes) - (int)(low_node*(low_node + 1)/2);
    // printf("Edge (%d, %d) \n", start_node, end_node);
    // printf("Number of nodes: %d\n", graph->num_nodes);
    // printf("Result edge weight index: %d\n", start_idx + high_node - low_node - 1);
    return start_idx + high_node - low_node - 1;
}


double MvcEnv::getRemainingCNDScore()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }

    std::set<int> lccIDs;
    for(int i =0;i< graph->num_nodes; i++){
        lccIDs.insert(disjoint_Set.unionSet[i]);
    }

    double CCDScore = 0.0;
    for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
    {
        double num_nodes = (double) disjoint_Set.getRank(*it);
        CCDScore += (double) num_nodes * (num_nodes-1) / 2;
    }
    //printf("CCDscore: %f \n", CCDScore);
    return CCDScore;
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
