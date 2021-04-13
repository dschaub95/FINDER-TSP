#include "utils.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include "stdio.h"
#include <queue>
#include <stack>
#include <numeric>



Utils::Utils()
{
    MaxWccSzList.clear();
}


std::vector<int> Utils::reInsert(std::shared_ptr<Graph> graph,std::vector<int> solution,const std::vector<int> allVex,int decreaseStrategyID,int reinsertEachStep){
    std::shared_ptr<decreaseComponentStrategy> decreaseStrategy;

    switch(decreaseStrategyID){
        case 1:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentCount());
            break;
        }
        case 2:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank());
            break;
        }
        case 3:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentMultiple());
            break;
        }
        default:
        {
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank());
            break;
        }
    }

        return reInsert_inner(solution,graph,allVex,decreaseStrategy,reinsertEachStep);

}


std::vector<int> Utils::reInsert_inner(const std::vector<int> &beforeOutput, std::shared_ptr<Graph> &graph, const std::vector<int> &allVex, std::shared_ptr<decreaseComponentStrategy> &decreaseStrategy,int reinsertEachStep)
{
    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());

    std::vector<std::vector<int> > currentAdjListGraph;

    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;

    std::vector<bool> currentAllVex(graph->num_nodes, false);

    for (int eachV : allVex)
    {
        currentAllVex[eachV] = true;
    }

    std::unordered_set<int> leftOutput(beforeOutput.begin(), beforeOutput.end());

    std::vector<int> finalOutput;

    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    while (leftOutput.size() != 0)
    {
//        printf (" reInsertCount:%d\n", leftOutput.size());

        std::vector<std::pair<long long, int> >  batchList;

        for (int eachNode : leftOutput)
        {
            //min is better
            long long decreaseValue = decreaseStrategy->decreaseComponentNumIfAddNode(backupCompletedAdjListGraph, currentAllVex, disjoint_Set, eachNode);
            batchList.push_back(make_pair(decreaseValue, eachNode));
        }


        if (reinsertEachStep >= (int)batchList.size())
        {
            reinsertEachStep = (int)batchList.size();
        }
        else
        {
            std::nth_element(batchList.begin(), batchList.begin() + reinsertEachStep, batchList.end());
        }

        for (int i = 0; i < reinsertEachStep; i++)
        {
            finalOutput.push_back(batchList[i].second);
            leftOutput.erase(batchList[i].second);
            graphutil->recoverAddNode(backupCompletedAdjListGraph, currentAllVex, currentAdjListGraph, batchList[i].second, disjoint_Set);
        }

    }

    std::reverse(finalOutput.begin(), finalOutput.end());

    return finalOutput;
}



double Utils::getRobustness(std::shared_ptr<Graph> graph, std::vector<int> solution)
{
    assert(graph);
    MaxWccSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
    std::vector<std::vector<int>> current_adj_list;
    std::shared_ptr<GraphUtil> graphutil = std::shared_ptr<GraphUtil>(new GraphUtil());
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    std::vector<bool> backupAllVex(graph->num_nodes, false); // initialized as false, length of num_nodes
    double totalMaxNum = 0.0;
    double temp = 0.0;
    double norm = (double)graph->num_nodes * (double)(graph->num_nodes-1) / 2.0;
    //printf("Norm:%.8f\n", norm);
    for (std::vector<int>::reverse_iterator it = solution.rbegin(); it != solution.rend(); ++it)
    {
        int Node =(*it);
        graphutil->recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        // calculate the remaining components and its nodes inside
        //std::set<int> lccIDs;
        //for(int i =0;i< graph->num_nodes; i++){
        //    lccIDs.insert(disjoint_Set.unionSet[i]);
        //}
        //double CCDScore = 0.0;
        //for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
       // {
        //    double num_nodes = (double) disjoint_Set.getRank(*it);
        //    CCDScore += (double) num_nodes * (num_nodes-1) / 2;
       // }

        totalMaxNum += disjoint_Set.CCDScore / norm;
        MaxWccSzList.push_back(disjoint_Set.CCDScore / norm);
        temp = disjoint_Set.CCDScore / norm;
    }

    totalMaxNum = totalMaxNum - temp;
    std::reverse(MaxWccSzList.begin(), MaxWccSzList.end());

    return (double)totalMaxNum / (double)graph->num_nodes;
}

double Utils::getTourLength(std::shared_ptr<Graph> graph, std::vector<int> solution)
{
    // int solSum = std::accumulate(solution.begin(), solution.end(), decltype(solution)::value_type(0));
    // printf("Solsum: %d\n", solSum);
    // int missingNode = (graph->num_nodes + 1)*graph->num_nodes/2 - solSum;
    // printf("Missing node: %d\n", missingNode);
    
    double orig_node_num = (double) graph->num_nodes;
    double tourLength = 0.0;
    // make sure there are at least two nodes in the current tour
    for (int i = 0; i < (int)solution.size(); ++i)
    {   
        if (i == 0)
        {
            continue;
        }
        tourLength += graph->getEdgeWeight(solution[i-1], solution[i]);
    }
    // add path from last to first node also considering the last node determined by all the others, which is not specifically selected
    tourLength += graph->getEdgeWeight(solution[solution.size()-1], solution[0]);
    return tourLength / orig_node_num;
}

int Utils::getMxWccSz(std::shared_ptr<Graph> graph)
// calculates 
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    for (int i = 0; i < (int)graph->adj_list.size(); i++)
    {
        for (int j = 0; j < (int)graph->adj_list[i].size(); j++)
        {
            disjoint_Set.merge(i, graph->adj_list[i][j]);
        }
    }
    return disjoint_Set.maxRankCount;
}