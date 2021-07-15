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



Utils::Utils(){}

double Utils::getTourLength(std::shared_ptr<Graph> graph, std::vector<int> solution)
{
    double orig_node_num = (double) graph->num_nodes;
    double tourLength = 0.0;
    int sol_size = solution.size();
    // make sure there are at least two nodes in the current tour
    for (int i = 0; i < sol_size; ++i)
    {   
        if (i == 0)
        {
            continue;
        }
        tourLength += graph->EdgeWeight[solution[i-1]][solution[i]];
    }
    // add path from last to first node also considering the last node determined by all the others, which is not specifically selected
    tourLength += graph->EdgeWeight[solution[sol_size-1]][solution[0]];
    return tourLength;
}