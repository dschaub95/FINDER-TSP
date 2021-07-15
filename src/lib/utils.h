#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"
#include "graph_utils.h"
#include "decrease_strategy.cpp"

class Utils
{
public:
    Utils();

    double getTourLength(std::shared_ptr<Graph> graph, std::vector<int> solution);

};

#endif