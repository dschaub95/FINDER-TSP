﻿
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/mvc_env.h":
    cdef cppclass MvcEnv:
        MvcEnv()
        MvcEnv(double _norm, int _help_func, int _sign, int _fix_start_node)
        MvcEnv(shared_ptr[MvcEnv] mvc_env)
        void s0(shared_ptr[Graph] _g)except+
        double step(int a)except+
        void stepWithoutReward(int a)except+
        int randomAction()except+
        bool isTerminal()except+
        double add_node(int new_node)except+
        double getReward()except+
        double getLastTourDifference()except+
        double getTourDifference(int new_node)except+
        double norm
        int sign
        int help_func
        int fix_start_node
        shared_ptr[Graph] graph
        vector[vector[int]]  state_seq
        vector[int] act_seq
        vector[int] state
        vector[double] reward_seq
        vector[double] sum_rewards
        int numCoveredEdges
        set[int] covered_set
        vector[int] avail_list
