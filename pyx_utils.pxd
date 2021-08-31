
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/utils.h":
    cdef cppclass Utils:
        Utils()
        double getTourLength(shared_ptr[Graph] graph,vector[int] solution)except+

