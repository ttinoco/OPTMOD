
cdef extern from "manager.h":

    ctypedef struct Manager
    Manager* MANAGER_new(int max_nodes)
    void MANAGER_del(Manager* m)
