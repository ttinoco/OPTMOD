
cdef extern from "manager.h":

    ctypedef struct Manager
    ctypedef NodeType

    void MANAGER_add_node(Manager* m, NodeType type, long id, double value, long* arg_ids, int num_args)
    Manager* MANAGER_new(int max_nodes)
    void MANAGER_del(Manager* m)
    void MANAGER_show(Manager* m)
