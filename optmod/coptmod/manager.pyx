cimport manager

cdef class Manager:

    cdef manager.Manager* _ptr

    def __init__(self, max_nodes):

        pass

    def __cinit__(self, max_nodes):

        self._ptr = manager.MANAGER_new(max_nodes)

    def __dealloc__(self):

        manager.MANAGER_del(self._ptr)
        self._ptr = NULL

    def add_node(self, type, id, value, arg_ids):

        pass
        
    def show(self):

        manager.MANAGER_show(self._ptr)
