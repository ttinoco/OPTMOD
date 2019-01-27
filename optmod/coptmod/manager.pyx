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

        cdef np.ndarray[long,mode='c'] x = np.array(arg_ids, dtype=long)
        manager.MANAGER_add_node(self._ptr, type, id, value, <long*>(x.data), x.size)
        
    def show(self):

        manager.MANAGER_show(self._ptr)

    property max_nodes:
        def __get__(self): return manager.MANAGER_get_max_nodes(self._ptr)

    property num_nodes:
        def __get__(self): return manager.MANAGER_get_num_nodes(self._ptr)

    
