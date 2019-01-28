cimport evaluator

cdef class Evaluator:

    cdef evaluator.Evaluator* _ptr

    def __init__(self, num_inputs, num_outputs):

        pass

    def __cinit__(self, num_inputs, num_outputs):

        self._ptr = evaluator.EVALUATOR_new(num_inputs, num_outputs)

    def __dealloc__(self):

        evaluator.EVALUATOR_del(self._ptr)
        self._ptr = NULL

    def add_node(self, type, id, value, arg_ids):

        cdef np.ndarray[long,mode='c'] x = np.array(arg_ids, dtype=long)
        evaluator.EVALUATOR_add_node(self._ptr, type, id, value, <long*>(x.data), x.size)

    def get_value(self):

        pass

    def eval(self, x):

        pass

    def set_output_node(self, i, id):

        evaluator.EVALUATOR_set_output_node(self._ptr, i, id)

    def set_input_var(self, i, id):

        evaluator.EVALUATOR_set_input_var(self._ptr, i, id)
        
    def show(self):

        evaluator.EVALUATOR_show(self._ptr)

    property max_nodes:
        def __get__(self): return evaluator.EVALUATOR_get_max_nodes(self._ptr)

    property num_nodes:
        def __get__(self): return evaluator.EVALUATOR_get_num_nodes(self._ptr)

    property num_inputs:
        def __get__(self): return evaluator.EVALUATOR_get_num_inputs(self._ptr)

    property num_outputs:
        def __get__(self): return evaluator.EVALUATOR_get_num_outputs(self._ptr)

    
