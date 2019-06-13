import sys
cimport evaluator

cdef class Evaluator:

    cdef evaluator.Evaluator* _ptr
    cdef tuple shape
    cdef bint scalar_output

    def __init__(self, num_inputs, num_outputs, shape=None, scalar_output=False):

        pass

    def __cinit__(self, num_inputs, num_outputs, shape=None, scalar_output=False):

        if shape is None:
            shape = (1, num_outputs)
        
        try:
            if len(shape) != 2:
                raise ValueError('invalid shape')
        except Exception:
            raise ValueError('invalid shape')

        if shape[0]*shape[1] != num_outputs:
            raise ValueError('invalid shape')

        self._ptr = evaluator.EVALUATOR_new(num_inputs, num_outputs)

        self.shape = shape
        self.scalar_output = scalar_output

    def __dealloc__(self):

        evaluator.EVALUATOR_del(self._ptr)
        self._ptr = NULL

    def add_node(self, type, id, value, arg_ids):
        
        cdef np.ndarray[long, mode='c'] x
        if 'win32' in sys.platform.lower():
            x = np.array(arg_ids, dtype=np.int32)
        else:
            x = np.array(arg_ids, dtype=long)
        evaluator.EVALUATOR_add_node(self._ptr, type, id, value, <long*>(x.data), x.size)

    def get_value(self):
        
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp>(self.shape[0])
        shape[1] = <np.npy_intp>(self.shape[1])
        
        arr = np.PyArray_SimpleNewFromData(2,
                                           shape,
                                           np.NPY_DOUBLE,
                                           evaluator.EVALUATOR_get_values(self._ptr))
        
        PyArray_CLEARFLAGS(arr, np.NPY_OWNDATA)
        if arr.shape == (1,1) and self.scalar_output:
            return arr[0,0]
        else:
            return np.asmatrix(arr)

    def eval(self, var_values):

        cdef np.ndarray[double, mode='c'] x = np.array(var_values, dtype=float)

        assert(x.ndim == 1)
        assert(x.size == self.num_inputs)
        
        evaluator.EVALUATOR_eval(self._ptr, <double*>(x.data))

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

    property shape:
        def __get__(self): return self.shape

    property scalar_output:
        def __get__(self): return self.scalar_output
    
