cdef extern from "node.h":

    cdef int NODE_TYPE_UNKNOWN
    cdef int NODE_TYPE_CONSTANT
    cdef int NODE_TYPE_VARIABLE
    cdef int NODE_TYPE_ADD
    cdef int NODE_TYPE_SUBTRACT
    cdef int NODE_TYPE_NEGATE
    cdef int NODE_TYPE_MULTIPLY
    cdef int NODE_TYPE_SIN
    cdef int NODE_TYPE_COS
