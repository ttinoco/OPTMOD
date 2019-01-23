cdef extern from "node.h":

    ctypedef enum NodeType:
        NODE_TYPE_UNKNOWN,
        NODE_TYPE_CONSTANT,
        NODE_TYPE_VARIABLE,
        NODE_TYPE_ADD,
        NODE_TYPE_SUBTRACT,
        NODE_TYPE_NEGATE,
        NODE_TYPE_MULTIPLY,
        NODE_TYPE_SIN,
        NODE_TYPE_COS
