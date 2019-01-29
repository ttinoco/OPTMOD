import numpy as np
from .expression import Expression, ExpressionMatrix

def repr_number(x):

    return '%.2e' %x

def sum(x, axis=None):

    if isinstance(x, Expression):
        return np.sum(x, axis=axis)
    else:
        s = np.sum(ExpressionMatrix(x).data, axis=axis)
        if isinstance(s, np.ndarray):
            return ExpressionMatrix(s)
        else:
            return s    

    
