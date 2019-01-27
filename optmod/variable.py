import numpy as np
from . import coptmod
from .expression import Expression, ExpressionMatrix, make_Expression

class VariableScalar(Expression):

    def __init__(self, name='var', value=0.):
        
        Expression.__init__(self)

        self.name = name
        self.value = np.float64(value) if value is not None else 0.

    def __repr__(self):

        return self.name

    def __node__(self, prefix):

        return ('', id(self))

    def __manager_node_type__(self):

        return coptmod.NODE_TYPE_VARIABLE

    def __fill_manager__(self, manager):

        manager.add_node(self.__manager_node_type__(),
                         id(self),
                         self.value,
                         [])

    def __analyze__(self, G, prefix):

        G.add_node(self.__node__(prefix), item=self)

        return {'affine': True,
                'a': {self: 1.},
                'b': 0.}

    def get_derivative(self, var, G=None):

        if self is var:
            return make_Expression(1.)
        else:
            return make_Expression(0.)

    def get_variables(self):

        return set([self])
    
    def get_value(self):
        
        return self.value

    def is_variable(self):

        return True

    def set_value(self, val):

        self.value = val
        
class VariableMatrix(ExpressionMatrix):

    def __init__(self, name='var', value=None, shape=None):
        
        ExpressionMatrix.__init__(self)

        if shape is None and value is None:
            shape = (1,1)
        
        if value is None:
            value = np.zeros(shape, dtype=np.float64)
        value = np.asmatrix(value)

        if shape is None:
            shape = value.shape

        if len(shape) == 1:
            shape = shape[0],1

        if value.shape != shape:
            value = value.reshape(shape)
            
        self.shape = shape
        self.data = np.asmatrix([[VariableScalar(name=name+'[%d,%d]' %(i,j),
                                                 value=np.float64(value[i,j]))
                                  for j in range(shape[1])]
                                 for i in range(shape[0])],
                                dtype=object)

def Variable(name='var', value=None, shape=None):

    mat = False
    if shape is not None:
        mat = True

    if (value is not None) and (not np.isscalar(value)):
        mat = True

    if not mat:
        return VariableScalar(name=name, value=value)
    else:
        return VariableMatrix(name=name, value=value, shape=shape)
