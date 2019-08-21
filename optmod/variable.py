import numpy as np
from . import coptmod
from itertools import count
from .expression import Expression, ExpressionMatrix, make_Expression

class VariableDict(dict):

    def __init__(self, keys, name='var', value=None, type='continuous'):

        if value is None:
            value = {}

        for key in keys:
            self[key] = VariableScalar(name=name+'_%s' %(key,),
                                       value=value[key] if (value is not None and key in value) else 0.,
                                       type=type)

class VariableScalar(Expression):

    _ids = count(0)

    def __init__(self, name='var', value=0., type='continuous'):

        if type not in ['integer', 'continuous']:
            raise ValueError('invalid variable type')

        if not np.isscalar(value):
            raise ValueError('value is not a scalar')
        
        Expression.__init__(self)

        self.name = name
        self.__value__ = np.float64(value) if value is not None else 0.
        self.type = type
        self.id = next(self._ids)
        
    def __repr__(self):

        return self.name

    def __evaluator_node_type__(self):

        return coptmod.NODE_TYPE_VARIABLE

    def __fill_evaluator__(self, evaluator):

        evaluator.add_node(self.__evaluator_node_type__(),
                           id(self),
                           self.__value__,
                           [])

    def __analyze__(self):

        return {'affine': True,
                'a': {self: 1.},
                'b': 0.}

    def get_derivatives(self, vars):

        return dict((var, make_Expression(1.) if var is self else make_Expression(0.))
                    for var in vars)

    def get_variables(self):

        return set([self])
    
    def is_variable(self):

        return True

    def is_continuous(self):

        return self.type == 'continuous'

    def is_integer(self):

        return self.type == 'integer'

    def set_value(self, val):

        self.__value__ = val
        
class VariableMatrix(ExpressionMatrix):

    def __init__(self, name='var', value=None, shape=None, type='continuous'):
        
        ExpressionMatrix.__init__(self)

        if shape is None and value is None:
            shape = (1,1)
        
        if value is None:
            value = np.zeros(shape, dtype=np.float64)
        value = np.asmatrix(value)

        if shape is None:
            shape = value.shape

        if len(shape) == 1:
            shape = shape[0], 1

        if value.shape != shape:
            value = value.reshape(shape)
            
        self.shape = shape
        self.data = np.asmatrix([[VariableScalar(name=name+'[%d,%d]' %(i,j),
                                                 value=np.float64(value[i,j]),
                                                 type=type)
                                  for j in range(shape[1])]
                                 for i in range(shape[0])],
                                dtype=object)

    def set_value(self, val):

        val = np.asmatrix(val)

        if val.shape != self.shape:
            raise ValueError('invalid shape of value')

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.data[i,j].set_value(val[i,j])

