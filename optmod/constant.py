import numpy as np
from . import utils
from . import coptmod
from .expression import Expression

class Constant(Expression):

    def __init__(self, value):
        
        Expression.__init__(self)

        self.name = 'const'
        try:
            self.__value__ = float(value)
        except:
            raise TypeError('invalid value')

    def __repr__(self):

        return utils.repr_number(self.__value__)

    def __analyze__(self):
        
        return {'affine': True,
                'a': {}, 
                'b': self.__value__}

    def __evaluator_node_type__(self):

        return coptmod.NODE_TYPE_CONSTANT

    def __fill_evaluator__(self, evaluator):

        evaluator.add_node(self.__evaluator_node_type__(),
                           id(self),
                           self.__value__,
                           [])

    def is_zero(self):

        return self.__value__ == 0.

    def is_one(self):

        return self.__value__ == 1.

    def is_constant(self, val=None):

        if val is None:
            return True
        else:
            return self.__value__ == val
