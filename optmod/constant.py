import numpy as np
from . import utils
from . import coptmod
from .expression import Expression

class Constant(Expression):

    value = 0.

    def __init__(self, value):
        
        Expression.__init__(self)

        self.name = 'const'
        try:
            self.value = float(value)
        except:
            raise TypeError('invalid value')

    def __repr__(self):

        return utils.repr_number(self.value)

    def __analyze__(self, G, prefix):
        
        G.add_node(self.__node__(prefix), item=self)

        return {'affine': True,
                'a': {}, 
                'b': self.value}

    def __evaluator_node_type__(self):

        return coptmod.NODE_TYPE_CONSTANT

    def __fill_evaluator__(self, evaluator):

        evaluator.add_node(self.__evaluator_node_type__(),
                           id(self),
                           self.value,
                           [])

    def get_value(self):

        return self.value

    def is_zero(self):

        return self.value == 0.

    def is_one(self):

        return self.value == 1.

    def is_constant(self, val=None):

        if val is None:
            return True
        else:
            return self.value == val
