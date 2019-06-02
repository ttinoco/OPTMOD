import numpy as np
from . import coptmod
from functools import reduce
from collections import OrderedDict

class Expression(object):

    name = ''
    __array_priority__ = 1000.
    __value__ = 0.

    def __init__(self):

        pass

    def __repr__(self):
        
        raise NotImplementedError

    def __hash__(self):

        return id(self)

    def __neg__(self):
        
        return -1.*self
    
    def __add__(self, x):
                
        # Arrray
        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__add__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r

        # Expresiosn matrix
        if isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return x
            return x.__add__(self)

        # Other
        x = make_Expression(x)
        if self.is_zero():
            return x
        if x.is_zero():
            return self
        if self.is_constant() and x.is_constant():
            return make_Expression(self.__value__ + x.__value__)

        # Flat args
        args = []
        for a in [self, x]:
            if isinstance(a, add):
                args.extend(a.arguments)
            else:
                args.append(a)

        # Return
        return add(args)

    def __radd__(self, x):

        return self.__add__(x)

    def __sub__(self, x):

        # Array
        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(-x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__rsub__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r

        # Expression matrix
        if isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return -x
            return x.__rsub__(self)

        # Other
        x = make_Expression(x)
        return self.__add__(x.__mul__(-1.))

    def __rsub__(self, x):

        # Array
        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__sub__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r

        # Expression matrix
        if isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return x
            return x.__sub__(self)
                
        # Other
        x = make_Expression(x)
        return x.__add__(self.__mul__(-1.))

    def __mul__(self, x):

        # Array
        if isinstance(x, np.ndarray):
            if self.is_one():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__mul__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r

        # Expression matrix
        if isinstance(x, ExpressionMatrix):
            if self.is_one():
                return x
            return x.__mul__(self)

        # Other
        x = make_Expression(x)
        if self.is_one():
            return x
        if x.is_one():
            return self
        if self.is_zero() or x.is_zero():
            return make_Expression(0.)
        if self.is_constant() and x.is_constant():
            return make_Expression(self.__value__*x.__value__)
        if self.is_constant() and isinstance(x, add):
            return add([self.__mul__(arg) for arg in x.arguments])
        if x.is_constant() and isinstance(self, add):
            return add([x.__mul__(arg) for arg in self.arguments])
        if self.is_constant() and isinstance(x, multiply):
            if x.arguments[0].is_constant():
                return x.arguments[1].__mul__(self.__value__*x.arguments[0].__value__)
            if x.arguments[1].is_constant():
                return x.arguments[0].__mul__(self.__value__*x.arguments[1].__value__)
            return multiply([self, x])
        if x.is_constant() and isinstance(self, multiply):
            if self.arguments[0].is_constant():
                return self.arguments[1].__mul__(x.__value__*self.arguments[0].__value__)
            if self.arguments[1].is_constant():
                return self.arguments[0].__mul__(x.__value__*self.arguments[1].__value__)
            return multiply([self, x])
        return multiply([self, x])

    def __rmul__(self, x):

        return self.__mul__(x)

    def __eq__(self, x):
        
        return self.__cmp_util__('==', x)

    def __le__(self, x):

        return self.__cmp_util__('<=', x)

    def __ge__(self, x):

        return self.__cmp_util__('>=', x)

    def __cmp_util__(self, op, x):
                
        if isinstance(x, np.ndarray):
            return ConstraintArray(np.vectorize(self.__cmp_util__)(op, x))
    
        elif isinstance(x, ExpressionMatrix):
            return ConstraintArray(np.vectorize(self.__cmp_util__)(op, x.data))
                
        else:
            return Constraint(self, op, x)

    def __node__(self, prefix):
        
        return (prefix, id(self))

    def __analyze__(self, G, prefix):

        return {'affine': False,
                'a': {},
                'b': np.NaN}
    
    def __get_std_components__(self):

        phi = self
        gphi_list = []
        Hphi_list = []

        prop = self.__analyze__()

        # Affine
        if prop['affine']:
            for var, value in prop['a'].items():
                gphi_list.append((var, make_Expression(value)))

        # Not affine
        else:
            vars_set = set(prop['a'].keys())
            vars_list = list(vars_set)
            derivs = self.get_derivatives(vars_set)
            for i, var1 in enumerate(vars_list):
                d = derivs[var1]
                gphi_list.append((var1, d))
                dvars_list = vars_list[i:]
                dvars_set = set(dvars_list)
                dderivs = d.get_derivatives(dvars_set) 
                for var2 in dvars_list:
                    dd = dderivs[var2]
                    if not dd.is_constant(0.):
                        Hphi_list.append((var1, var2, dd))

        # Return
        return {'phi': phi,
                'gphi_list': gphi_list,
                'Hphi_list': Hphi_list,
                'phi_prop': prop}

    def __evaluator_node_type__(self):

        return NotImplemented

    def __fill_evaluator__(self, evaluator):

        raise NotImplementedError

    def __set_value__(self):

        pass

    def get_derivative(self, var):

        return self.get_derivatives(set([var]))[var]

    def get_derivatives(self, vars):

        return dict((var, make_Expression(0.)) for var in vars)

    def get_value(self):

        return self.__value__

    def get_variables(self):
    
        return set()

    def get_fast_evaluator(self, variables):

        assert(isinstance(variables, list))

        e = coptmod.Evaluator(len(variables),
                              1,
                              scalar_output=True)
        self.__fill_evaluator__(e)
        for i, var in enumerate(variables):
            e.set_input_var(i, id(var))
        e.set_output_node(0, id(self))
        
        return e

    def is_zero(self):

        return False

    def is_one(self):

        return False

    def is_constant(self, val=None):

        return False

    def is_variable(self):

        return False

    def is_function(self):

        return False

def make_Expression(obj):
    
    if isinstance(obj, Expression):
        return obj
    else:
        return Constant(obj)
        
class ExpressionMatrix(object):

    data = None
    shape = None
    __array_priority__ = 10000.
    
    def __init__(self, obj=None):
        
        if isinstance(obj, ExpressionMatrix):
            self.data = obj.data
            self.shape = obj.shape
            
        elif isinstance(obj, Expression):
            self.data = np.asmatrix(obj)
            self.shape = self.data.shape
            
        elif obj is not None:
            obj = np.asmatrix(obj)
            if obj.size:
                self.data = np.vectorize(make_Expression)(obj)
            else:
                self.data = obj
            self.shape = obj.shape

    def __repr__(self):

        s = '['
        for i in range(self.shape[0]):
            s += '[ ' if i == 0 else ' [ '
            for j in range(self.shape[1]):
                if j < self.shape[1]-1:
                    s += '%s, ' %str(self[i,j])
                else:
                    s += '%s' %str(self[i,j])
            s += ' ]'
            if i < self.shape[0]-1:
                s += ',\n'
        s += ']\n'
        return s

    def __getitem__(self, key):

        m = np.asmatrix(self.data)[key]
        if isinstance(m, Expression):
            return m
        else:
            return ExpressionMatrix(m)

    def __neg__(self):
        
        return ExpressionMatrix(np.vectorize(lambda x: -x)(self.data))

    def __add__(self, x):

        if isinstance(x, ExpressionMatrix):
            return ExpressionMatrix(self.data.__add__(x.data))

        else:
            return ExpressionMatrix(self.data.__add__(np.asmatrix(x)))
        
    def __radd__(self, x):
        
        return self.__add__(x)

    def __sub__(self, x):

        if isinstance(x, ExpressionMatrix):
            return ExpressionMatrix(self.data.__sub__(x.data))

        else:
            return ExpressionMatrix(self.data.__sub__(np.asmatrix(x)))

    def __rsub__(self, x):

        if isinstance(x, ExpressionMatrix):
            return ExpressionMatrix(self.data.__rsub__(x.data))

        else:
            return ExpressionMatrix(self.data.__rsub__(np.asmatrix(x)))

    def __mul__(self, x):

        if isinstance(x, ExpressionMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            return NotImplemented

        else:
            return ExpressionMatrix(np.asarray(self.data)*x)
    
    def __rmul__(self, x):

        if isinstance(x, ExpressionMatrix):
            return NotImplemented

        elif isinstance(x, np.ndarray):
            return NotImplemented

        else:
            return self.__mul__(x)

    def __eq__(self, x):

        return self.__cmp_util__('==', x)

    def __le__(self, x):

        return self.__cmp_util__('<=', x)

    def __ge__(self, x):

        return self.__cmp_util__('>=', x)

    def __cmp_util__(self, op, x):

        if isinstance(x, ExpressionMatrix):
            return ConstraintArray(np.vectorize(lambda a,b: a.__cmp_util__(op, b))(self.data, x.data))

        else:
            return ConstraintArray(np.vectorize(lambda a,b: a.__cmp_util__(op, b))(self.data, np.asarray(x)))

    def get_data(self):

        return self.data
        
    def get_value(self):

        if self.data.size:
            return np.vectorize(lambda x: x.get_value())(self.data)
        else:
            return np.asmatrix(self.data, dtype=np.float64)

    def get_variables(self):

        return reduce(lambda x,y: x.union(y),
                      map(lambda arg: arg.get_variables(),
                      np.asarray(self.data).flatten().tolist()),
                      set())

    def get_fast_evaluator(self, variables):

        assert(isinstance(variables, list))

        e = coptmod.Evaluator(len(variables),
                              self.shape[0]*self.shape[1],
                              shape=self.shape,
                              scalar_output=False)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x = self.data[i,j]
                x.__fill_evaluator__(e)
                e.set_output_node(i*self.shape[1]+j, id(x))
        for i, var in enumerate(variables):
            e.set_input_var(i, id(var))
        return e
              
class SparseExpressionMatrix:

    pass

# Circular imports
from .constant import Constant
from .function import add, multiply
from .constraint import Constraint, ConstraintArray
        
