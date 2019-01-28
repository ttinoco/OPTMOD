import numpy as np
import networkx as nx
from . import coptmod

class Expression(object):

    name = ''
    __array_priority__ = 1000.

    def __init__(self):

        pass

    def __repr__(self):
        
        raise NotImplementedError

    def __hash__(self):

        return id(self)

    def __neg__(self):

        from .function import negate

        return negate(self)
    
    def __add__(self, x):
        
        from .function import add
        
        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__add__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r
    
        elif isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return x
            return x.__add__(self)
                
        else:
            x = make_Expression(x)
            if self.is_zero():
                return x
            elif x.is_zero():
                return self
            elif self.is_constant() and x.is_constant():
                return make_Expression(self.value + x.value)
            else:
                return add([self, x])

    def __radd__(self, x):

        return self.__add__(x)

    def __sub__(self, x):
        
        from .function import subtract
        
        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(-x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__rsub__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r
        
        elif isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return -x
            return x.__rsub__(self)

        else:
            x = make_Expression(x)
            if self.is_zero():
                return -x
            elif x.is_zero():
                return self
            elif self.is_constant() and x.is_constant():
                return make_Expression(self.value - x.value)
            else:
                return subtract([self, x])

    def __rsub__(self, x):

        from .function import subtract

        if isinstance(x, np.ndarray):
            if self.is_zero():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__sub__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r
    
        elif isinstance(x, ExpressionMatrix):
            if self.is_zero():
                return x
            return x.__sub__(self)
                
        else:
            x = make_Expression(x)
            if self.is_zero():
                return x
            elif x.is_zero():
                return -self
            elif self.is_constant() and x.is_constant():
                return make_Expression(x.value - self.value)
            else:
                return subtract([x, self])

    def __mul__(self, x):

        from .function import multiply

        if isinstance(x, np.ndarray):
            if self.is_one():
                return ExpressionMatrix(x)
            self.__array_priority__ = 0.
            r =  ExpressionMatrix(x.__mul__(self))
            self.__array_priority__ = Expression.__array_priority__
            return r

        elif isinstance(x, ExpressionMatrix):
            if self.is_one():
                return x
            return x.__mul__(self)
            
        else:
            x = make_Expression(x)
            if self.is_one():
                return x
            elif x.is_one():
                return self
            elif self.is_constant() and x.is_constant():
                return make_Expression(self.value*x.value)
            else:
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
        
        from .constraint import Constraint, ConstraintArray
        
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

        G = nx.MultiDiGraph()
        prop = self.__analyze__(G, '')
        vars = list(prop['a'].keys())
        
        n = len(vars)
        
        for i in range(0, n):

            var1 = vars[i]

            d = self.get_derivative(var1, G=G)

            gphi_list.append((var1, d))
            
            dG = nx.MultiDiGraph()
            dprop = d.__analyze__(dG, '')
            dvars = set(dprop['a'].keys())
            
            for j in range(i, n):

                var2 = vars[j]

                if var2 not in dvars:
                    continue

                dd = d.get_derivative(var2, G=dG)

                Hphi_list.append((var1, var2, dd))
                
        return {'phi': phi,
                'gphi_list': gphi_list,
                'Hphi_list': Hphi_list,
                'phi_prop': prop}

    def __manager_node_type__(self):

        return NotImplemented

    def __fill_manager__(self, manager):

        raise NotImplementedError

    def get_derivative(self, var, G=None):

        return make_Expression(0.)

    def get_value(self):

        return 0.

    def get_variables(self):
    
        return set()

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

    from .constant import Constant
    
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

        from .function import negate

        return negate(self)

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

        from .constraint import ConstraintArray

        if isinstance(x, ExpressionMatrix):
            return ConstraintArray(np.vectorize(lambda a,b: a.__cmp_util__(op, b))(self.data, x.data))

        else:
            return ConstraintArray(np.vectorize(lambda a,b: a.__cmp_util__(op, b))(self.data, np.asarray(x)))
        
    def get_value(self):

        if self.data.size:
            return np.vectorize(lambda x: x.get_value())(self.data)
        else:
            return np.asmatrix(self.data, dtype=np.float64)

    def get_fast_evaluator(self, variables):

        assert(isinstance(variables, list))

        m = coptmod.Manager(len(variables), self.shape[0]*self.shape[1])
        for i, var in enumerate(variables):
            m.set_input_var(i, id(var))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x = self.data[i.j]
                x.__fill_manager__(m)
                m.set_output_node(i*self.shape[1]+j, id(x))
        
        return m
              
class SparseExpressionMatrix:

    pass


