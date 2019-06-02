import numpy as np
from . import coptmod
from functools import reduce
from .constant import Constant
from collections import defaultdict
from .variable import VariableScalar
from .expression import Expression, ExpressionMatrix, make_Expression

class Function(Expression):
    
    arguments = []

    def __init__(self, args=[]):

        Expression.__init__(self)
        self.arguments = args

    def __repr__(self):
        
        args = [x.__repr__() for x in self.arguments]
        return '%s(%s)' %(self.name, ','.join(args))

    def __analyze__(self):

        a = {}

        for i, arg in enumerate(self.arguments):
            prop = arg.__analyze__()
            a.update(prop['a'])
            
        return {'affine': False,
                'a': a,
                'b': np.NaN}
            
    def __partial__(self, arg):

        return NotImplemented
    
    def __fill_evaluator__(self, evaluator):

        evaluator.add_node(self.__evaluator_node_type__(),
                           id(self),
                           0.,
                           [id(arg) for arg in self.arguments])

        for arg in self.arguments:
            arg.__fill_evaluator__(evaluator)

    def __all_simple_paths__(self, vars, path):

        if any([not isinstance(var, VariableScalar) for var in vars]):
            raise TypeError('agrument must be set of variables')

        vars = set(vars)
        
        path = path + (self,)
        paths = defaultdict(list)
        for arg in self.arguments:
            if arg in vars:
                paths[arg].append(path+(arg,))
            elif arg.is_function():
                for key, value in arg.__all_simple_paths__(vars, path).items():
                    paths[key].extend(value)
        return paths

    def get_derivatives(self, vars):
        
        paths = self.__all_simple_paths__(vars, ())
        derivs = {}
        for var in vars:
            d = 0.
            for path in paths[var]:            
                prod = make_Expression(1.)
                for parent, child in zip(path[:-1], path[1:]):
                    prod = prod*parent.__partial__(child)
                d = d + prod
            derivs[var] = make_Expression(d)
        return derivs
        
    def get_variables(self):
    
        return reduce(lambda x,y: x.union(y),
                      map(lambda arg: arg.get_variables(), self.arguments),
                      set())

    def get_value(self):

        process = [self]
        processed = []
        while process:
            new_nodes = []
            for n in process:
                processed.insert(0, n)
                if isinstance(n, Function):
                    for nn in n.arguments:
                        new_nodes.append(nn)
            process = new_nodes
        for n in processed:
            n.__set_value__()
        return self.__value__

    def is_function(self):

        return True

class ElementWiseFunction(Function):

    def __new__(cls, arg):
        
        if isinstance(arg, ExpressionMatrix):
            return ExpressionMatrix(np.vectorize(cls)(arg.data))

        elif isinstance(arg, np.ndarray):
            return ExpressionMatrix(np.vectorize(cls)(arg))

        else:
            return super(ElementWiseFunction, cls).__new__(cls)
        
    def __init__(self, arg):

        Function.__init__(self, [make_Expression(arg)])
        assert(len(self.arguments) == 1)
        
class add(Function):

    def __init__(self, args):
        
        Function.__init__(self, args)
        
        self.name = 'add'
        assert(len(self.arguments) >= 2)
        
    def __repr__(self):

        args = [x.__repr__()+' + ' for x in self.arguments]
        return (''.join(args))[:-3]

    def __partial__(self, arg):

        for x in self.arguments:
            if arg is x:
                return make_Expression(1.)
        raise ValueError('invalid argument')

    def __analyze__(self):

        args = self.arguments
        
        props = []
        for i, arg in enumerate(args):
            props.append(arg.__analyze__())

        new_a = props[0]['a']
        for prop in props[1:]:
            for x in prop['a']:
                if x in new_a:
                    new_a[x] += prop['a'][x]
                else:
                    new_a[x] = prop['a'][x]

        return {'affine': all([prop['affine'] for prop in props]),
                'a': new_a,
                'b': sum([prop['b'] for prop in props])}

    def __evaluator_node_type__(self):
            
        return coptmod.NODE_TYPE_ADD

    def __set_value__(self):

        self.__value__ = np.sum(list(map(lambda a: a.__value__, self.arguments)))
    
class multiply(Function):

    def __init__(self, args):

        Function.__init__(self, args)

        self.name = 'multiply'
        assert(len(self.arguments) == 2)
        
    def __repr__(self):

        a = self.arguments[0]
        b = self.arguments[1]

        needp = lambda x: True if isinstance(x, add) else False

        return '%s*%s' %('(%s)' %str(a) if needp(a) else '%s' %str(a),
                         '(%s)' %str(b) if needp(b) else '%s' %str(b))

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return self.arguments[1]
        
        elif arg is self.arguments[1]:
            return self.arguments[0]
        
        else:
            raise ValueError('invalid argument')

    def __analyze__(self):

        arg1, arg2 = self.arguments
        
        prop1 = arg1.__analyze__()
        prop2 = arg2.__analyze__()

        a1 = dict([(x, val*prop2['b']) for x, val in prop1['a'].items()])
        a2 = dict([(x, val*prop1['b']) for x, val in prop2['a'].items()])

        new_a = a1
        new_a.update(a2)

        return {'affine': (prop1['affine'] and not prop2['a']) or (prop2['affine'] and not prop1['a']),
                'a': new_a,
                'b': prop1['b']*prop2['b']}

    def __evaluator_node_type__(self):
    
        return coptmod.NODE_TYPE_MULTIPLY

    def __set_value__(self):

        self.__value__ = np.prod([a.__value__ for a in self.arguments])

class sin(ElementWiseFunction):

    def __new__(cls, arg):

        if isinstance(arg, Constant):
            return make_Expression(np.sin(arg.__value__))

        else:
            return super(sin, cls).__new__(cls, arg)

    def __init__(self, arg):

        ElementWiseFunction.__init__(self, arg)

        self.name = 'sin'

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return cos(arg)

        else:
            raise ValueError('invalid argument')

    def __evaluator_node_type__(self):
            
        return coptmod.NODE_TYPE_SIN
        
    def __set_value__(self):

        self.__value__ = np.sin(self.arguments[0].__value__)

class cos(ElementWiseFunction):

    def __new__(cls, arg):

        if isinstance(arg, Constant):
            return make_Expression(np.cos(arg.__value__))

        else:
            return super(cos, cls).__new__(cls, arg)

    def __init__(self, arg):

        ElementWiseFunction.__init__(self, arg)
        
        self.name = 'cos'

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return -sin(arg)

        else:
            raise ValueError('invalid argument')

    def __evaluator_node_type__(self):
            
        return coptmod.NODE_TYPE_COS

    def __set_value__(self):

        self.__value__ = np.cos(self.arguments[0].__value__)
