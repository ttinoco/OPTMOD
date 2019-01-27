import numpy as np
import networkx as nx
from . import coptmod
from .constant import Constant
from .variable import VariableScalar
from .expression import Expression, ExpressionMatrix, make_Expression

class Function(Expression):

    arguments = []

    def __init__(self, args=[]):

        Expression.__init__(self)
        args = args if isinstance(args, list) else [args]
        self.arguments = list(map(make_Expression, args))

    def __repr__(self):
        
        args = [x.__repr__() for x in self.arguments]
        return '%s(%s)' %(self.name, ','.join(args))

    def __analyze__(self, G, prefix):

        a = {}

        G.add_node(self.__node__(prefix), item=self)
        for i, arg in enumerate(self.arguments):
            G.add_edge(self.__node__(prefix), arg.__node__(prefix+'%d.' %i))
            prop = arg.__analyze__(G, prefix+'%d.' %i)
            a.update(prop['a'])
            
        return {'affine': False,
                'a': a,
                'b': np.NaN}
            
    def __partial__(self, arg):

        return NotImplemented
    
    def __fill_manager__(self, manager):

        manager.add_node(self.__manager_node_type__(),
                         id(self),
                         0.,
                         [id(arg) for arg in self.arguments])

        for arg in self.arguments:
            arg.__fill_manager__(manager)

    def get_derivative(self, var, G=None):

        if not isinstance(var, VariableScalar):
            raise TypeError('argument must be a variable')

        if G is None:
            G = nx.MultiDiGraph()
            self.__analyze__(G, '')
            
        try:
            paths = nx.all_simple_paths(G, source=self.__node__(''), target=var.__node__(''))
        except Exception:
            return make_Expression(0.)

        d = 0.
        for path in paths:            
            prod = make_Expression(1.)
            for parent_node, child_node in zip(path[:-1], path[1:]):
                parent = G.node[parent_node]['item']
                child = G.node[child_node]['item']
                prod = prod*parent.__partial__(child)
            d = d + prod
        return d

    def get_variables(self):
    
        return reduce(lambda x,y: x.union(y),
                      map(lambda arg: arg.get_variables(), self.arguments),
                      set())

    def is_function(self):

        return True

class ElementWiseFunction(Function):

    def __new__(cls, args):
        
        if isinstance(args, ExpressionMatrix):
            return ExpressionMatrix(np.vectorize(cls)(args.data))

        elif isinstance(args, np.ndarray):
            return ExpressionMatrix(np.vectorize(cls)(args))

        else:
            return super(ElementWiseFunction, cls).__new__(cls)
        
    def __init__(self, args):

        Function.__init__(self, args)
        assert(len(self.arguments) == 1)
        
class add(Function):

    def __init__(self, args):
        
        Function.__init__(self, args)
        
        self.name = 'add'
        assert(len(self.arguments) == 2)
        
    def __repr__(self):

        args = [x.__repr__()+' + ' for x in self.arguments]
        return (''.join(args))[:-3]

    def __partial__(self, arg):
        
        if arg is self.arguments[0] or arg is self.arguments[1]:
            return make_Expression(1.)

        else:
            raise ValueError('invalid argument')

    def __analyze__(self, G, prefix):

        arg1, arg2 = self.arguments
        
        G.add_node(self.__node__(prefix), item=self)
        G.add_edge(self.__node__(prefix), arg1.__node__(prefix+'0.'))
        G.add_edge(self.__node__(prefix), arg2.__node__(prefix+'1.'))

        prop1 = arg1.__analyze__(G, prefix+'0.')
        prop2 = arg2.__analyze__(G, prefix+'1.')

        new_a = prop1['a']
        for x in prop2['a']:
            if x in new_a:
                new_a[x] += prop2['a'][x]
            else:
                new_a[x] = prop2['a'][x]

        return {'affine': prop1['affine'] and prop2['affine'],
                'a': new_a,
                'b': prop1['b'] + prop2['b']}

    def __manager_node_type__(self):
            
        return coptmod.NODE_TYPE_ADD

    def get_value(self):
        
        return np.sum(list(map(lambda a: a.get_value(), self.arguments)))

class subtract(Function):

    def __init__(self, args):
        
        Function.__init__(self, args)
        
        self.name = 'subtract'
        assert(len(self.arguments) == 2)
        
    def __repr__(self):

        a = self.arguments[0]
        b = self.arguments[1]

        return '%s - %s' %(a.__repr__(),
                           ('(%s)' %b.__repr__()) if (isinstance(b, add) or
                                                      isinstance(b, subtract)) else b.__repr__())

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return make_Expression(1.)

        elif arg is self.arguments[1]:
            return make_Expression(-1.)

        else:
            raise ValueError('invalid argument')

    def __analyze__(self, G, prefix):
        
        arg1, arg2 = self.arguments
        
        G.add_node(self.__node__(prefix), item=self)
        G.add_edge(self.__node__(prefix), arg1.__node__(prefix+'0.'))
        G.add_edge(self.__node__(prefix), arg2.__node__(prefix+'1.'))

        prop1 = arg1.__analyze__(G, prefix+'0.')
        prop2 = arg2.__analyze__(G, prefix+'1.')

        new_a = prop1['a']
        for x in prop2['a']:
            if x in new_a:
                new_a[x] -= prop2['a'][x]
            else:
                new_a[x] = -prop2['a'][x]

        return {'affine': prop1['affine'] and prop2['affine'],
                'a': new_a,
                'b': prop1['b'] - prop2['b']}

    def __manager_node_type__(self):
            
        return coptmod.NODE_TYPE_SUBTRACT
        
    def get_value(self):
        
        return self.arguments[0].get_value()-self.arguments[1].get_value()
    
class multiply(Function):

    def __init__(self, args):

        Function.__init__(self, args)

        self.name = 'multiply'
        assert(len(self.arguments) == 2)
        
    def __repr__(self):

        a = self.arguments[0]
        b = self.arguments[1]

        needp = lambda x: True if (isinstance(x, add) or
                                   isinstance(x, subtract) or
                                   isinstance(x, negate)) else False

        return '%s*%s' %('(%s)' %str(a) if needp(a) else '%s' %str(a),
                         '(%s)' %str(b) if needp(b) else '%s' %str(b))

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return self.arguments[1]
        
        elif arg is self.arguments[1]:
            return self.arguments[0]
        
        else:
            raise ValueError('invalid argument')

    def __analyze__(self, G, prefix):

        arg1, arg2 = self.arguments
        
        G.add_node(self.__node__(prefix), item=self)
        G.add_edge(self.__node__(prefix), arg1.__node__(prefix+'0.'))
        G.add_edge(self.__node__(prefix), arg2.__node__(prefix+'1.'))

        prop1 = arg1.__analyze__(G, prefix+'0.')
        prop2 = arg2.__analyze__(G, prefix+'1.')

        a1 = dict([(x, val*prop2['b']) for x, val in prop1['a'].items()])
        a2 = dict([(x, val*prop1['b']) for x, val in prop2['a'].items()])

        new_a = a1
        new_a.update(a2)

        return {'affine': (prop1['affine'] and not prop2['a']) or (prop2['affine'] and not prop1['a']),
                'a': new_a,
                'b': prop1['b']*prop2['b']}

    def __manager_node_type__(self):
    
        return coptmod.NODE_TYPE_MULTIPLY

    def get_value(self):

        return np.prod([a.get_value() for a in self.arguments])

class negate(ElementWiseFunction):

    def __new__(cls, args):

        if isinstance(args, negate):
            return args.arguments[0]

        elif isinstance(args, Constant):
            return make_Expression(-args.get_value())

        else:
            return super(negate, cls).__new__(cls, args)
            
    def __init__(self, args):

        ElementWiseFunction.__init__(self, args)
        
        self.name = 'negate'

    def __repr__(self):
        
        a = self.arguments[0]
        
        needp = lambda x: True if (isinstance(x, add) or
                                   isinstance(x, subtract)) else False

        return '-%s' %('(%s)' %str(a) if needp(a) else '%s' %str(a))

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return make_Expression(-1.)

        else:
            raise ValueError('invalid argument')

    def __analyze__(self, G, prefix):

        arg = self.arguments[0]
        
        G.add_node(self.__node__(prefix), item=self)
        G.add_edge(self.__node__(prefix), arg.__node__(prefix+'0.'))

        prop = arg.__analyze__(G, prefix+'0.')

        new_a = dict([(x, -val) for x, val in prop['a'].items()])

        return {'affine': prop['affine'],
                'a': new_a,
                'b': -prop['b']}

    def __manager_node_type__(self):
            
        return coptmod.NODE_TYPE_NEGATE
        
    def get_value(self):

        return -self.arguments[0].get_value()

class sin(ElementWiseFunction):

    def __new__(cls, args):

        if isinstance(args, Constant):
            return make_Expression(np.sin(args.get_value()))

        else:
            return super(sin, cls).__new__(cls, args)

    def __init__(self, args):

        ElementWiseFunction.__init__(self, args)

        self.name = 'sin'

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return cos(arg)

        else:
            raise ValueError('invalid argument')

    def __manager_node_type__(self):
            
        return coptmod.NODE_TYPE_SIN
        
    def get_value(self):

        return np.sin(self.arguments[0].get_value())

class cos(ElementWiseFunction):

    def __new__(cls, args):

        if isinstance(args, Constant):
            return make_Expression(np.cos(args.get_value()))

        else:
            return super(cos, cls).__new__(cls, args)

    def __init__(self, args):

        ElementWiseFunction.__init__(self, args)
        
        self.name = 'cos'

    def __partial__(self, arg):

        if arg is self.arguments[0]:
            return -sin(arg)

        else:
            raise ValueError('invalid argument')

    def __manager_node_type__(self):
            
        return coptmod.NODE_TYPE_COS

    def get_value(self):

        return np.cos(self.arguments[0].get_value())
