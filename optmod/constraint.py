import numpy as np
from .variable import VariableScalar
from .expression import make_Expression

class Constraint(object):
    
    lhs = None
    op = None
    rhs = None

    def __init__(self, lhs, op, rhs):
        
        self.lhs = make_Expression(lhs)
        self.op = op
        self.rhs = make_Expression(rhs)
        self.slack = VariableScalar(name='s')
        self.dual = 0.

        assert(op in ['==', '>=', '<='])

    def __repr__(self):

        return '%s %s %s' %(self.lhs, self.op, self.rhs)

    @classmethod
    def __get_std_keys__(cls):

        return ['cA_list',
                'cJ_list',
                'A_list',
                'b_list',
                'f_list',
                'J_list',
                'H_list',
                'u_list',
                'l_list',
                'prop_list']

    def __get_std_components__(self, counters=None):

        cA_list = [] # list of constraints
        cJ_list = [] # list of constraints
        
        A_list = [] # list of (row index, variable, data value)
        b_list = [] # list of values
        f_list = [] # list of expressions
        J_list = [] # list of (row index, variable, expression)
        H_list = [] # list of list of (variable, variable, expression)
        u_list = [] # list of (variable, data value, constraint)
        l_list = [] # list of (variable, data value, constraint)
        prop_list = []

        if counters is None:
            counters = {'A_row': 0, 'J_row': 0}

        exp = self.lhs-self.rhs
        op = self.op

        exp_comp = exp.__get_std_components__()
        phi = exp_comp['phi']
        gphi_list = exp_comp['gphi_list']
        Hphi_list = exp_comp['Hphi_list']
        phi_prop = exp_comp['phi_prop']

        a = phi_prop['a']
        b = phi_prop['b']
        affine = phi_prop['affine']

        prop_list.append(phi_prop)
        
        # Bound
        if affine and len(a) == 1 and list(a.values())[0] == 1. and op != '==':
            
            if op == '<=': # x + b <= 0
                u_list.append((list(a.keys())[0], -b, self))
            else:          # x + b >= 0
                l_list.append((list(a.keys())[0], -b, self))
            
        # Linear
        elif affine:

            if op == '==': # a^Tx + b == 0

                for x, val in a.items():
                    A_list.append((counters['A_row'], x, val))
                b_list.append(-b)
                cA_list.append(self)
                counters['A_row'] += 1

            else : # a^Tx + b - s == 0 and s <= 0 or s >= 0:

                s = self.slack
                for x, val in a.items():
                    A_list.append((counters['A_row'], x, val))
                A_list.append((counters['A_row'], s, -1.))
                b_list.append(-b)
                cA_list.append(self)
                if op == '<=':
                    u_list.append((s, 0, self))
                else:
                    l_list.append((s, 0, self))
                counters['A_row'] += 1
                a[s] = 1.
                
        # Nonlinear
        else:

            H_list.append(Hphi_list)

            if op == '==': # f(x) == 0

                f_list.append(phi)
                cJ_list.append(self)
                for x, val in gphi_list:
                    J_list.append((counters['J_row'], x, val))
                counters['J_row'] += 1

            else: # f(x) - s == 0 and s <= 0 or s >= 0:

                s = self.slack
                f_list.append(phi-s)
                cJ_list.append(self)
                for x, val in gphi_list:
                    J_list.append((counters['J_row'], x, val))
                J_list.append((counters['J_row'], s, make_Expression(-1.)))
                if op == '<=':
                    u_list.append((s, 0, self))
                else:
                    l_list.append((s, 0, self))
                counters['J_row'] += 1
                a[s] = 1.
        
        # Return
        return {'cA_list': cA_list,
                'cJ_list': cJ_list,
                'A_list': A_list,
                'b_list': b_list,
                'f_list': f_list,
                'J_list': J_list,
                'H_list': H_list,
                'u_list': u_list,
                'l_list': l_list,
                'prop_list': prop_list}

    def flatten(self):

        return self

    def get_violation(self):

        if self.op == '==':
            return np.abs(self.lhs.get_value()-self.rhs.get_value())
        elif self.op == '>=':
            return np.maximum(-self.lhs.get_value()+self.rhs.get_value(), 0.)
        elif self.op == '<=':
            return np.maximum(self.lhs.get_value()-self.rhs.get_value(), 0.)
        else:
            raise RuntimeError('Invalid constraints')

    def get_variables(self):

        return self.lhs.get_variables().union(self.rhs.get_variables())

    def get_dual(self):

        return self.dual

    def is_equality(self):

        return self.op == '=='

    def is_inequality(self):

        return self.op != '=='

    def tolist(self):

        return [self]

    def set_dual(self, dual):

        self.dual = dual
        
class ConstraintArray(object):

    data = None
    shape = None

    def __init__(self, obj=None):

        if isinstance(obj, ConstraintArray):
            self.data = obj.data
            self.shape = obj.shape
            
        elif isinstance(obj, Constraint):
            self.data = np.asarray(obj)
            self.shape = self.data.shape
            
        elif obj is not None:
            obj = np.asarray(obj)
            def fn(x):
                if isinstance(x, Constraint):
                    return x
                else:
                    raise TypeError('invalid object')
            self.data = np.vectorize(fn)(obj)
            self.shape = obj.shape
        
    def __getitem__(self, key):

        m = np.asarray(self.data)[key]
        if isinstance(m, Constraint):
            return m
        else:
            return ConstraintArray(m)

    def flatten(self):

        return ConstraintArray(self.data.flatten())

    def tolist(self):

        return self.data.tolist()
