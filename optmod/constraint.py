import numpy as np
from .expression import make_Expression
from .variable import VariableScalar

class Constraint(object):
    
    lhs = None
    op = None
    rsh = None

    def __init__(self, lhs, op, rhs):

        self.lhs = make_Expression(lhs)
        self.op = op
        self.rhs = make_Expression(rhs)
        self.slack = None

        assert(op in ['==', '>=', '<='])

    def __repr__(self):

        return '%s %s %s' %(self.lhs, self.op, self.rhs)

    @classmethod
    def __get_std_keys__(cls):

        return ['A_list',
                'b_list',
                'f_list',
                'J_list',
                'H_list',
                'u_list',
                'l_list',
                'prop_list']

    def __get_std_components__(self, counters=None):

        A_list = []
        b_list = []
        f_list = []
        J_list = []
        H_list = []
        u_list = []
        l_list = []
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
        if affine and len(a) == 1 and list(a.values())[0] ==1. and op != '==':
            
            if op == '<=': # x + b <= 0
                u_list.append((list(a.keys())[0], -b))
            else:          # x + b >= 0
                l_list.append((list(a.keys())[0], -b))
            
        # Linear
        elif affine:

            if op == '==': # a^Tx + b == 0

                for x, val in a.items():
                    A_list.append((counters['A_row'], x, val))
                b_list.append(-b)
                counters['A_row'] += 1

            else : # a^Tx + b - s == 0 and s <= 0 or s >= 0:

                s = VariableScalar(name='s')
                for x, val in a.items():
                    A_list.append((counters['A_row'], x, val))
                A_list.append((counters['A_row'], s, -1.))
                b_list.append(-b)
                if op == '<=':
                    u_list.append((s, 0))
                else:
                    l_list.append((s, 0))
                counters['A_row'] += 1
                a[s] = 1.
                self.slack = s
                
        # Nonlinear
        else:

            H_list.append(Hphi_list)

            if op == '==': # f(x) == 0

                f_list.append(phi)
                for x, val in gphi_list:
                    J_list.append((counters['J_row'], x, val))
                counters['J_row'] += 1

            else: # f(x) - s == 0 and s <= 0 or s >= 0:

                s = VariableScalar(name='s')
                f_list.append(phi-s)
                for x, val in gphi_list:
                    J_list.append((counters['J_row'], x, val))
                J_list.append((counters['J_row'], s, make_Expression(-1.)))
                if op == '<=':
                    u_list.append((s, 0))
                else:
                    l_list.append((s, 0))
                counters['J_row'] += 1
                a[s] = 1.
                self.slack = s
        
        # Return
        return {'A_list': A_list,
                'b_list': b_list,
                'f_list': f_list,
                'J_list': J_list,
                'H_list': H_list,
                'u_list': u_list,
                'l_list': l_list,
                'prop_list': prop_list}

    def flatten(self):

        return self

    def get_variables(self):

        return self.lhs.get_variables().union(self.rhs.get_variables())

    def is_equality(self):

        return self.op == '=='

    def is_inequality(self):

        return self.op != '=='

    def tolist(self):

        return [self]
        
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
