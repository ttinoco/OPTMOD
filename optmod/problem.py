import types
import numpy as np
from .constraint import Constraint
from scipy.sparse import coo_matrix
from .expression import make_Expression, ExpressionMatrix

class Objective(object):

    function = None

    def __init__(self, func):

        self.function = make_Expression(func)

    def __get_std_components__(self):

        return NotImplemented

    def get_function(self):

        return self.function

    #def get_variables(self):
    #
    #    return self.function.get_variables()

class minimize(Objective):

    def __init__(self, func):

        Objective.__init__(self, func)

    def __repr__(self):

        return 'minimize %s' %self.function

    def __get_std_components__(self):

        return self.function.__get_std_components__()

class maximize(Objective):

    def __init__(self, func):
        
        Objective.__init__(self, func)

    def __repr__(self):
            
        return 'maximize %s' %self.function

    def __get_std_components__(self):

        return (-self.function).__get_std_components__()

class EmptyObjective(minimize):

    def __init__(self):

        minimize.__init__(self, 0.)

    def __repr__(self):

        return 'empty'
    
    def __get_std_components__(self):

        return self.function.__get_std__components__()

class Problem(object):

    INF = 1e8
    
    objective = None
    constraints = None
    sense = None

    def __init__(self, objective=None, constraints=[]):

        if objective is None:
            objective =  EmptyObjective()

        if not isinstance(objective, Objective):
            raise TypeError('invalid objective type')
        for c in constraints:
            if not isinstance(c, Constraint):
                raise TypeError('invalid constraint type')

        self.objective = objective
        self.constraints = sum([c.flatten().tolist() for c in constraints], [])
        
    def __repr__(self):

        s = ('\nObjective:\n' +
             '  %s' %str(self.objective))
        s += '\n'
        s += '\nConstraints:\n'
        for c in self.constraints:
            s += '  %s\n' %str(c)
        if not self.constraints:
            s + 'none'

        return s

    def __get_std_components__(self):
        
        obj_comp = self.objective.__get_std_components__()

        row_counters = {'A_row': 0, 'J_row': 0}
        constr_comp = dict([(key, list()) for key in Constraint.__get_std_keys__()])
        for c in self.constraints:
            comp = c.__get_std_components__(row_counters)
            for key in comp:
                constr_comp[key] += comp[key]
                                
        return dict(obj_comp, **constr_comp)

    def __get_std_problem__(self):

        import optalg

        comp = self.__get_std_components__()

        # Vars
        vars = set(comp['phi_prop']['a'].keys())
        for prop in comp['prop_list']:
            vars |= set(list(prop['a'].keys()))
        vars = list(vars)
        num_vars = len(vars)
        var_values = np.array([x.get_value() for x in vars])

        # Index map
        var2index = dict(zip(vars, range(len(vars))))
        index2var = dict(zip(range(len(vars)), vars))
        
        # Objective
        phi_data = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']
        gphi_indices = np.array([var2index[x] for x, exp in gphi_list])
        gphi_data = ExpressionMatrix([exp for x, exp in gphi_list])
        row = []
        col = []
        data = []
        for vari, varj, d in Hphi_list:
            i = var2index[vari]
            j = var2index[varj]
            if i >= j:
                row.append(i)
                col.append(j)
            else:
                row.append(j)
                col.append(i)
            data.append(d)
        Hphi_row = np.array(row, dtype=int)
        Hphi_col = np.array(col, dtype=int)
        Hphi_data = ExpressionMatrix(data)
        
        # Linear constraints
        A_list = comp['A_list']
        b_list = comp['b_list']
        row, col, data = zip(*A_list) if A_list else ([], [], [])
        A = coo_matrix((np.array(data, dtype=float),
                        (np.array(row, dtype=int),
                         np.array([var2index[x] for x in col], dtype=int))),
                       shape=(len(b_list), num_vars))
        b = np.array(b_list, dtype=float)

        # Nonlinear constraints
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']        
        f_data = ExpressionMatrix(f_list)
        row, col, data = zip(*J_list) if J_list else ([], [], [])
        J_row = np.array(row, dtype=int)
        J_col = np.array([var2index[x] for x in col], dtype=int)
        J_data = ExpressionMatrix(data)
        H_comb_row = []
        H_comb_col = []
        H_comb_data = []
        H_comb_nnz = []
        for HH_list in H_list:
            row = []
            col = []
            data = []
            for vari, varj, d in HH_list:
                i = var2index[vari]
                j = var2index[varj]
                if i >= j:
                    row.append(i)
                    col.append(j)
                else:
                    row.append(j)
                    col.append(i)
                data.append(d)
            H_comb_row.extend(row)
            H_comb_col.extend(col)
            H_comb_data.extend(data)
            H_comb_nnz.append(len(data))
        H_comb_row = np.array(H_comb_row)
        H_comb_col = np.array(H_comb_col)
        H_comb_data = ExpressionMatrix(H_comb_data)
        H_comb_nnz = np.array(H_comb_nnz)

        # Bounds
        u_list = comp['u_list']
        l_list = comp['l_list']
        u = Problem.INF*np.ones(num_vars)
        l = -Problem.INF*np.ones(num_vars)
        for x, val in u_list:
            index = var2index[x]
            u[index] = np.minimum(u[index], val)
        for x, val in l_list:
            index = var2index[x]
            l[index] = np.maximum(l[index], val)
                    
        # Problem
        p = optalg.opt_solver.OptProblem()
        
        p.phi = 0
        p.gphi = np.zeros(num_vars)
        p.Hphi = coo_matrix((np.zeros(Hphi_data.shape[1]),
                             (Hphi_row, Hphi_col)),
                            shape=(num_vars, num_vars))

        p.A = A
        p.b = b

        p.f = np.zeros(f_data.shape[1])
        p.J = coo_matrix((np.zeros(J_data.shape[1]),
                          (J_row, J_col)),
                         shape=(p.f.size, num_vars))        
        p.H_combined = coo_matrix((np.zeros(int(np.sum(H_comb_nnz))),
                                   (H_comb_row, H_comb_col)),
                                  shape=(num_vars, num_vars))

        p.u = u
        p.l = l

        p.x = var_values

        # Aux data
        p.var2index = var2index       # dict: var -> index
        p.index2var = index2var       # dict: index -> var
        p.phi_data = phi_data         # expression
        p.gphi_indices = gphi_indices # array of indices
        p.gphi_data = gphi_data       # expression array
        p.Hphi_data = Hphi_data       # expression array
        p.f_data = f_data             # expression array
        p.J_data = J_data             # expression array
        p.H_comb_data = H_comb_data   # expression array
        p.H_comb_nnz = H_comb_nnz     # array
        
        # Eval
        def eval(obj, x):

            # Set values
            for i, var in obj.index2var.items():
                var.set_value(x[i])
                
            # Eval experssions                    
            obj.phi = obj.phi_data.get_value()
            obj.gphi[obj.gphi_indices] = obj.gphi_data.get_value()
            obj.Hphi.data[:] = obj.Hphi_data.get_value()
            obj.f[:] = obj.f_data.get_value()
            obj.J.data[:] = obj.J_data.get_value()
            obj.H_combined.data[:] = obj.H_comb_data.get_value()
        p.eval = types.MethodType(eval, p)
                
        # Combine H
        def combine_H(obj, lam, ensure_psd=False):            
            offset = 0
            assert(lam.size == obj.H_comb_nnz.size)
            for i, nnz in enumerate(obj.H_comb_nnz):                
                obj.H_combined.data[offset:offset+nnz] *= lam[i]
                offset += nnz
            
        p.combine_H = types.MethodType(combine_H, p)
        
        # Return
        return p
            
    #def get_variables(self):
    #
    #    return self.get_variables().union(*[c.get_variables() for c in self.constraints])

    def solve(self, solver='ipopt', parameters=None):

        import optalg

        if parameters is None:
            parameters = {}

        std_prob = self.__get_std_problem__()
        
        if solver == 'augl':
            solver = optalg.opt_solver.OptSolverAugL()
        elif solver == 'ipopt':
            solver = optalg.opt_solver.OptSolverIpopt()
        elif solver == 'inlp':
            solver = optalg.opt_solver.OptSolverINLP()

        solver.set_parameters(parameters)

        try:
            solver.solve(std_prob)
        except optalg.opt_solver.OptSolverError:
            pass

        return solver.get_status()
