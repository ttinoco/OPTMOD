import time
import types
import numpy as np
from . import coptmod
from scipy.sparse import coo_matrix
from .constraint import Constraint, ConstraintArray
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

        return self.function.__get_std_components__()

class Problem(object):

    INF = 1e8
    
    objective = None
    constraints = None
    sense = None

    def __init__(self, objective=None, constraints=[]):

        if objective is None:
            objective = EmptyObjective()

        if not isinstance(objective, Objective):
            raise TypeError('invalid objective type')

        self.objective = objective
        self.constraints = []
        for c in constraints:
            if isinstance(c, Constraint):
                self.constraints.append(c)
            elif isinstance(c, bool) or isinstance(c, np.bool_):
                if not c:
                    raise ValueError('infeasible constraint')
            else:
                self.constraints.extend(ConstraintArray(c).flatten().tolist())
        
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

        counters = {'A_row': 0, 'J_row': 0}
        constr_comp = dict([(key, list()) for key in Constraint.__get_std_keys__()])
        for c in self.constraints:
            comp = c.__get_std_components__(counters=counters)
            for key in comp:
                constr_comp[key] += comp[key]
                                
        return dict(obj_comp, **constr_comp)

    def __get_std_problem__(self, fast_evaluator=True):

        from optalg.opt_solver import OptProblem

        comp = self.__get_std_components__()

        # Vars
        vars = set(comp['phi_prop']['a'].keys())
        for prop in comp['prop_list']:
            vars |= set(list(prop['a'].keys()))
        vars = sorted(list(vars), key=lambda x: x.id)
        num_vars = len(vars)
        var_values = np.array([x.get_value() for x in vars])
        
        # Index map
        var2index = dict(zip(vars, range(len(vars))))
        index2var = dict(zip(range(len(vars)), vars))
        
        # Objective
        phi_data = ExpressionMatrix(comp['phi'])
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
        Aindex2constr = dict(enumerate(comp['cA_list']))
        A_list = comp['A_list']
        b_list = comp['b_list']
        row, col, data = zip(*A_list) if A_list else ([], [], [])
        A = coo_matrix((np.array(data, dtype=float),
                        (np.array(row, dtype=int),
                         np.array([var2index[x] for x in col], dtype=int))),
                       shape=(len(b_list), num_vars))
        b = np.array(b_list, dtype=float)

        # Nonlinear constraints
        Jindex2constr = dict(enumerate(comp['cJ_list']))
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
        H_comb_broad_col = []
        for k, HH_list in enumerate(H_list):
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
            nnz = len(data)
            H_comb_row.extend(row)
            H_comb_col.extend(col)
            H_comb_data.extend(data)
            H_comb_nnz.append(nnz)
            H_comb_broad_col.extend([k]*nnz)
        H_comb_row = np.array(H_comb_row)
        H_comb_col = np.array(H_comb_col)
        H_comb_data = ExpressionMatrix(H_comb_data)
        H_comb_nnz = np.array(H_comb_nnz)

        # Bounds
        uindex2constr = {}
        lindex2constr = {}
        u_list = comp['u_list']
        l_list = comp['l_list']
        u = Problem.INF*np.ones(num_vars)
        l = -Problem.INF*np.ones(num_vars)
        for x, val, c in u_list:
            index = var2index[x]
            if val <= u[index]:
                u[index] = val
                uindex2constr[index] = c

        for x, val, c in l_list:
            index = var2index[x]
            if val >= l[index]:
                l[index] = val
                lindex2constr[index] = c
                    
        # Problem
        p = OptProblem()
        
        p.phi = 0.
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
        p.H_combined_broad = coo_matrix((np.ones(p.H_combined.nnz),
                                         (range(p.H_combined.nnz), H_comb_broad_col)),
                                        shape=(p.H_combined.nnz, p.f.size)).tocsr()

        p.u = u
        p.l = l

        p.P = np.array([vars[i].is_integer() for i in range(num_vars)], dtype=bool)

        p.x = var_values

        # Aux data
        p.var2index = var2index         # dict: var -> index
        p.index2var = index2var         # dict: index -> var
        p.Aindex2constr = Aindex2constr # dict: index -> constraint
        p.Jindex2constr = Jindex2constr # dict: index -> constraint
        p.uindex2constr = uindex2constr # dict: index -> constraint
        p.lindex2constr = lindex2constr # dict: index -> constraint
        p.phi_data = phi_data           # expression matrix
        p.gphi_indices = gphi_indices   # array of indices
        p.gphi_data = gphi_data         # expression matrix
        p.Hphi_data = Hphi_data         # expression matrix
        p.f_data = f_data               # expression matrix
        p.J_data = J_data               # expression matrix
        p.H_comb_data = H_comb_data     # expression matrix
        p.H_comb_nnz = H_comb_nnz       # array

        # Properties (curvature)
        p.properties = []
        if comp['phi_prop']['affine'] and all([prop['affine'] for prop in comp['prop_list']]):
            p.properties.append(OptProblem.PROP_CURV_LINEAR)
        else:
            p.properties.append(OptProblem.PROP_CURV_NONLINEAR)

        # Properties (var types)
        if np.any(p.P):
            p.properties.append(OptProblem.PROP_VAR_INTEGER)
        else:
            p.properties.append(OptProblem.PROP_VAR_CONTINUOUS)

        # Properties (problem type)
        if not comp['phi_prop']['a'].keys():
            p.properties.append(OptProblem.PROP_TYPE_FEASIBILITY)
        else:
            p.properties.append(OptProblem.PROP_TYPE_OPTIMIZATION)

        # Slow evaluator
        if not fast_evaluator:
        
            # Eval
            def eval(obj, x):
                
                # Set values
                for i, var in obj.index2var.items():
                    var.set_value(x[i])
                
                # Eval experssions                    
                obj.phi = obj.phi_data[0,0].get_value()
                if obj.gphi_indices.size:
                    obj.gphi[obj.gphi_indices] = obj.gphi_data.get_value()
                obj.Hphi.data[:] = obj.Hphi_data.get_value()
                obj.f[:] = obj.f_data.get_value()
                obj.J.data[:] = obj.J_data.get_value()
                obj.H_combined.data[:] = obj.H_comb_data.get_value()
                
            p.eval = types.MethodType(eval, p)

        # Fast evaluator
        else:

            offset_phi_data = 0
            offset_gphi_data = offset_phi_data + phi_data.shape[1]
            offset_Hphi_data = offset_gphi_data + gphi_data.shape[1]
            offset_f_data = offset_Hphi_data + Hphi_data.shape[1]
            offset_J_data = offset_f_data + f_data.shape[1]
            offset_H_comb_data = offset_J_data + J_data.shape[1]
            total_size = offset_H_comb_data + H_comb_data.shape[1]

            e = coptmod.Evaluator(len(vars),
                                  total_size,
                                  shape=(1, total_size),
                                  scalar_output=False)

            for offset, exp_mat in [(offset_phi_data, phi_data),
                                    (offset_gphi_data, gphi_data),
                                    (offset_Hphi_data, Hphi_data),
                                    (offset_f_data, f_data),
                                    (offset_J_data, J_data),
                                    (offset_H_comb_data, H_comb_data)]:
                                    
                data = exp_mat.get_data()
                for i in range(data.size):
                    data[0,i].__fill_evaluator__(e)
                    e.set_output_node(offset+i, id(data[0,i]))

            for i, var in enumerate(vars):
                e.set_input_var(i, id(var))
                
            p.e = e
            p.offset_phi_data = offset_phi_data
            p.offset_gphi_data = offset_gphi_data
            p.offset_Hphi_data = offset_Hphi_data
            p.offset_f_data = offset_f_data
            p.offset_J_data = offset_J_data
            p.offset_H_comb_data  = offset_H_comb_data

            # Eval
            def eval(obj, x):
                
                # Eval experssions
                obj.e.eval(x)

                # Extract values
                value = obj.e.get_value()
                obj.phi = value[0,obj.offset_phi_data]
                if obj.gphi_indices.size:
                    obj.gphi[obj.gphi_indices] = value[0,obj.offset_gphi_data:obj.offset_Hphi_data]
                obj.Hphi.data[:] = value[0,obj.offset_Hphi_data:obj.offset_f_data]
                obj.f[:] = value[0,obj.offset_f_data:obj.offset_J_data]
                obj.J.data[:] = value[0,obj.offset_J_data:obj.offset_H_comb_data]
                obj.H_combined.data[:] = value[0,obj.offset_H_comb_data:]
                
            p.eval = types.MethodType(eval, p)
                
        # Combine H
        def combine_H(obj, lam, ensure_psd=False):
            obj.H_combined.data *= obj.H_combined_broad*lam
            
        p.combine_H = types.MethodType(combine_H, p)
                
        # Return
        return p
            
    #def get_variables(self):
    #
    #    return self.get_variables().union(*[c.get_variables() for c in self.constraints])

    def solve(self, solver=None, parameters=None, fast_evaluator=True):

        import optalg

        # Solver
        if solver is None:
            solver = optalg.opt_solver.OptSolverINLP()

        # Params
        if parameters is None:
            parameters = {}

        # Problem
        t0 = time.time()
        std_prob = self.__get_std_problem__(fast_evaluator=fast_evaluator)
        time_transformation = time.time()-t0

        # Info
        if 'quiet' not in parameters or not parameters['quiet']:
            std_prob.show()

        # Properties
        if not solver.supports_properties(std_prob.properties):
            raise TypeError('problem type not supported by solver')

        # Configure solver
        solver.set_parameters(parameters)

        # Solve
        t0 = time.time()
        try:
            solver.solve(std_prob)
        except optalg.opt_solver.OptSolverError:
            pass
        time_solver = time.time()-t0

        # Get primal values
        x = solver.get_primal_variables()
        if x is not None and x.size:
            for i, var in std_prob.index2var.items():
                var.set_value(x[i])

        # Get dual variables
        lam, nu, mu, pi = solver.get_dual_variables()        
        if (lam is not None) and lam.size:
            for i, c in std_prob.Aindex2constr.items():
                c.set_dual(lam[i])
        if (nu is not None) and nu.size:
            for i, c in std_prob.Jindex2constr.items():
                c.set_dual(nu[i])
        if (mu is not None) and mu.size:
            for i, c in std_prob.uindex2constr.items():
                c.set_dual(mu[i])
        if (pi is not None) and pi.size:
            for i, c in std_prob.lindex2constr.items():
                c.set_dual(pi[i])
                    
        # Info
        info = {'status': solver.get_status(),
                'iterations': solver.get_iterations(),
                'time_transformation': time_transformation,
                'time_solver': time_solver}

        # Return
        return info
