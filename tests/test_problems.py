import optmod
import unittest
import numpy as np
from numpy.linalg import norm
from optmod import minimize, maximize, EmptyObjective

class TestProblems(unittest.TestCase):

    def test_construction(self):
        
        x = optmod.variable.VariableScalar('x')

        f = x*x

        c1 = x >= 1
        c2 = x <= 5
        c = [c1, c2]

        p = optmod.Problem(minimize(f), c)

        self.assertTrue(isinstance(p.objective, minimize))
        self.assertTrue(p.objective.get_function() is f)

        self.assertEqual(len(p.constraints), 2)
        self.assertTrue(p.constraints[0] is c1)
        self.assertTrue(p.constraints[1] is c2)

        p = optmod.Problem()
        self.assertTrue(p.objective.get_function().is_constant(0.))
        self.assertEqual(len(p.constraints), 0)

        p = optmod.Problem(maximize(f))
        self.assertTrue(isinstance(p.objective, maximize))
        self.assertTrue(p.objective.get_function() is f)
        self.assertEqual(len(p.constraints), 0)

        p = optmod.Problem()
        self.assertEqual(len(p.constraints), 0)
        self.assertTrue(isinstance(p.objective, optmod.problem.EmptyObjective))

        p = optmod.Problem(objective=None)
        self.assertEqual(len(p.constraints), 0)
        self.assertTrue(isinstance(p.objective, optmod.problem.EmptyObjective))

        self.assertRaises(TypeError, optmod.Problem, objective=f)
        self.assertRaises(TypeError, optmod.Problem, None, ['foo'])

    def test_construction_constr_not_flat(self):

        x = optmod.variable.VariableMatrix('x', value=np.random.randn(3,2))
        y = optmod.variable.VariableScalar('y')

        c0 = x == 0
        self.assertTrue(isinstance(c0, optmod.constraint.ConstraintArray))

        c1 = y == 2

        p = optmod.Problem(EmptyObjective(), constraints=[c0, [[c1]]])
        
        self.assertEqual(len(p.constraints), 7)
        for c in p.constraints:
            self.assertTrue(isinstance(c, optmod.constraint.Constraint))

    def test_std_components(self):

        x = optmod.variable.VariableScalar('x')
        y = optmod.variable.VariableScalar('y')

        f = x*x + 2*x*y + y*y
        c0 = x <= 10
        c1 = x + y == 3
        c2 = 3*x <= 10
        c3 = y >= 19.
        c4 = x*y == 20.
        c5 = optmod.sin(x) <= 3
        c6 = optmod.cos(y) >= 4.

        p = optmod.Problem(minimize(f), [c0, c1, c2, c3, c4, c5, c6])
        
        comp = p.__get_std_components__()

        self.assertEqual(len(comp), 12)

        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']
        phi_prop = comp['phi_prop']

        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(str(phi), 'x*x + x*2.00e+00*y + y*y')
        self.assertTrue(str(gphi_list) == '[(x, y*2.00e+00 + x + x), (y, y + y + x*2.00e+00)]' or
                        str(gphi_list) == '[(x, y*2.00e+00 + x + x), (y, x*2.00e+00 + y + y)]' or
                        str(gphi_list) == '[(x, x + x + y*2.00e+00), (y, x*2.00e+00 + y + y)]' or
                        str(gphi_list) == '[(x, x + x + y*2.00e+00), (y, y + y + x*2.00e+00)]')
        self.assertEqual(str(Hphi_list), '[(x, x, 2.00e+00), (x, y, 2.00e+00), (y, y, 2.00e+00)]')

        self.assertEqual(str(A_list), '[(0, x, 1.0), (0, y, 1.0), (1, x, 3.0), (1, s, -1.0)]')
        self.assertEqual(str(b_list), '[3.0, 10.0]')

        self.assertEqual(str(f_list), '[x*y - 2.00e+01, sin(x) - 3.00e+00 - s, cos(y) - 4.00e+00 - s]')
        self.assertEqual(str(J_list), '[(0, x, y), (0, y, x), (1, x, cos(x)), (1, s, -1.00e+00), (2, y, -sin(y)), (2, s, -1.00e+00)]')
        self.assertEqual(str(H_list), '[[(x, y, 1.00e+00)], [(x, x, -sin(x))], [(y, y, -1.00e+00*cos(y))]]')

        self.assertEqual(str(u_list), '[(x, 10.0), (s, 0), (s, 0)]')
        self.assertEqual(str(l_list), '[(y, 19.0), (s, 0)]')

    def test_std_problem(self):

        np.random.seed(0)

        x = optmod.variable.VariableScalar('x', value=3.)
        y = optmod.variable.VariableScalar('y', value=4.)
        
        f = x*x + 2*x*y + y*y
        c0 = x <= 10
        c1 = x + y == 3
        c2 = 3*x <= 10           # slack s1
        c3 = y >= 19.
        c4 = x*y == 20.
        c5 = optmod.sin(x) <= 3  # slack s2
        c6 = optmod.cos(y) >= 4. # slack s3

        p = optmod.Problem(minimize(f), [c0, c1, c2, c3, c4, c5, c6])

        std_prob = p.__get_std_problem__()
        
        # vars
        self.assertEqual(len(std_prob.var2index), 5)
        self.assertEqual(len(std_prob.index2var), 5)
        index_x = std_prob.var2index[x]
        index_y = std_prob.var2index[y]
        index_s1 = std_prob.var2index[c2.slack]
        index_s2 = std_prob.var2index[c5.slack]
        index_s3 = std_prob.var2index[c6.slack]

        # x
        self.assertEqual(std_prob.x.size, 5)
        temp = np.zeros(5)
        temp[index_x] = 3.
        temp[index_y] = 4.
        self.assertTrue(np.all(std_prob.x == temp))

        # A, b
        self.assertTupleEqual(std_prob.A.shape, (2, 5))
        self.assertEqual(std_prob.A.nnz, 4)
        self.assertTrue(np.all(std_prob.A.row == np.array([0, 0, 1, 1])))
        self.assertTrue(np.all(std_prob.A.col == np.array([index_x, index_y, index_x, index_s1])))
        self.assertTrue(np.all(std_prob.A.data == np.array([1, 1, 3, -1])))
        self.assertTrue(np.all(std_prob.b == np.array([3., 10.])))

        # u, l
        inf = 1e8
        self.assertEqual(std_prob.u.size, 5)
        self.assertEqual(std_prob.l.size, 5)
        temp = np.zeros(5)
        temp[index_x] = 10.
        temp[index_y] = inf
        temp[index_s1] = 0
        temp[index_s2] = 0
        temp[index_s3] = inf
        self.assertTrue(np.all(std_prob.u == temp))
        temp = np.zeros(5)
        temp[index_x] = -inf
        temp[index_y] = 19
        temp[index_s1] = -inf
        temp[index_s2] = -inf
        temp[index_s3] = 0
        self.assertTrue(np.all(std_prob.l == temp))

        # integer flags
        self.assertTrue(isinstance(std_prob.P, np.ndarray))
        self.assertEqual(std_prob.P.size, 5)
        self.assertNotEqual(std_prob.P.dtype, int)
        self.assertEqual(std_prob.P.dtype, bool)
        self.assertTrue(np.all(std_prob.P == False))
        self.assertFalse(np.any(std_prob.P == True))

        # phi, gphi, Hphi before eval
        self.assertEqual(std_prob.phi, 0.)
        self.assertTrue(np.all(std_prob.gphi == np.zeros(5)))
        self.assertTupleEqual(std_prob.Hphi.shape, (5, 5))
        self.assertEqual(std_prob.Hphi.nnz, 3)
        if index_x <= index_y:
            self.assertTrue(np.all(std_prob.Hphi.row == np.array([index_x, index_y, index_y])))
            self.assertTrue(np.all(std_prob.Hphi.col == np.array([index_x, index_x, index_y])))
        else:
            self.assertTrue(np.all(std_prob.Hphi.row == np.array([index_x, index_x, index_y])))
            self.assertTrue(np.all(std_prob.Hphi.col == np.array([index_x, index_y, index_y])))
        self.assertTrue(np.all(std_prob.Hphi.data == np.zeros(3)))

        # f, J, H_combined before eval
        self.assertEqual(std_prob.f.size, 3)
        self.assertTrue(np.all(std_prob.f == np.zeros(3)))
        self.assertTupleEqual(std_prob.J.shape, (3, 5))
        self.assertEqual(std_prob.J.nnz, 6)
        self.assertTrue(np.all(std_prob.J.row == np.array([0, 0, 1, 1, 2, 2])))
        self.assertTrue(np.all(std_prob.J.col == np.array([index_x, index_y, index_x, index_s2, index_y, index_s3])))
        self.assertTrue(np.all(std_prob.J.data == np.zeros(6)))
        self.assertEqual(std_prob.H_combined.nnz, 3)
        if index_x <= index_y:
            self.assertTrue(np.all(std_prob.H_combined.row == np.array([index_y, index_x, index_y])))
            self.assertTrue(np.all(std_prob.H_combined.col == np.array([index_x, index_x, index_y])))
        else:
            self.assertTrue(np.all(std_prob.H_combined.row == np.array([index_x, index_x, index_y])))
            self.assertTrue(np.all(std_prob.H_combined.col == np.array([index_y, index_x, index_y])))
        self.assertTrue(np.all(std_prob.H_combined.data == np.zeros(3)))
        
        var = np.random.randn(5)

        # Eval
        std_prob.eval(var)

        # phi, gphi, Hphi after eval
        self.assertAlmostEqual(std_prob.phi, (var[index_x]*var[index_x] +
                                              2.*var[index_x]*var[index_y] +
                                              var[index_y]*var[index_y]))
        temp = np.zeros(5)
        temp[index_x] = 2.*var[index_x] + 2.*var[index_y]
        temp[index_y] = 2.*var[index_x] + 2.*var[index_y]
        self.assertTrue(norm(std_prob.gphi - temp) < 1e-8)
        self.assertTrue(np.all(std_prob.Hphi.data == np.array([2., 2., 2.])))

        # f, J after eval
        self.assertTrue(norm(std_prob.f - np.array([var[index_x]*var[index_y]-20.,
                                                    np.sin(var[index_x])-var[index_s2]-3.,
                                                    np.cos(var[index_y])-var[index_s3]-4.])) < 1e-8)
        self.assertTrue(np.all(std_prob.J.data == np.array([var[index_y],
                                                            var[index_x],
                                                            np.cos(var[index_x]),
                                                            -1.,
                                                            -np.sin(var[index_y]),
                                                            -1.])))

        # Combine H - ones
        std_prob.combine_H(np.ones(3))
        self.assertTrue(np.all(std_prob.H_combined.data == np.array([1.,
                                                                     -np.sin(var[index_x]),
                                                                     -np.cos(var[index_y])])))

        # Combine H - rand
        lam = np.random.randn(3)
        std_prob.combine_H(lam)
        self.assertTrue(np.all(std_prob.H_combined.data == np.array([1.*lam[0],
                                                                     -np.sin(var[index_x])*lam[1],
                                                                     -np.cos(var[index_y])*lam[2]])))

        # Properties
        self.assertTrue(len(std_prob.properties), 3)
        self.assertTrue('continuous' in std_prob.properties)
        self.assertTrue('optimization' in std_prob.properties)
        self.assertTrue('nonlinear' in std_prob.properties)        

    def test_solve_LP(self):

        x = optmod.VariableScalar('x')
        y = optmod.VariableScalar('y')

        # Problem
        p = optmod.Problem(objective=maximize(-2*x+5*y),
                           constraints=[100 <= x,
                                        x <= 200,
                                        80 <= y,
                                        y <= 170,
                                        y >= -x + 200])

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['linear', 'continuous', 'optimization'])
                
        info = p.solve(solver='inlp', parameters={'quiet': True})

        self.assertTrue('iterations' in info)
        self.assertTrue('time_problem' in info)
        self.assertTrue('time_solver' in info)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

        info = p.solve(solver='augl', parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

    def test_solve_LP_clp(self):

        x = optmod.VariableScalar('x')
        y = optmod.VariableScalar('y')

        # Problem
        p = optmod.Problem(objective=maximize(-2*x+5*y),
                           constraints=[100 <= x,
                                        x <= 200,
                                        80 <= y,
                                        y <= 170,
                                        y >= -x + 200])

        try:
            info = p.solve(solver='clp', parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

    def test_solve_LP_clp_cmd(self):

        x = optmod.VariableScalar('x')
        y = optmod.VariableScalar('y')

        # Problem
        p = optmod.Problem(objective=maximize(-2*x+5*y),
                           constraints=[100 <= x,
                                        x <= 200,
                                        80 <= y,
                                        y <= 170,
                                        y >= -x + 200])

        try:
            info = p.solve(solver='clp_cmd', parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

    def test_solve_QP(self):

        x = optmod.VariableScalar('x')
        y = optmod.VariableScalar('y')

        f = 3*x*x+y*y+2*x*y+x+6*y+2

        # Problem
        p = optmod.Problem(objective=minimize(f))

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])
                
        info = p.solve(solver='inlp', parameters={'quiet': True})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 1.25, places=4)
        self.assertAlmostEqual(y.get_value(), -4.25, places=4)

        info = p.solve(solver='augl', parameters={'quiet': True})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 1.25, places=4)
        self.assertAlmostEqual(y.get_value(), -4.25, places=4)

        # Problem
        p = optmod.Problem(objective=minimize(f),
                           constraints=[2*x+3*y >= 4,
                                        x >= 0,
                                        y >= 0])

        info = p.solve(solver='inlp', parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.5, places=4)
        self.assertAlmostEqual(y.get_value(), 1, places=4)
        self.assertAlmostEqual(f.get_value(), 11.25, places=4)

        info = p.solve(solver='augl', parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.5, places=4)
        self.assertAlmostEqual(y.get_value(), 1, places=4)
        self.assertAlmostEqual(f.get_value(), 11.25, places=4)

    def test_solve_NLP_unconstrained(self):

        n = 5

        x = optmod.VariableMatrix(name='x', value=[1.3, 0.7, 0.8, 1.9, 1.2], shape=(5,1))

        f = 0.
        for i in range(0, n-1):
            f = f + 100*(x[i+1,0]-x[i,0]*x[i,0])*(x[i+1,0]-x[i,0]*x[i,0]) + (1.-x[i,0])*(1.-x[i,0])

        # Problem
        p = optmod.Problem(minimize(f))

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])

        try:
            info = p.solve(solver='ipopt', parameters={'quiet': True}, fast_evaluator=True)
        except ImportError:
            raise unittest.SkipTest('ipopt not available')
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 0, places=4)
        self.assertLess(norm(x.get_value()-np.ones((5,1)), np.inf), 1e-2)

        x.set_value(np.matrix([1.3, 0.7, 0.8, 1.9, 1.2]).T)

        info = p.solve(solver='ipopt', parameters={'quiet': True}, fast_evaluator=False)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 0, places=4)
        self.assertLess(norm(x.get_value()-np.ones((5,1)), np.inf), 1e-2)
        
    def test_solve_NLP_constrained(self):

        # Hock-Schittkowski
        # Problem 71
        
        x1 = optmod.VariableScalar('x1', value=1)
        x2 = optmod.VariableScalar('x2', value=5)
        x3 = optmod.VariableScalar('x3', value=5)
        x4 = optmod.VariableScalar('x4', value=1)

        f = x1*x4*(x1+x2+x3) + x3
        
        constraints = [x1*x2*x3*x4 >= 25,
                       x1*x1 + x2*x2 + x3*x3 + x4*x4 == 40,
                       1 <= x1, x1 <= 5,
                       1 <= x2, x2 <= 5,
                       1 <= x3, x3 <= 5,
                       1 <= x4, x4 <= 5]

        p = optmod.Problem(minimize(f), constraints=constraints)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])

        try:
            info = p.solve(solver='ipopt', parameters={'quiet': True}, fast_evaluator=True)
        except ImportError:
            raise unittest.SkipTest('ipopt not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 17.0140173, places=3)
        self.assertAlmostEqual(x1.get_value(), 1., places=3)
        self.assertAlmostEqual(x2.get_value(), 4.7429994, places=3)
        self.assertAlmostEqual(x3.get_value(), 3.8211503, places=3)
        self.assertAlmostEqual(x4.get_value(), 1.3794082, places=3)
        
        x1.set_value(1.)
        x2.set_value(5.)
        x3.set_value(5.)
        x4.set_value(1.)

        info = p.solve(solver='ipopt', parameters={'quiet': True}, fast_evaluator=False)
    
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 17.0140173, places=3)
        self.assertAlmostEqual(x1.get_value(), 1., places=3)
        self.assertAlmostEqual(x2.get_value(), 4.7429994, places=3)
        self.assertAlmostEqual(x3.get_value(), 3.8211503, places=3)
        self.assertAlmostEqual(x4.get_value(), 1.3794082, places=3)

    def test_solve_MILP_cbc(self):

        x1 = optmod.VariableScalar('x1', type='integer')
        x2 = optmod.VariableScalar('x2', type='integer')
        x3 = optmod.VariableScalar('x3')
        x4 = optmod.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x4 >= 0,
                  x3 <= 0]
        
        p = optmod.Problem(minimize(obj), constr)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['linear', 'integer', 'optimization'])

        try:
            self.assertRaises(TypeError, p.solve, 'ipopt')
        except ImportError:
            raise unittest.SkipTest('ipopt not available')
        self.assertRaises(TypeError, p.solve, 'augl')
        try:
            self.assertRaises(TypeError, p.solve, 'clp')
        except ImportError:
            raise unittest.SkipTest('clp not available')
        self.assertRaises(TypeError, p.solve, 'inlp')

        try:
            info = p.solve('cbc', parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc not available')

        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 1.)
        self.assertEqual(x2.get_value(), 2.)

        x1.type = 'continuous'
        x2.type = 'continuous'

        try:
            info = p.solve('cbc', parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc not available')
        
        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 4.)
        self.assertEqual(x2.get_value(), 4.5)

    def test_solve_MILP_cbc_cmd(self):

        x1 = optmod.VariableScalar('x1', type='integer')
        x2 = optmod.VariableScalar('x2', type='integer')
        x3 = optmod.VariableScalar('x3')
        x4 = optmod.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x4 >= 0,
                  x3 <= 0]
        
        p = optmod.Problem(minimize(obj), constr)

        try:
            info = p.solve('cbc_cmd', parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 1.)
        self.assertEqual(x2.get_value(), 2.)
        
    def test_solve_feasibility(self):

        x = optmod.VariableScalar('x', value=1.)

        constr = [x*optmod.cos(x)-x*x == 0]
        
        p = optmod.Problem(EmptyObjective(), constr)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'feasibility'])

        try:
            self.assertRaises(TypeError, p.solve, 'clp')
        except ImportError:
            raise unittest.SkipTest('clp not available')
        try:
            self.assertRaises(TypeError, p.solve, 'cbc')
        except ImportError:
            raise unittest.SkipTest('cbc not available')

        info = p.solve('nr', parameters={'quiet': True, 'feastol': 1e-10})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.739085133215161, places=7)

        x.value = 1.
        self.assertNotAlmostEqual(x.get_value(), 0.739085133215161, places=7)

        info = p.solve('nr', parameters={'quiet': True, 'feastol': 1e-10}, fast_evaluator=False)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.739085133215161, places=7)
