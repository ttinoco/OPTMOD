import time
import optmod
import unittest
import numpy as np

class TestExpressions(unittest.TestCase):

    def test_get_variables(self):
        
        x = optmod.VariableMatrix(name='x', shape=(2,3))
        y = optmod.VariableScalar(name='y')

        f = optmod.sin(x*3) + optmod.cos(y+10.)*x

        vars = f.get_variables()
        self.assertEqual(len(vars), 7)

        self.assertSetEqual(f.get_variables(),
                            set([x[i,j] for i in range(2) for j in range(3)]+[y]))

    def test_get_derivatives(self):

        x = optmod.VariableScalar(name='x', value=5.)
        y = optmod.VariableScalar(name='y', value=3.)
        z = optmod.VariableScalar(name='z', value=10.)

        f = 2*x + optmod.cos(x*y)

        d = f.get_derivatives([x,y,z])

        self.assertEqual(d[x].get_value(), 2 - np.sin(5*3.)*3)
        self.assertEqual(d[y].get_value(), -np.sin(5*3.)*5)
        self.assertEqual(d[z].get_value(), 0.)

    def test_scalar_get_fast_evaluator(self):

        x = optmod.VariableScalar(name='x', value=2.)
        y = optmod.VariableScalar(name='y', value=3.)
        
        f = 3*(x+3)+optmod.sin(y+4*x)

        e = f.get_fast_evaluator([x,y])
        
        self.assertTrue(isinstance(e, optmod.coptmod.Evaluator))
        self.assertEqual(e.get_value(), 0.)
        e.eval([4.,-3.])
        self.assertTrue(np.isscalar(e.get_value()))
        self.assertEqual(e.get_value(), 3.*(4+3.)+np.sin(-3.+4*4.))

        x = np.array([2.,3.])
        
        t0 = time.time()
        for i in range(50000):
            f.get_value()
        t1 = time.time()
        for i in range(50000):
            e.eval(x)
        t2 = time.time()
        self.assertGreater((t1-t0)/(t2-t1), 15.)

    def test_matrix_get_fast_evaluator(self):

        xval = np.random.randn(4,3)
        x = optmod.VariableMatrix(name='x', value=xval)
        y = optmod.VariableScalar(name='y', value=10.)

        self.assertTupleEqual(x.shape, (4,3))

        f = optmod.sin(3*x + 10.)*optmod.cos(y - optmod.sum(x*y))

        self.assertTupleEqual(f.shape, (4,3))

        variables = list(f.get_variables())
        self.assertEqual(len(variables), 13)

        e = f.get_fast_evaluator(variables)

        val = e.get_value()
        self.assertTrue(isinstance(val, np.matrix))

        self.assertTupleEqual(val.shape, (4,3))
        self.assertTrue(np.all(val == 0))

        e.eval(np.array([x.get_value() for x in variables]))

        val = e.get_value()
        val1 = np.sin(3*xval+10.)*np.cos(10.-np.sum(xval*10.))

        self.assertTupleEqual(val.shape, (4,3))
        self.assertLess(np.linalg.norm(val-val1), 1e-10)

        x = np.array([v.get_value() for v in variables])
        e.eval(x)

        self.assertLess(np.max(np.abs(e.get_value() - f.get_value())), 1e-10)

        t0 = time.time()
        for i in range(500):
            f.get_value()
        t1 = time.time()
        for i in range(500):
            e.eval(x)
        t2 = time.time()
        self.assertGreater((t1-t0)/(t2-t1), 400.)
