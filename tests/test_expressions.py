import time
import optmod
import unittest
import numpy as np

class TestExpressions(unittest.TestCase):

    def test_get_variables(self):

        x = optmod.Variable(name='x', shape=(2,3))
        y = optmod.Variable(name='y')

        f = optmod.sin(x*3) + optmod.cos(y+10.)*x

        vars = f.get_variables()
        self.assertEqual(len(vars), 7)

        self.assertSetEqual(f.get_variables(),
                            set([x[i,j] for i in range(2) for j in range(3)]+[y]))

    def test_scalar_get_fast_evaluator(self):

        x = optmod.Variable(name='x', value=2.)
        y = optmod.Variable(name='y', value=3.)

        f = 3*(x+3)+optmod.sin(y+4*x)

        e = f.get_fast_evaluator([x,y])
        
        self.assertTrue(isinstance(e, optmod.coptmod.Evaluator))
        self.assertEqual(e.get_value(), 0.)
        e.eval([4.,-3.])        
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

    def test_matrix_get_fast_evalulator(self):

        r = np.random.randn(4,3)
        xval = np.random.randn(4,3)
        x = optmod.Variable(name='x', value=xval)
        y = optmod.Variable(name='y', value=10.)

        self.assertTupleEqual(x.shape, (4,3))

        f = optmod.sin(3*x + 10.)*optmod.cos(y - optmod.sum(x*y))

        self.assertTupleEqual(f.shape, (4,3))

        variables = np.asarray(x.data).flatten().tolist()+[y]

        e = f.get_fast_evaluator(variables)

        val = e.get_value()

        self.assertTupleEqual(val.shape, (4,3))
        self.assertTrue(np.all(val == 0))

        e.eval(np.array(r.flatten().tolist()+[13.]))

        val = e.get_value()
        val1 = np.sin(3*r+10.)*np.cos(13.-np.sum(r*13.))

        self.assertTupleEqual(val.shape, (4,3))
        self.assertLess(np.linalg.norm(val-val1), 1e-10)

        x = np.array(xval.flatten().tolist()+[10.])
        e.eval(x)

        self.assertTrue(np.all(e.get_value() == f.get_value()))

        t0 = time.time()
        for i in range(500):
            f.get_value()
        t1 = time.time()
        for i in range(500):
            e.eval(x)
        t2 = time.time()
        self.assertGreater((t1-t0)/(t2-t1), 400.)
