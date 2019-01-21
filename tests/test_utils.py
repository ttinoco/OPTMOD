import optmod
import unittest
import numpy as np

class TestUtils(unittest.TestCase):

    def test_sum(self):

        r = np.random.randn(3,2)

        x = optmod.Variable('x', value=4.)
        y = optmod.Variable('y', value=r)

        self.assertTupleEqual(y.shape, (3,2))
        self.assertTrue(np.all(y.get_value() == r))

        # scalar
        f = optmod.sum(x)
        self.assertTrue(f is x)

        self.assertTrue(optmod.sum(x, axis=0) is x)
        self.assertRaises(np.core._internal.AxisError, optmod.sum, x, 1)
        
        # matrix
        f = optmod.sum(y)
        self.assertTrue(isinstance(f, optmod.expression.Expression))
        self.assertEqual(str(f), 'y[0,0] + y[0,1] + y[1,0] + y[1,1] + y[2,0] + y[2,1]')

        # matrix axis
        f = optmod.sum(y, axis=0)
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (1,2))
        self.assertEqual(str(f), ('[[ y[0,0] + y[1,0] + y[2,0],' +
                                  ' y[0,1] + y[1,1] + y[2,1] ]]\n'))

        # matrix axis
        f = optmod.sum(y, axis=1)
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (3,1))
        self.assertEqual(str(f), ('[[ y[0,0] + y[0,1] ],\n' +
                                  ' [ y[1,0] + y[1,1] ],\n' +
                                  ' [ y[2,0] + y[2,1] ]]\n'))
        
        self.assertRaises(np.core._internal.AxisError, optmod.sum, x, 2)