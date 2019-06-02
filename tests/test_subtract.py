import optmod
import unittest
import numpy as np

class TestSubtract(unittest.TestCase):

    def test_contruction(self):
        
        x = optmod.variable.VariableScalar(name='x')

        f = x-1
        self.assertEqual(f.name, 'add')
        self.assertEqual(len(f.arguments), 2)
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(isinstance(f.arguments[1], optmod.constant.Constant))
        self.assertEqual(f.arguments[1].get_value(), -1.)

    def test_constant_constant(self):

        a = optmod.constant.Constant(4.)
        b = optmod.constant.Constant(5.)

        f = a - b
        self.assertTrue(f.is_constant())
        self.assertEqual(f.get_value(), -1)

    def test_scalar_scalar(self):

        rn = optmod.utils.repr_number

        x = optmod.variable.VariableScalar(name='x', value=2.)
        y = optmod.variable.VariableScalar(name='y', value=3.)

        f = x - 1.
        self.assertTrue(isinstance(f, optmod.function.add))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(isinstance(f.arguments[1], optmod.constant.Constant))
        self.assertEqual(f.arguments[1].get_value(), -1.)
        self.assertEqual(f.get_value(), 1.)
        self.assertEqual(str(f), 'x + %s' %rn(-1.))
        
        f = 1. - x
        self.assertTrue(isinstance(f, optmod.function.add))
        self.assertTrue(f.arguments[1].is_function())
        self.assertTrue(isinstance(f.arguments[0], optmod.constant.Constant))
        self.assertEqual(f.arguments[1].get_value(), -2)
        self.assertEqual(f.get_value(), -1.)
        self.assertEqual(str(f), '%s + x*%s' %(rn(1.), rn(-1.)))

        f = x - y
        self.assertTrue(isinstance(f, optmod.function.add))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_function())
        self.assertEqual(f.get_value(), -1.)
        self.assertEqual(str(f), 'x + y*%s' %rn(-1.))

        f = 3. - x - y
        self.assertTrue(isinstance(f, optmod.function.add))
        self.assertTrue(isinstance(f.arguments[1], optmod.function.multiply))
        self.assertTrue(isinstance(f.arguments[2], optmod.function.multiply))
        self.assertTrue(f.arguments[0].is_constant())
        self.assertEqual(f.arguments[0].get_value(), 3)
        self.assertEqual(f.get_value(), -2.)
        self.assertEqual(str(f), '%s + x*%s + y*%s' %(rn(3.), rn(-1.), rn(-1.)))

    def test_scalar_matrix(self):

        rn = optmod.utils.repr_number
        
        value = [[1., 2., 3.], [4., 5., 6.]]
        x = optmod.variable.VariableScalar(name='x', value=2.)
        y = optmod.variable.VariableMatrix(name='y', value=value)
        r = np.random.random((2,3))
        
        f = x - r
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
                self.assertTrue(fij.arguments[0] is x)
                self.assertEqual(fij.arguments[1].get_value(), -r[i,j])
        self.assertTrue(isinstance(f.get_value(), np.matrix))
        self.assertTrue(np.all(f.get_value() == 2. - r))
        self.assertEqual(str(f),
                         ('[[ x + %s, x + %s, x + %s ],\n' %(rn(-r[0,0]), rn(-r[0,1]), rn(-r[0,2])) +
                          ' [ x + %s, x + %s, x + %s ]]\n' %(rn(-r[1,0]), rn(-r[1,1]), rn(-r[1,2]))))

        f = r - x
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
                self.assertTrue(fij.arguments[0].is_constant())
                self.assertEqual(fij.arguments[0].get_value(), r[i,j])
        self.assertTrue(isinstance(f.get_value(), np.matrix))
        self.assertTrue(np.all(f.get_value() == r - 2.))

        f = x - np.matrix(r)
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTrue(np.all(f.get_value() == 2. - r))

        f = np.matrix(r) - x
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTrue(np.all(f.get_value() == r - 2.))
                         
        f = y - 1
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
                self.assertTrue(fij.arguments[0] is y[i,j])
                self.assertEqual(fij.arguments[1].get_value(), -1.)
        self.assertTrue(isinstance(f.get_value(), np.matrix))
        self.assertTrue(np.all(f.get_value() == np.array(value) - 1))
        self.assertEqual(str(f),
                         ('[[ y[0,0] + %s, y[0,1] + %s, y[0,2] + %s ],\n' %(rn(-1), rn(-1), rn(-1)) +
                          ' [ y[1,0] + %s, y[1,1] + %s, y[1,2] + %s ]]\n' %(rn(-1), rn(-1), rn(-1))))

        f = 1 - y
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTrue(np.all(f.get_value() == 1. - np.array(value)))

        f = x - y
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
                self.assertTrue(fij.arguments[1].is_function())
                self.assertTrue(fij.arguments[0] is x)
        self.assertTrue(isinstance(f.get_value(), np.matrix))
        self.assertTrue(np.all(f.get_value() == 2. - np.array(value)))
        self.assertEqual(str(f),
                         ('[[ x + y[0,0]*%s, x + y[0,1]*%s, x + y[0,2]*%s ],\n' %(rn(-1.), rn(-1.), rn(-1.)) +
                          ' [ x + y[1,0]*%s, x + y[1,1]*%s, x + y[1,2]*%s ]]\n' %(rn(-1.), rn(-1.), rn(-1.))))

        f = y - x
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.function.add))
                self.assertTrue(fij.arguments[0] is y[i,j])
                self.assertTrue(fij.arguments[1].is_function())
        self.assertTrue(isinstance(f.get_value(), np.matrix))
        self.assertTrue(np.all(f.get_value() == np.array(value) - 2.))
        self.assertEqual(str(f),
                         ('[[ y[0,0] + x*%s, y[0,1] + x*%s, y[0,2] + x*%s ],\n' %(rn(-1.), rn(-1.), rn(-1.)) +
                          ' [ y[1,0] + x*%s, y[1,1] + x*%s, y[1,2] + x*%s ]]\n' %(rn(-1.), rn(-1.), rn(-1.))))

        f = (y - 1) - (3 - x)
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                self.assertEqual(str(f[i,j]), 'y[%d,%d] + %s + %s + x' %(i, j, rn(-1), rn(-3)))
                self.assertTrue(f[i,j].is_function())
                self.assertEqual(len(f[i,j].arguments), 4)
        self.assertTrue(np.all(f.get_value() == (np.array(value) - 1.) - (3. - 2.)))

    def test_matrix_matrix(self):

        rn = optmod.utils.repr_number
        
        value1 = [[1., 2., 3.], [4., 5., 6.]]
        value2 = np.random.random((2,3))
        x = optmod.variable.VariableMatrix(name='x', value=value1)
        y = optmod.variable.VariableMatrix(name='y', value=value2)

        f = x - value2
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (2,3))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertEqual(str(fij), 'x[%d,%d] + %s' %(i, j, rn(-value2[i,j])))
        self.assertTrue(np.all(f.get_value() == np.matrix(value1) - value2))
                
        f = value2 - x
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (2,3))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertEqual(str(fij), '%s + x[%d,%d]*%s' %(rn(value2[i,j]), i,j, rn(-1)))
        self.assertTrue(np.all(f.get_value() == value2 - np.matrix(value1)))

        f = x - y
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (2,3))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertEqual(str(fij), 'x[%d,%d] + y[%d,%d]*%s' %(i, j, i, j, rn(-1.)))
        self.assertTrue(np.all(f.get_value() == np.matrix(value1) - value2))

        f = y - x
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        self.assertTupleEqual(f.shape, (2,3))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertEqual(str(fij), 'y[%d,%d] + x[%d,%d]*%s' %(i, j, i, j, rn(-1.)))
        self.assertTrue(np.all(f.get_value() == value2 - np.matrix(value1)))

    def test_zero(self):

        x = optmod.variable.VariableScalar(name='x', value=3.)
        
        f = x - 0
        self.assertTrue(f is x)

        f = 0 - x
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertEqual(str(f), 'x*-1.00e+00')

    def test_analyze(self):

        x = optmod.variable.VariableScalar('x')
        y = optmod.variable.VariableScalar('y')

        f = x - 1
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], -1.)
        self.assertEqual(len(prop['a']), 1)
        self.assertEqual(prop['a'][x], 1.)

        f = 2 - x
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], 2.)
        self.assertEqual(len(prop['a']), 1)
        self.assertEqual(prop['a'][x], -1.)

        f = x - y - x
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], 0.)
        self.assertEqual(len(prop['a']), 2)
        self.assertEqual(prop['a'][x], 0.)
        self.assertEqual(prop['a'][y], -1.)

        f = x - y - 10. - x
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], -10.)
        self.assertEqual(len(prop['a']), 2)
        self.assertEqual(prop['a'][x], 0)
        self.assertEqual(prop['a'][y], -1.)

    def test_derivative(self):

        x = optmod.variable.VariableScalar(name='x', value=3.)
        y = optmod.variable.VariableScalar(name='y', value=4.)

        f = x - 1
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)
        self.assertTrue(isinstance(fx, optmod.constant.Constant))
        self.assertEqual(fx.get_value(), 1.)
        self.assertTrue(isinstance(fy, optmod.constant.Constant))
        self.assertEqual(fy.get_value(), 0.)
        
        f = x - y        
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)
        self.assertTrue(isinstance(fx, optmod.constant.Constant))
        self.assertEqual(fx.get_value(), 1.)
        self.assertTrue(isinstance(fy, optmod.constant.Constant))
        self.assertEqual(fy.get_value(), -1.)

        f = (x - 1) - (x - 3) - (y - (x - 5.))
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)        
        self.assertTrue(isinstance(fx, optmod.constant.Constant))
        self.assertEqual(fx.get_value(), 1.)
        self.assertTrue(isinstance(fy, optmod.constant.Constant))
        self.assertEqual(fy.get_value(), -1.)        
