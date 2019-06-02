import optmod
import unittest
import numpy as np

class TestCos(unittest.TestCase):

    def test_contruction(self):

        x = optmod.variable.VariableScalar(name='x')
        f = optmod.cos(x)
        self.assertTrue(isinstance(f, optmod.cos))
        self.assertEqual(f.name, 'cos')
        self.assertEqual(len(f.arguments), 1)
        self.assertTrue(f.arguments[0] is x)

        self.assertRaises(TypeError, optmod.cos, [x, 1., 2.])

    def test_constant(self):

        a = optmod.constant.Constant(4.)
        
        f = optmod.cos(a)
        self.assertTrue(f.is_constant())
        self.assertEqual(f.get_value(), np.cos(4))

    def test_scalar(self):
        
        x = optmod.variable.VariableScalar(name='x', value=2.)

        f = optmod.cos(x)
        self.assertTrue(isinstance(f, optmod.cos))
        self.assertEqual(f.name, 'cos')
        self.assertEqual(len(f.arguments), 1)
        self.assertTrue(f.arguments[0] is x)
        
        self.assertEqual(f.get_value(), np.cos(2.))
        self.assertEqual(str(f), 'cos(x)')

    def test_matrix(self):

        value = np.random.randn(2,3)
        x = optmod.variable.VariableMatrix(name='x', value=value)

        f = optmod.cos(x)
        self.assertTrue(isinstance(f, optmod.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, optmod.cos))
                self.assertEqual(len(fij.arguments), 1)
                self.assertTrue(fij.arguments[0] is x[i,j])
                self.assertEqual(fij.get_value(), np.cos(value[i,j]))
                self.assertEqual(str(fij), 'cos(x[%d,%d])' %(i,j))

    def test_function(self):

        rn = optmod.utils.repr_number

        x = optmod.variable.VariableScalar(name='x', value=3.)

        f = optmod.cos(3*x)
        self.assertEqual(f.get_value(), np.cos(3.*3.))
        self.assertEqual(str(f), 'cos(x*%s)' %rn(3.))

        f = optmod.cos(x + 1)
        self.assertEqual(f.get_value(), np.cos(3.+1.))
        self.assertEqual(str(f), 'cos(x + %s)' %rn(1.))

        f = optmod.cos(optmod.cos(x))
        self.assertEqual(f.get_value(), np.cos(np.cos(3.)))
        self.assertEqual(str(f), 'cos(cos(x))')

    def test_analyze(self):

        x = optmod.variable.VariableScalar(name='x')
        y = optmod.variable.VariableScalar(name='y')

        f = optmod.cos(x)
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 1.)
        self.assertTrue(x in prop['a'])

        f = optmod.cos(-4.*(-y+3*x-2))
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 2.)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])

        f = optmod.cos((4.+x)*(-y+3*x-2))
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 2.)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])

    def test_derivative(self):

        x = optmod.variable.VariableScalar(name='x', value=2.)
        y = optmod.variable.VariableScalar(name='y', value=3.)
        
        f = optmod.cos(x)
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)

        self.assertTrue(isinstance(fx, optmod.function.Function))
        self.assertEqual(fx.get_value(), -np.sin(2.))
        self.assertTrue(isinstance(fy, optmod.constant.Constant))
        self.assertEqual(fy.get_value(), 0.)

        f = optmod.cos(x + optmod.cos(x*y))
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)
        
        self.assertTrue(fx.is_function())
        self.assertAlmostEqual(fx.get_value(), -np.sin(2.+np.cos(2.*3.))*(1.-np.sin(2.*3.)*3.))

        self.assertTrue(fy.is_function())
        self.assertAlmostEqual(fy.get_value(), -np.sin(2.+np.cos(2.*3.))*(0.-np.sin(2.*3.)*2.))

        f1 = optmod.cos(x + y*x)
        f1x = f1.get_derivative(x)
        f1y = f1.get_derivative(y)

        self.assertTrue(f1x.is_function())
        self.assertAlmostEqual(f1x.get_value(), -np.sin(2.+3.*2.)*(1. + 3.))
        self.assertAlmostEqual(f1y.get_value(), -np.sin(2.+3.*2.)*(0.+2.))

        f2 = f1*y
        f2x = f2.get_derivative(x)
        f2y = f2.get_derivative(y)

        self.assertTrue(f2x.is_function())
        self.assertAlmostEqual(f2x.get_value(), -np.sin(2.+3.*2.)*(1. + 3.)*3.)
        self.assertAlmostEqual(f2y.get_value(), -np.sin(2.+3.*2.)*3.*(0.+2.) + np.cos(2.+2.*3.))

        f3 = f1 + f2
        f3x = f3.get_derivative(x)
        self.assertAlmostEqual(f3.get_value(), f1.get_value()+f2.get_value())
        self.assertAlmostEqual(f3x.get_value(), f1x.get_value()+f2x.get_value())

    def test_std_components(self):

        x = optmod.variable.VariableScalar('x', value=2.)
        y = optmod.variable.VariableScalar('y', value=3.)

        f = optmod.cos(x + y*x)

        comp = f.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']

        self.assertTrue(phi is f)
        self.assertEqual(len(gphi_list), 2)

        v, exp = gphi_list[0]
        self.assertTrue(v is x)
        self.assertTrue(exp.is_function())
        self.assertEqual(exp.get_value(), -np.sin(2.+2.*3.)*(1.+3.))
        
        v, exp = gphi_list[1]
        self.assertTrue(v is y)
        self.assertTrue(exp.is_function())
        self.assertEqual(exp.get_value(), -np.sin(2.+2.*3.)*(0.+2.))
