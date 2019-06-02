import optmod
import unittest
import numpy as np

class TestMultiply(unittest.TestCase):

    def test_contruction(self):

        x = optmod.variable.VariableScalar(name='x')
        f = optmod.function.multiply([x, optmod.expression.make_Expression(1.)])
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertEqual(f.name, 'multiply')
        self.assertEqual(len(f.arguments), 2)
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_constant())
        self.assertEqual(f.arguments[1].get_value(), 1.)

        self.assertRaises(AssertionError, optmod.function.multiply, [1., x, 2.])
        self.assertRaises(AssertionError, optmod.function.multiply, [x])
        self.assertRaises(TypeError, optmod.function.multiply, x)

    def test_constant(self):

        a = optmod.constant.Constant(4.)
        b = optmod.constant.Constant(5.)

        f = a*b
        self.assertTrue(f.is_constant(20.))
        
    def test_scalar_scalar(self):
        
        rn = optmod.utils.repr_number
        
        x = optmod.variable.VariableScalar(name='x', value=2.)
        y = optmod.variable.VariableScalar(name='y', value=3.)

        f = x*2
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_constant())
        self.assertEqual(f.get_value(), 4.)
        self.assertEqual(str(f), 'x*%s' %rn(2.))

        f = 2.*x
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_constant())
        self.assertEqual(f.get_value(), 4.)
        self.assertEqual(str(f), 'x*%s' %rn(2.))

        f = x*y
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1] is y)
        self.assertEqual(f.get_value(), 6)
        self.assertEqual(str(f), 'x*y')

        f = x*(y+3.)
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_function())
        self.assertEqual(f.get_value(), 12)
        self.assertEqual(str(f), 'x*(y + %s)' %rn(3.))

        f = (1-y)*x
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0].is_function())
        self.assertTrue(f.arguments[1] is x)
        self.assertEqual(f.get_value(), -4)
        self.assertEqual(str(f), '(%s + y*%s)*x' %(rn(1.), rn(-1.)))

        f = (4.*x)*(3*y)
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0].is_function())
        self.assertTrue(f.arguments[1].is_function())
        self.assertEqual(f.get_value(), 72)
        self.assertEqual(str(f), 'x*%s*y*%s' %(rn(4), rn(3)))

        f = -x*5
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[1].is_constant())
        self.assertTrue(f.arguments[0].is_variable())
        self.assertEqual(str(f), 'x*%s' %rn(-5))
        self.assertEqual(f.get_value(), -10.)

        f = y*-x
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is y)
        self.assertTrue(f.arguments[1].is_function())
        self.assertEqual(str(f), 'y*x*%s' %rn(-1))
        self.assertEqual(f.get_value(), -6.)

        f = optmod.sin(x)*y
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0].is_function())
        self.assertTrue(f.arguments[1] is y)
        self.assertEqual(str(f), 'sin(x)*y')
        self.assertEqual(f.get_value(), np.sin(2.)*3.)

        f = x*optmod.sin(y)
        self.assertTrue(isinstance(f, optmod.function.multiply))
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(f.arguments[1].is_function())
        self.assertEqual(str(f), 'x*sin(y)')
        self.assertEqual(f.get_value(), np.sin(3.)*2.)
        
    def test_scalar_matrix(self):

        rn = optmod.utils.repr_number
        
        value = [[1., 2., 3.], [4., 5., 6.]]
        x = optmod.variable.VariableScalar(name='x', value=2.)
        y = optmod.variable.VariableMatrix(name='y', value=value)
        r = np.random.random((2,3))
        
    def test_matrix_matrix(self):

        pass

    def test_one(self):

        x = optmod.variable.VariableScalar(name='x', value=3.)
        
    def test_derivative(self):

        rn = optmod.utils.repr_number

        x = optmod.variable.VariableScalar(name='x', value=3.)
        y = optmod.variable.VariableScalar(name='y', value=4.)
        z = optmod.variable.VariableScalar(name='z', value=5.)

        f = x*x
        fx = f.get_derivative(x)
        self.assertEqual(fx.get_value(), 2.*3.)
        self.assertEqual(str(fx), 'x + x')

        f = x*y
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)
        fz = f.get_derivative(z)
        self.assertTrue(fx is y)
        self.assertTrue(fy is x)
        self.assertTrue(fz.is_constant())
        self.assertEqual(fz.get_value(), 0)

        f = x*y*z
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)
        fz = f.get_derivative(z)

        self.assertEqual(str(fx), 'z*y')
        self.assertEqual(fx.get_value(), 20.)
        self.assertEqual(str(fy), 'z*x')
        self.assertEqual(fy.get_value(), 15.)
        self.assertEqual(str(fz), 'x*y')
        self.assertEqual(fz.get_value(), 12.)

    def test_analyze(self):

        x = optmod.variable.VariableScalar(name='x', value=3.)
        y = optmod.variable.VariableScalar(name='y', value=4.)
        z = optmod.variable.VariableScalar(name='z', value=5.)

        f = 3*x
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], 0.)
        self.assertEqual(len(prop['a']), 1)
        self.assertEqual(prop['a'][x], 3.)

        f = x*7
        prop = f.__analyze__()
        self.assertTrue(prop['affine'])
        self.assertEqual(prop['b'], 0.)
        self.assertEqual(len(prop['a']), 1)
        self.assertEqual(prop['a'][x], 7.)

        f = y*x
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertEqual(prop['b'], 0.)
        self.assertEqual(len(prop['a']), 2)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])

        f = y*x*z
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertEqual(prop['b'], 0.)
        self.assertEqual(len(prop['a']), 3)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])
        self.assertTrue(z in prop['a'])        

    def test_std_components(self):

        x = optmod.variable.VariableScalar(name='x', value=3.)
        y = optmod.variable.VariableScalar(name='y', value=4.)
        z = optmod.variable.VariableScalar(name='z', value=5.)

        f = x*y
        comp = f.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']

        self.assertTrue(phi is f)

        self.assertEqual(len(gphi_list), 2)

        v, exp = gphi_list[0]
        self.assertTrue(v is x)
        self.assertTrue(exp is y)
        
        v, exp = gphi_list[1]
        self.assertTrue(v is y)
        self.assertTrue(exp is x)

        self.assertEqual(len(Hphi_list), 1)

        v1, v2, exp = Hphi_list[0]
        self.assertTrue(v1 is x)
        self.assertTrue(v2 is y)
        self.assertTrue(exp.is_constant(1.))

        f = x*x
        comp = f.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']

        self.assertTrue(phi is f)

        self.assertEqual(len(gphi_list), 1)

        v, exp = gphi_list[0]
        self.assertTrue(v is x)
        self.assertTrue(str(exp), 'x + x')
        
        self.assertEqual(len(Hphi_list), 1)

        v1, v2, exp = Hphi_list[0]
        self.assertTrue(v1 is x)
        self.assertTrue(v2 is x)
        self.assertTrue(exp.is_constant(2.))
        
        
