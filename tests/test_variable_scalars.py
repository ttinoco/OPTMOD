import optmod
import unittest
import numpy as np

class TestVariableScalars(unittest.TestCase):

    def test_construction(self):

        x = optmod.variable.VariableScalar()
        self.assertEqual(x.name, 'var')
        self.assertEqual(x.get_value(), 0.)
        self.assertTrue(isinstance(x.id, int))
        saved_id = x.id
        
        x = optmod.variable.VariableScalar(name='x', value=3)
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.get_value(), 3.)
        self.assertTrue(isinstance(x.get_value(), np.float64))
        self.assertTrue(isinstance(x.id, int))
        self.assertEqual(x.id, saved_id + 1)

        x = optmod.VariableScalar()
        self.assertTrue(isinstance(x, optmod.variable.VariableScalar))
        self.assertEqual(x.name, 'var')
        self.assertEqual(x.get_value(), 0.)
        self.assertEqual(x.type, 'continuous')
        
        x = optmod.VariableScalar(name='x', value=2., type='integer')
        self.assertTrue(isinstance(x, optmod.variable.VariableScalar))
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.get_value(), 2.)
        self.assertEqual(x.type, 'integer')
        self.assertTrue(x.is_integer())

        x = optmod.VariableScalar(type='integer')
        self.assertEqual(x.get_value(), 0.)
        self.assertEqual(x.name, 'var')
        self.assertTrue(x.is_integer())
        self.assertFalse(x.is_continuous())

        self.assertRaises(ValueError, optmod.VariableScalar, 'x', np.random.randn(2,3))
        
    def test_type(self):

        x = optmod.variable.VariableScalar()
        self.assertTrue(x.is_continuous())
        self.assertFalse(x.is_integer())

        x = optmod.variable.VariableScalar(type='continuous')
        self.assertTrue(x.is_continuous())
        self.assertFalse(x.is_integer())

        x = optmod.variable.VariableScalar(type='integer')
        self.assertFalse(x.is_continuous())
        self.assertTrue(x.is_integer())

        self.assertRaises(ValueError, optmod.variable.VariableScalar, 'var', 0., 'foo')

    def test_get_variables(self):

        x = optmod.VariableScalar(name='x')

        self.assertSetEqual(x.get_variables(), set([x]))
        
    def test_repr(self):

        x = optmod.variable.VariableScalar(name='x', value=3)
        s = str(x)
        self.assertEqual(s, 'x')

    def test_value(self):

        x = optmod.variable.VariableScalar(name='x', value=5.)
        self.assertEqual(x.get_value(), 5.)

    def test_is_type(self):
        
        v = optmod.variable.VariableScalar(name='v')
        self.assertFalse(v.is_constant())
        self.assertTrue(v.is_variable())
        self.assertFalse(v.is_function())

    def test_derivatives(self):

        x = optmod.variable.VariableScalar(name='x')
        y = optmod.variable.VariableScalar(name='y')

        dx = x.get_derivative(x)
        self.assertTrue(isinstance(dx, optmod.constant.Constant))
        self.assertEqual(dx.get_value(), 1.)

        dy = x.get_derivative(y)
        self.assertTrue(isinstance(dy, optmod.constant.Constant))
        self.assertEqual(dy.get_value(), 0.)

    def test_hashing(self):

        x = optmod.variable.VariableScalar(name='x')
        y = optmod.variable.VariableScalar(name='y')
        z = optmod.variable.VariableScalar(name='z')
        
        s = set([x,y,x,y])

        self.assertEqual(len(s), 2)
        self.assertTrue(x in s)
        self.assertTrue(y in s)
        self.assertFalse(z in s)

        s = {x: 1, y:2}
        self.assertTrue(x in s)
        self.assertTrue(y in s)
        self.assertFalse(z in s)
        self.assertEqual(s[x], 1)
        self.assertEqual(s[y], 2)
        s[x] = 10
        s[y] = 20
        self.assertEqual(s[x], 10)
        self.assertEqual(s[y], 20)
        self.assertFalse(z in s)
        s[z] = 100
        self.assertTrue(z in s)
        self.assertEqual(s[z], 100)

    def test_std_components(self):

        x = optmod.variable.VariableScalar(name='x')

        comp = x.__get_std_components__()

        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']
        
        self.assertTrue(phi is x)

        self.assertEqual(len(gphi_list), 1)
        self.assertTrue(gphi_list[0][0] is x)
        self.assertTrue(gphi_list[0][1].is_constant())
        self.assertEqual(gphi_list[0][1].get_value(), 1.)

        self.assertEqual(len(Hphi_list), 0)


