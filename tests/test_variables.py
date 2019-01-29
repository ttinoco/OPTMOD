import optmod
import unittest
import numpy as np

class TestVariables(unittest.TestCase):

    def test_user_function(self):

        x = optmod.Variable()
        self.assertTrue(isinstance(x, optmod.variable.VariableScalar))
        self.assertEqual(x.name, 'var')
        self.assertEqual(x.value, 0.)

        x = optmod.Variable(name='x', value=2.)
        self.assertTrue(isinstance(x, optmod.variable.VariableScalar))
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.value, 2.)

        x = optmod.Variable('x', None, (3,))
        self.assertTupleEqual(x.shape, (3,1))
        
        x = optmod.Variable(name='x', shape=(3,2))
        self.assertTrue(isinstance(x, optmod.variable.VariableMatrix))
        val = x.get_value()
        self.assertTrue(isinstance(val, np.matrix))
        self.assertTupleEqual(val.shape, x.shape)
        self.assertTupleEqual(val.shape, (3,2))
        self.assertTrue(np.all(val == np.zeros((3,2))))

        self.assertRaises(ValueError, optmod.Variable, 'x', np.zeros((3,2)), (4,2))

        x = optmod.Variable(name='x', value=[[1,2,3],[4,5,6]])
        self.assertTrue(isinstance(x, optmod.variable.VariableMatrix))
        val = x.get_value()
        self.assertTrue(isinstance(val, np.matrix))
        self.assertTupleEqual(val.shape, x.shape)
        self.assertTupleEqual(val.shape, (2,3))
        self.assertTrue(np.all(val == np.array([[1,2,3],[4,5,6]])))

        x = optmod.Variable(name='x', shape=(1,3), value=[[1,2,3]])
        
        x = optmod.Variable('x', [[3,4,5]], (3,))
        self.assertTupleEqual(x.shape, (3,1))

    def test_construction(self):

        x = optmod.variable.VariableScalar()
        self.assertEqual(x.name, 'var')
        self.assertEqual(x.value, 0.)
        
        x = optmod.variable.VariableScalar(name='x', value=3)
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.value, 3.)
        self.assertTrue(isinstance(x.value, np.float64))

    def test_get_variables(self):

        x = optmod.Variable(name='x')

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


