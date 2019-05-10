import optmod
import unittest
import numpy as np

class TestConstants(unittest.TestCase):

    def test_construction(self):

        c = optmod.constant.Constant(4.)
        self.assertTrue(isinstance(c, optmod.expression.Expression))
        self.assertTrue(isinstance(c, optmod.constant.Constant))
        self.assertEqual(c.name, 'const')
        self.assertEqual(c.get_value(), 4.)

        self.assertRaises(TypeError, optmod.constant.Constant, [1,2,3])
        self.assertRaises(TypeError, optmod.constant.Constant, optmod.constant.Constant(3.))

    def test_get_variables(self):

        c = optmod.constant.Constant(4.)

        self.assertSetEqual(c.get_variables(), set())

    def test_repr(self):

        c = optmod.constant.Constant(5.)
        s = str(c)
        self.assertEqual(s, optmod.utils.repr_number(5.))

    def test_value(self):

        c = optmod.constant.Constant(6.)
        self.assertEqual(c.get_value(), 6.)        

    def test_is_zero(self):

        c = optmod.constant.Constant(2.)
        self.assertFalse(c.is_zero())
        c = optmod.constant.Constant(0.)
        self.assertTrue(c.is_zero())

    def test_is_one(self):

        c = optmod.constant.Constant(2.)
        self.assertFalse(c.is_one())
        c = optmod.constant.Constant(1.)
        self.assertTrue(c.is_one())

    def test_is_type(self):

        c = optmod.constant.Constant(4.)
        self.assertTrue(c.is_constant())
        self.assertFalse(c.is_variable())
        self.assertFalse(c.is_function())
        
        self.assertFalse(c.is_constant(5.))
        self.assertTrue(c.is_constant(4.))

    def test_derivatives(self):

        c = optmod.constant.Constant(4.)
        x = optmod.variable.VariableScalar('x')

        dc = c.get_derivative(x)
        self.assertTrue(dc.is_constant())
        self.assertEqual(dc.get_value(), 0.)

    def test_std_components(self):

        c = optmod.constant.Constant(4.)

        comp = c.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']

        self.assertTrue(phi is c)
        self.assertEqual(len(gphi_list), 0)
        self.assertEqual(len(Hphi_list), 0)
