import optmod
import unittest
import numpy as np

class TestVariableDicts(unittest.TestCase):

    def test_construction(self):

        x = optmod.VariableDict(['a', 'b'], name='foo')

        self.assertTrue(isinstance(x, dict))

        self.assertEqual(len(x), 2)
        
        xa = x['a']
        self.assertTrue(isinstance(xa, optmod.VariableScalar))
        self.assertEqual(xa.value, 0.)
        self.assertTrue(xa.is_continuous())
        self.assertEqual(xa.name, 'foo_a')

        xb = x['b']
        self.assertTrue(isinstance(xb, optmod.VariableScalar))
        self.assertEqual(xb.value, 0.)
        self.assertTrue(xb.is_continuous())
        self.assertEqual(xb.name, 'foo_b')

        x = optmod.VariableDict(['a', 'b'], name='bar', value={'a': 10, 'c': 100})

        self.assertTrue(isinstance(x, dict))

        self.assertEqual(len(x), 2)
        
        xa = x['a']
        self.assertTrue(isinstance(xa, optmod.VariableScalar))
        self.assertEqual(xa.value, 10.)
        self.assertTrue(xa.is_continuous())
        self.assertEqual(xa.name, 'bar_a')

        xb = x['b']
        self.assertTrue(isinstance(xb, optmod.VariableScalar))
        self.assertEqual(xb.value, 0.)
        self.assertTrue(xb.is_continuous())
        self.assertEqual(xb.name, 'bar_b')
        
