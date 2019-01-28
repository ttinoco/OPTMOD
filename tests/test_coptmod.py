import optmod
import unittest
import numpy as np

class TestCoptmod(unittest.TestCase):

    def test_module(self):
        
        c = optmod.coptmod

        self.assertTrue(hasattr(c, 'Evaluator'))

        self.assertListEqual([c.NODE_TYPE_UNKNOWN,
                              c.NODE_TYPE_CONSTANT,
                              c.NODE_TYPE_VARIABLE,
                              c.NODE_TYPE_ADD,
                              c.NODE_TYPE_SUBTRACT,
                              c.NODE_TYPE_NEGATE,
                              c.NODE_TYPE_MULTIPLY,
                              c.NODE_TYPE_SIN,
                              c.NODE_TYPE_COS],
                             list(range(9)))
        
    def test_evaluator_construct(self):
        
        x = optmod.Variable(name='x', value=3.)
        y = optmod.Variable(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        E = optmod.coptmod.Evaluator(2, 20)

        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 0)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)

        f.__fill_evaluator__(E)
        
        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 9)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)

    def test_evaluator_dynamic_resize(self):

        x = optmod.Variable(name='x', value=3.)
        y = optmod.Variable(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        E = optmod.coptmod.Evaluator(2, 5)

        self.assertEqual(E.max_nodes, 5)
        self.assertEqual(E.num_nodes, 0)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 5)

        f.__fill_evaluator__(E)
        
        self.assertEqual(E.max_nodes, 10)
        self.assertEqual(E.num_nodes, 9)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 5)
