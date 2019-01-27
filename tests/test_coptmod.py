import optmod
import unittest
import numpy as np

class TestCoptmod(unittest.TestCase):

    def test_module(self):
        
        c = optmod.coptmod

        # Manager
        self.assertTrue(hasattr(c, 'Manager'))

        # Node types
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
        
    def test_manager_construct(self):

        x = optmod.Variable(name='x', value=3.)
        y = optmod.Variable(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        m = optmod.coptmod.Manager(20)

        self.assertEqual(m.max_nodes, 20)
        self.assertEqual(m.num_nodes, 0)

        f.__fill_manager__(m)
        
        self.assertEqual(m.max_nodes, 20)
        self.assertEqual(m.num_nodes, 9)

    def test_manager_dynamic_resize(self):

        x = optmod.Variable(name='x', value=3.)
        y = optmod.Variable(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        m = optmod.coptmod.Manager(5)

        self.assertEqual(m.max_nodes, 5)
        self.assertEqual(m.num_nodes, 0)

        f.__fill_manager__(m)
        
        self.assertEqual(m.max_nodes, 10)
        self.assertEqual(m.num_nodes, 9)
        
