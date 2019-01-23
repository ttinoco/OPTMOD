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

        pass

    def test_manager_dynamic_resize(self):

        pass
