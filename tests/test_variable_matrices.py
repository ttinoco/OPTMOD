import optmod
import unittest
import numpy as np

class TestVariableMatrices(unittest.TestCase):

    def test_construction(self):
        
        x = optmod.variable.VariableMatrix()
        self.assertTupleEqual(x.shape, (1,1))
        self.assertTrue(isinstance(x.data, np.matrix))
        self.assertTrue(x.data.dtype is np.dtype(object))
        self.assertTrue(isinstance(x[0,0], optmod.variable.VariableScalar))
        self.assertEqual(x[0,0].name, 'var[0,0]')
        self.assertEqual(x[0,0].get_value(), 0.)

        value = np.random.randn(2,3)
        x = optmod.variable.VariableMatrix(name='x', shape=(2,3), value=value)
        self.assertTupleEqual(x.shape, (2,3))
        self.assertTrue(isinstance(x.data, np.matrix))
        self.assertTrue(x.data.dtype is np.dtype(object))
        for i in range(2):
            for j in range(3):
                self.assertEqual(x[i,j].name, 'x[%d,%d]' %(i,j))
                self.assertEqual(x[i,j].get_value(), value[i,j])

        value = np.random.randn(4,5)
        x = optmod.variable.VariableMatrix(name='x', value=value)
        self.assertEqual(x.shape, (4,5))
        for i in range(4):
            for j in range(5):
                self.assertEqual(x[i,j].name, 'x[%d,%d]' %(i,j))
                self.assertEqual(x[i,j].get_value(), value[i,j])

        x = optmod.variable.VariableMatrix(name='x', shape=(3,2))
        self.assertEqual(x.shape, (3,2))
        for i in range(3):
            for j in range(2):
                self.assertEqual(x[i,j].name, 'x[%d,%d]' %(i,j))
                self.assertEqual(x[i,j].get_value(), 0.)
        
        self.assertRaises(ValueError, optmod.variable.VariableMatrix, x, np.random.randn(1,4), (2,3))
        self.assertRaises(ValueError, optmod.variable.VariableMatrix, x, [[1,2], [3,4]], (2,3))

        x = optmod.VariableMatrix('x', None, (3,))
        self.assertTupleEqual(x.shape, (3,1))
        
        x = optmod.VariableMatrix(name='x', shape=(3,2))
        self.assertTrue(isinstance(x, optmod.variable.VariableMatrix))
        val = x.get_value()
        self.assertTrue(isinstance(val, np.matrix))
        self.assertTupleEqual(val.shape, x.shape)
        self.assertTupleEqual(val.shape, (3,2))
        self.assertTrue(np.all(val == np.zeros((3,2))))

        self.assertRaises(ValueError, optmod.VariableMatrix, 'x', np.zeros((3,2)), (4,2))

        x = optmod.VariableMatrix(name='x', value=[[1,2,3],[4,5,6]])
        self.assertTrue(isinstance(x, optmod.variable.VariableMatrix))
        val = x.get_value()
        self.assertTrue(isinstance(val, np.matrix))
        self.assertTupleEqual(val.shape, x.shape)
        self.assertTupleEqual(val.shape, (2,3))
        self.assertTrue(np.all(val == np.array([[1,2,3],[4,5,6]])))
        for i in range(2):
            for j in range(3):
                self.assertFalse(x[i,j].is_integer())
                self.assertTrue(x[i,j].is_continuous())                              
        
        x = optmod.VariableMatrix(name='x', shape=(1,3), value=[[1,2,3]])
        
        x = optmod.VariableMatrix('x', [[3,4,5]], (3,))
        self.assertTupleEqual(x.shape, (3,1))

        self.assertRaises(ValueError, optmod.VariableMatrix, 'x', 0., None, 'foo')

        x = optmod.VariableMatrix('x', shape=(2,3), type='integer')
        self.assertTupleEqual(x.shape, (2,3))
        for i in range(2):
            for j in range(3):
                self.assertTrue(x[i,j].is_integer())
                self.assertFalse(x[i,j].is_continuous())

    def test_type(self):

        x = optmod.variable.VariableMatrix(name='x', shape=(2,3))
        for i in range(2):
            for j in range(3):
                self.assertTrue(x[i,j].is_continuous())
                self.assertFalse(x[i,j].is_integer())

        x = optmod.variable.VariableMatrix(name='x', shape=(2,3), type='integer')
        for i in range(2):
            for j in range(3):
                self.assertFalse(x[i,j].is_continuous())
                self.assertTrue(x[i,j].is_integer())        

    def test_get_variables(self):

        x = optmod.VariableMatrix(name='x', shape=(2,3))

        self.assertSetEqual(x.get_variables(), set([x[i,j] for i in range(2) for j in range(3)]))

    def test_repr(self):

        x = optmod.VariableMatrix(name='x', shape=(2,3))
        s1 = str(x)
        s2 = ('[[ x[0,0], x[0,1], x[0,2] ],\n' +
              ' [ x[1,0], x[1,1], x[1,2] ]]\n')
        self.assertEqual(s1, s2)              

    def test_value(self):

        x = optmod.variable.VariableMatrix(name='x', shape=(2,3))
        self.assertTrue(isinstance(x.get_value(), np.matrix))
        self.assertTrue(np.all(x.get_value() == np.zeros((2,3))))

        r = np.random.randn(2,3)
        x = optmod.variable.VariableMatrix(name='x', shape=(2,3), value=r)
        self.assertTrue(isinstance(x.get_value(), np.matrix))
        self.assertTrue(np.all(x.get_value() == r))

    def test_set_value(self):

        x = optmod.variable.VariableMatrix(name='x', shape=(2,3))

        r = np.random.randn(2,3)
        
        self.assertTrue(isinstance(x.get_value(), np.matrix))
        self.assertTrue(np.all(x.get_value() == np.zeros((2,3))))

        self.assertRaises(ValueError, x.set_value, np.random.randn(3,2))
        self.assertRaises(ValueError, x.set_value, 10)
        
        x.set_value(r)

        self.assertTrue(isinstance(x.get_value(), np.matrix))
        self.assertTrue(np.all(x.get_value() == r))

        for i in range(2):
            for j in range(3):
                self.assertEqual(x[i,j].get_value(), r[i,j])
        
