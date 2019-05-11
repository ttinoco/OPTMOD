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
        
        x = optmod.VariableScalar(name='x', value=3.)
        y = optmod.VariableScalar(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        E = optmod.coptmod.Evaluator(2, 20)

        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 0)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)
        self.assertTupleEqual(E.shape, (1, 20))
        self.assertFalse(E.scalar_output)

        f.__fill_evaluator__(E)
        
        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 8)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)

        f = 4*x + y + 10 + 2*(x+y)

        E = optmod.coptmod.Evaluator(2, 20)

        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 0)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)
        self.assertTupleEqual(E.shape, (1, 20))
        self.assertFalse(E.scalar_output)

        f.__fill_evaluator__(E)
        
        self.assertEqual(E.max_nodes, 20)
        self.assertEqual(E.num_nodes, 9)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 20)        

    def test_evaluator_dynamic_resize(self):

        x = optmod.VariableScalar(name='x', value=3.)
        y = optmod.VariableScalar(name='y', value=4.)

        f = 4*(x + 1) + optmod.sin(-y)

        E = optmod.coptmod.Evaluator(2, 5)

        self.assertEqual(E.max_nodes, 5)
        self.assertEqual(E.num_nodes, 0)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 5)
        self.assertTupleEqual(E.shape, (1, 5))
        self.assertFalse(E.scalar_output)

        f.__fill_evaluator__(E)
        
        self.assertEqual(E.max_nodes, 10)
        self.assertEqual(E.num_nodes, 8)
        self.assertEqual(E.num_inputs, 2)
        self.assertEqual(E.num_outputs, 5)

    def test_evaluator_eval_single_output(self):

        x = optmod.VariableScalar(name='x', value=3.)
        y = optmod.VariableScalar(name='y', value=4.)

        # var
        f = x
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([5.])
        self.assertEqual(e.get_value(), 5.)
        
        # constant
        f = optmod.constant.Constant(11.)
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([5.])
        self.assertEqual(e.get_value(), 11.)
        
        # add
        f = x + 3
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([9.])
        self.assertEqual(e.get_value(), 12.)

        # sub
        f = x - 3
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([9.])
        self.assertEqual(e.get_value(), 6.)

        # negate
        f = -(x+10.)
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([9.])
        self.assertEqual(e.get_value(), -19.)

        # multiply
        f = y*(x + 3)
        e = optmod.coptmod.Evaluator(2, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_input_var(1, id(y))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([9., -20.])
        self.assertEqual(e.get_value(), -20.*(9.+3.))

        # sin
        f = optmod.sin(x + 3)
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([9.])
        self.assertEqual(e.get_value(), np.sin(12.))

        # cos
        f = optmod.cos(3*y)
        e = optmod.coptmod.Evaluator(1, 1, scalar_output=True)
        f.__fill_evaluator__(e)
        e.set_input_var(0, id(y))
        e.set_output_node(0, id(f))
        self.assertEqual(e.get_value(), 0.)
        e.eval([5.])
        self.assertEqual(e.get_value(), np.cos(15.))
        
    def test_evaluator_eval_multi_output(self):

        x = optmod.VariableScalar(name='x', value=3.)
        y = optmod.VariableScalar(name='y', value=4.)

        # var
        f1 = 3*(x+optmod.sin(y))
        f2 = optmod.sum([x,y])
        e = optmod.coptmod.Evaluator(2, 2)
        f1.__fill_evaluator__(e)
        f2.__fill_evaluator__(e)
        e.set_input_var(0, id(x))
        e.set_input_var(1, id(y))        
        e.set_output_node(0, id(f1))
        e.set_output_node(1, id(f2))
        val = e.get_value()
        self.assertTupleEqual(val.shape, (1,2))
        self.assertTrue(isinstance(val, np.matrix))
        e.eval([5., 8.])
        val = e.get_value()
        self.assertAlmostEqual(val[0,0], 3.*(5.+np.sin(8.)))
        self.assertAlmostEqual(val[0,1], 5.+8.)        
