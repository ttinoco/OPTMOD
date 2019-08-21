import optmod
import unittest
import numpy as np

class TestConstraints(unittest.TestCase):

    def test_slack(self):

        x = optmod.variable.VariableScalar('x', value=2.)
        y = optmod.variable.VariableScalar('y', value=3.)

        c1 = x >= y
        c2 = x <= y
        c3 = x == y

        self.assertEqual(c1.op, '>=')
        self.assertEqual(c2.op, '<=')
        self.assertEqual(c3.op, '==')

        self.assertTrue(isinstance(c1.slack, optmod.variable.VariableScalar))
        self.assertTrue(isinstance(c2.slack, optmod.variable.VariableScalar))
        self.assertTrue(isinstance(c3.slack, optmod.variable.VariableScalar))

        self.assertEqual(c1.slack.name, 's')
        self.assertEqual(c2.slack.name, 's')
        self.assertEqual(c3.slack.name, 's')

    def test_violation(self):

        x = optmod.variable.VariableScalar('x', value=2.)
        y = optmod.variable.VariableScalar('y', value=3.)

        c = x == y
        self.assertEqual(c.get_violation(), 1.)

        c = y == x
        self.assertEqual(c.get_violation(), 1.)

        c = x >= y
        self.assertEqual(c.get_violation(), 1.)

        c = y >= x
        self.assertEqual(c.get_violation(), 0.)

        c = x <= y
        self.assertEqual(c.get_violation(), 0.)

        c = y <= x
        self.assertEqual(c.get_violation(), 1.)

        c = 5 == x
        self.assertEqual(c.get_violation(), 3.)

        c = x == 5
        self.assertEqual(c.get_violation(), 3.)

    def test_equal_contruction(self):

        x = optmod.variable.VariableScalar('x')
        y = optmod.variable.VariableScalar('y')
        r = np.random.randn(2,3)
        z = optmod.variable.VariableMatrix('z', shape=(2,3))
        
        c = x == 1
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '==')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 1.)

        c = 2 == x
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '==')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 2.)

        c = x == y
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '==')
        self.assertIs(c.lhs, x)
        self.assertIs(c.rhs, y)

        c = x == r
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, x)
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), r[i,j])

        c = r == x
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, x)
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), r[i,j])
        
        c = x == z
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, x)
                self.assertIs(cij.rhs, z[i,j])
        
        c = z == x
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.rhs, x)
                self.assertIs(cij.lhs, z[i,j])

        c = 3. == z
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, z[i,j])
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), 3.)

        c = z == 4.
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, z[i,j])
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), 4.)

        c = r == z
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, z[i,j])
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), r[i,j])

        c = z == r
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertIs(cij.lhs, z[i,j])
                self.assertTrue(cij.rhs.is_constant())
                self.assertEqual(cij.rhs.get_value(), r[i,j])

        c = 2*z == (z+1)
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        for i in range(2):
            for j in range(3):
                cij = c[i,j]
                self.assertTrue(isinstance(cij, optmod.constraint.Constraint))
                self.assertEqual(cij.op, '==')
                self.assertTrue(isinstance(cij.lhs, optmod.function.multiply))
                self.assertTrue(isinstance(cij.rhs, optmod.function.add))
                self.assertTrue(cij.lhs.arguments[0] is z[i,j])
                self.assertTrue(cij.rhs.arguments[0] is z[i,j])

    def test_bad_array_construction(self):

        x = optmod.VariableMatrix('x', np.random.randn(4,3))
        y = optmod.VariableScalar('y', 5)

        c0 = x == 1
        c1 = y == 0

        c = optmod.constraint.ConstraintArray(c0)
        c = optmod.constraint.ConstraintArray(c1)
        c = optmod.constraint.ConstraintArray([c1])
        
        self.assertRaises(TypeError, optmod.constraint.ConstraintArray, [x, y])
        self.assertRaises(TypeError, optmod.constraint.ConstraintArray, [c0, c1])
        self.assertRaises(TypeError, optmod.constraint.ConstraintArray, ['foo'])
        
    def test_flatten_to_list(self):

        x = optmod.VariableMatrix('x', np.random.randn(4,3))

        c = x == 1
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))

        self.assertTupleEqual(c.shape, (4,3))

        cf = c.flatten()
        self.assertTrue(isinstance(c, optmod.constraint.ConstraintArray))
        self.assertTupleEqual(cf.shape, (12,))

        cfl = cf.tolist()
        self.assertTrue(isinstance(cfl, list))
        self.assertEqual(len(cfl), 12)
        for i in range(4):
            for j in range(3):
                self.assertTrue(isinstance(cfl[i*3+j], optmod.constraint.Constraint))
                self.assertTrue(cfl[i*3+j].lhs is x[i,j])

    def test_less_equal_contruction(self):

        x = optmod.variable.VariableScalar('x')
        y = optmod.variable.VariableScalar('y')
        r = np.random.randn(2,3)
        z = optmod.variable.VariableMatrix('z', shape=(2,3))
        
        c = x <= 1
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '<=')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 1.)

        c = 2 <= x
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '>=')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 2.)
    
    def test_greater_equal_construction(self):

        x = optmod.variable.VariableScalar('x')
        y = optmod.variable.VariableScalar('y')
        r = np.random.randn(2,3)
        z = optmod.variable.VariableMatrix('z', shape=(2,3))
        
        c = x >= 1
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '>=')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 1.)

        c = 2 >= x
        self.assertTrue(isinstance(c, optmod.constraint.Constraint))
        self.assertEqual(c.op, '<=')
        self.assertIs(c.lhs, x)
        self.assertTrue(c.rhs.is_constant())
        self.assertEqual(c.rhs.get_value(), 2.)

    def test_std_components(self):

        x = optmod.variable.VariableScalar('x', value=3.)
        y = optmod.variable.VariableScalar('y', value=4.)

        self.assertEqual(x.name, 'x')
        self.assertEqual(y.name, 'y')

        # Upper bound
        c = x <= 5
        comp = c.__get_std_components__()
        
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(A_list), 0)
                
        self.assertEqual(len(b_list), 0)
        
        self.assertEqual(len(f_list), 0)
        self.assertEqual(len(J_list), 0)
        self.assertEqual(len(H_list), 0)
        
        self.assertEqual(len(u_list), 1)
        self.assertIs(u_list[0][0], x)
        self.assertEqual(u_list[0][1], 5.)
        self.assertEqual(len(l_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 1)

        # Lower bound
        c = x >= 10
        comp = c.__get_std_components__()
        
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(A_list), 0)
                
        self.assertEqual(len(b_list), 0)
        
        self.assertEqual(len(f_list), 0)
        self.assertEqual(len(J_list), 0)
        self.assertEqual(len(H_list), 0)
        
        self.assertEqual(len(l_list), 1)
        self.assertIs(l_list[0][0], x)
        self.assertEqual(l_list[0][1], 10.)
        self.assertEqual(len(u_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 1)

        # Linear ==
        c = x + 2*y == 1
        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(A_list), 2)
        
        self.assertEqual(A_list[0][0], 0)
        self.assertIs(A_list[0][1], x)
        self.assertEqual(A_list[0][2], 1.)

        self.assertEqual(A_list[1][0], 0)
        self.assertIs(A_list[1][1], y)
        self.assertEqual(A_list[1][2], 2.)
        
        self.assertEqual(len(b_list), 1)
        self.assertEqual(b_list[0], 1.)
        
        self.assertEqual(len(f_list), 0)
        self.assertEqual(len(J_list), 0)
        self.assertEqual(len(H_list), 0)
        self.assertEqual(len(u_list), 0)
        self.assertEqual(len(l_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 2)

        # Linear inequality <=
        c = 3*x + 2*y <= 1

        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(A_list), 3)
        
        self.assertEqual(A_list[0][0], 0)
        self.assertIs(A_list[0][1], x)
        self.assertEqual(A_list[0][2], 3.)

        self.assertEqual(A_list[1][0], 0)
        self.assertIs(A_list[1][1], y)
        self.assertEqual(A_list[1][2], 2.)

        self.assertEqual(A_list[2][0], 0)
        self.assertEqual(A_list[2][1].name, 's')
        self.assertEqual(A_list[2][2], -1)
        
        self.assertEqual(len(b_list), 1)
        self.assertEqual(b_list[0], 1)
        
        self.assertEqual(len(f_list), 0)
        self.assertEqual(len(J_list), 0)
        self.assertEqual(len(H_list), 0)
        
        self.assertEqual(len(u_list), 1)
        self.assertEqual(u_list[0][0].name, 's')
        self.assertEqual(u_list[0][1], 0.)
        
        self.assertEqual(len(l_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 3)

        # Linear inequality >=
        c = 3*x + 2*y >= 1
        
        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(A_list), 3)
        
        self.assertEqual(A_list[0][0], 0)
        self.assertIs(A_list[0][1], x)
        self.assertEqual(A_list[0][2], 3.)

        self.assertEqual(A_list[1][0], 0)
        self.assertIs(A_list[1][1], y)
        self.assertEqual(A_list[1][2], 2.)

        self.assertEqual(A_list[2][0], 0)
        self.assertEqual(A_list[2][1].name, 's')
        self.assertEqual(A_list[2][2], -1)
        
        self.assertEqual(len(b_list), 1)
        self.assertEqual(b_list[0], 1)
        
        self.assertEqual(len(f_list), 0)
        self.assertEqual(len(J_list), 0)
        self.assertEqual(len(H_list), 0)
        
        self.assertEqual(len(l_list), 1)
        self.assertEqual(l_list[0][0].name, 's')
        self.assertEqual(l_list[0][1], 0.)
        
        self.assertEqual(len(u_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 3)

        # Nonlinear ==
        c = 4*x + x*y + optmod.sin(y) == 1
        
        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(J_list), 2)
        
        self.assertEqual(J_list[0][0], 0)
        self.assertIs(J_list[0][1], x)
        self.assertEqual(J_list[0][2].get_value(), 4.+4.)

        self.assertEqual(J_list[1][0], 0)
        self.assertIs(J_list[1][1], y)
        self.assertEqual(J_list[1][2].get_value(), 3.+np.cos(4.))
        
        self.assertEqual(len(f_list), 1)
        self.assertEqual(f_list[0].get_value(), 4*3. + 3.*4. + np.sin(4.) - 1.)

        self.assertEqual(len(H_list), 1)
        self.assertEqual(len(H_list[0]), 2)
        self.assertIs(H_list[0][0][0], x)
        self.assertIs(H_list[0][0][1], y)
        self.assertTrue(H_list[0][0][2].is_constant(1.))

        self.assertIs(H_list[0][1][0], y)
        self.assertIs(H_list[0][1][1], y)
        self.assertEqual(H_list[0][1][2].get_value(), -np.sin(4.))
        
        self.assertEqual(len(b_list), 0)
        self.assertEqual(len(A_list), 0)
        self.assertEqual(len(u_list), 0)
        self.assertEqual(len(l_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 2)

        # Nonlinear <=
        c = 4*x + x*y + optmod.sin(y) <= 5 
        
        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(J_list), 3)
        
        self.assertEqual(J_list[0][0], 0)
        self.assertIs(J_list[0][1], x)
        self.assertEqual(J_list[0][2].get_value(), 4.+4.)

        self.assertEqual(J_list[1][0], 0)
        self.assertIs(J_list[1][1], y)
        self.assertEqual(J_list[1][2].get_value(), 3.+np.cos(4.))

        self.assertEqual(J_list[2][0], 0)
        self.assertEqual(J_list[2][1].name, 's')
        self.assertEqual(J_list[2][2].get_value(), -1.)
        
        self.assertEqual(len(f_list), 1)
        self.assertEqual(f_list[0].get_value(), 4*3. + 3.*4. + np.sin(4.) - 5.)

        self.assertEqual(len(H_list), 1)
        self.assertEqual(len(H_list[0]), 2)
        self.assertIs(H_list[0][0][0], x)
        self.assertIs(H_list[0][0][1], y)
        self.assertTrue(H_list[0][0][2].is_constant(1.))

        self.assertIs(H_list[0][1][0], y)
        self.assertIs(H_list[0][1][1], y)
        self.assertEqual(H_list[0][1][2].get_value(), -np.sin(4.))
        
        self.assertEqual(len(b_list), 0)
        self.assertEqual(len(A_list), 0)
        self.assertEqual(len(u_list), 1)
        self.assertEqual(u_list[0][0].name, 's')
        self.assertEqual(u_list[0][1], 0.)
        self.assertEqual(len(l_list), 0)

        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 3)

        # Nonlinear >=
        c = 4*x + x*y + optmod.sin(y) >= 13.
        
        comp = c.__get_std_components__()
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(J_list), 3)
        
        self.assertEqual(J_list[0][0], 0)
        self.assertIs(J_list[0][1], x)
        self.assertEqual(J_list[0][2].get_value(), 4.+4.)

        self.assertEqual(J_list[1][0], 0)
        self.assertIs(J_list[1][1], y)
        self.assertEqual(J_list[1][2].get_value(), 3.+np.cos(4.))

        self.assertEqual(J_list[2][0], 0)
        self.assertEqual(J_list[2][1].name, 's')
        self.assertEqual(J_list[2][2].get_value(), -1.)
        
        self.assertEqual(len(f_list), 1)
        self.assertEqual(f_list[0].get_value(), 4*3. + 3.*4. + np.sin(4.) - 13.)

        self.assertEqual(len(H_list), 1)
        self.assertEqual(len(H_list[0]), 2)
        self.assertIs(H_list[0][0][0], x)
        self.assertIs(H_list[0][0][1], y)
        self.assertTrue(H_list[0][0][2].is_constant(1.))

        self.assertIs(H_list[0][1][0], y)
        self.assertIs(H_list[0][1][1], y)
        self.assertEqual(H_list[0][1][2].get_value(), -np.sin(4.))
        
        self.assertEqual(len(b_list), 0)
        self.assertEqual(len(A_list), 0)
        self.assertEqual(len(l_list), 1)
        self.assertEqual(l_list[0][0].name, 's')
        self.assertEqual(l_list[0][1], 0.)
        self.assertEqual(len(u_list), 0)
                
        self.assertEqual(len(prop_list), 1)
        self.assertEqual(len(prop_list[0]['a']), 3)
