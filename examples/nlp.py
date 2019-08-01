# Hock-Schittkowski
# Problem 71

from optalg.opt_solver import OptSolverIpopt
from optmod import VariableScalar, Problem, minimize

x1 = VariableScalar('x1', value=1)
x2 = VariableScalar('x2', value=5)
x3 = VariableScalar('x3', value=5)
x4 = VariableScalar('x4', value=1)

f = x1*x4*(x1+x2+x3) + x3

constraints = [x1*x2*x3*x4 >= 25,
               x1*x1 + x2*x2 + x3*x3 + x4*x4 == 40,
               1 <= x1, x1 <= 5,
               1 <= x2, x2 <= 5,
               1 <= x3, x3 <= 5,
               1 <= x4, x4 <= 5]

p = Problem(minimize(f), constraints)

p.solve(solver=OptSolverIpopt(), parameters={'quiet': False})

print(f.get_value())

for x in [x1, x2, x3, x4]:
    print(x, x.get_value())
