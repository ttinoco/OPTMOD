from optalg.opt_solver import OptSolverCbcCMD
from optmod import VariableScalar, Problem, minimize

x1 = VariableScalar('x1', type='integer')
x2 = VariableScalar('x2', type='integer')
x3 = VariableScalar('x3')
x4 = VariableScalar('x4')

f = -x1-x2
constraints = [-2*x1+2*x2+x3 == 1,
               -8*x1+10*x2+x4 == 13,
               x4 >= 0,
               x3 <= 0]

p = Problem(minimize(f), constraints)

p.solve(solver=OptSolverCbcCMD(), parameters={'quiet': False})

print(f.get_value())

for x in [x1, x2, x3, x4]:
    print(x, x.get_value())
