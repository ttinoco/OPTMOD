import numpy as np
from optalg.opt_solver import OptSolverIpopt
from optmod import VariableMatrix, Problem, minimize

N = 1000

x = VariableMatrix(name='x', shape=(N,1))

f = 0.
for i in range(N-1):
    f = f + 100*(x[i+1,0]-x[i,0]*x[i,0]) * (x[i+1,0] - x[i,0]*x[i,0]) + (1-x[i,0])*(1-x[i,0])
    
p = Problem(minimize(f))

info = p.solve(solver=OptSolverIpopt(), parameters={'quiet': True, 'max_iter': 1500})

print(info)    
print(f.get_value())
print(np.all(x.get_value() == 1.))
