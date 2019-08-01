# Ported from https://github.com/JuliaOpt/JuMP.jl/blob/master/examples/clnlbeam.jl

import optmod
import optalg
from optmod import sum, cos, sin, minimize

N = 1000
h = 1./N
alpha = 350.

t = optmod.VariableMatrix('t', shape=(N+1,1))
x = optmod.VariableMatrix('x', shape=(N+1,1))
u = optmod.VariableMatrix('u', shape=(N+1,1))

f = sum([0.5*h*(u[i,0]*u[i,0]+u[i+1,0]*u[i+1,0]) +
         0.5*alpha*h*(cos(t[i,0]) + cos(t[i+1,0]))
         for i in range(N)])

constraints = []
for i in range(N):
    constraints.append(x[i+1,0] - x[i,0] - 0.5*h*(sin(t[i+1,0])+sin(t[i,0])) == 0)
    constraints.append(t[i+1,0] - t[i,0] - 0.5*h*(u[i+1,0] - u[i,0]) == 0)
constraints.append(t <= 1)
constraints.append(t >= -1)
constraints.append(-0.05 <= x)
constraints.append(x <= 0.05)

p = optmod.Problem(minimize(f), constraints)

info = p.solve(solver=optalg.opt_solver.OptSolverIpopt(), fast_evaluator=True)

print(info)
print(f.get_value())

