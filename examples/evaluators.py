import time
import numpy as np
from optmod import VariableMatrix, VariableScalar, sin, cos, sum

x = VariableMatrix(name='x', value=np.random.randn(4,3))
y = VariableScalar(name='y', value=10.)

f = sin(3*x+10.)*cos(y-sum(x*y))

vars = list(f.get_variables())

e = f.get_fast_evaluator(vars)
var_values = np.array([v.get_value() for v in vars])
e.eval(var_values)

print('same value:', np.all(e.get_value() == f.get_value()))

t0 = time.time()
for i in range(500):
    f.get_value()
t1 = time.time()
for i in range(500):
    e.eval(var_values)
t2 = time.time()
print('speedup:', (t1-t0)/(t2-t1))

