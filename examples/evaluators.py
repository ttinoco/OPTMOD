import time
import optmod
import numpy as np

x = optmod.Variable(name='x', value=np.random.randn(4,3))
y = optmod.Variable(name='y', value=10.)

f = optmod.sin(3*x+10.)*optmod.cos(y-optmod.sum(x*y))

vars = list(x.get_variables())+[y]

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

