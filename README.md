# OPTMOD

[![Build Status](https://travis-ci.org/ttinoco/OPTMOD.svg?branch=master)](https://travis-ci.org/ttinoco/OPTMOD)

## Overview

Experimental optimization modeling layer for OPTALG with automatic sparse first and second derivatives.

Expressions for derivatives are constructed once and can be subsequently evaluated multiple times. 

### Example NLP

```python
from optmod import Variable, Problem, minimize

x1 = Variable('x1', value=1)
x2 = Variable('x2', value=5)
x3 = Variable('x3', value=5)
x4 = Variable('x4', value=1)

f = x1*x4*(x1+x2+x3) + x3

constraints = [x1*x2*x3*x4 >= 25,
               x1*x1 + x2*x2 + x3*x3 + x4*x4 == 40,
               1 <= x1, x1 <= 5,
               1 <= x2, x2 <= 5,
               1 <= x3, x3 <= 5,
               1 <= x4, x4 <= 5]

p = Problem(minimize(f), constraints)

p.solve(solver='ipopt', parameters={'quiet': True})

print(f.get_value())

for x in [x1, x2, x3, x4]:
    print(x, x.get_value())
```

The solution output is
```python
17.014
x1, 0.999
x2, 4.742
x3, 3.821
x4, 1.379
```

### Fast Evaluators

Fast evaluators can be constructed for expressions. These evaluators evaluate the expression trees much faster. Eventually, these will be used to evaluate all expressions and their derivatives during the solution of a problem.

```python
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
```

The output is
```python
same value: True
speedup: 533.78
```

## License

BSD 2-clause license.
