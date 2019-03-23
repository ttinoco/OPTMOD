# OPTMOD

[![Build Status](https://travis-ci.org/ttinoco/OPTMOD.svg?branch=master)](https://travis-ci.org/ttinoco/OPTMOD)

## Overview

Experimental optimization modeling layer for [OPTALG](https://github.com/ttinoco/OPTALG) (dev branch) with automatic sparse first and second derivatives.

Expressions for derivatives are constructed once and can be then evaluated multiple times efficiently.

### Fast Evaluators

Fast evaluators can be constructed for expressions. These evaluators evaluate the expression trees much faster. These are used by default to evaluate all expressions and their derivatives during the solution of an optimization problem.

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

### Example MILP

```python
from optmod import Variable, Problem, minimize

x1 = Variable('x1', type='integer')
x2 = Variable('x2', type='integer')
x3 = Variable('x3')
x4 = Variable('x4')

f = -x1-x2
constraints = [-2*x1+2*x2+x3 == 1,
               -8*x1+10*x2+x4 == 13,
               x4 >= 0,
               x3 <= 0]

p = Problem(minimize(f), constraints)

p.solve(solver='cbc_cmd', parameters={'quiet': False})

print(f.get_value())

for x in [x1, x2, x3, x4]:
    print(x, x.get_value())
```

The solution output is
```python
-3.0
x1, 1.0
x2, 2.0
x3, -1.0
x4, 1.0
```

### Example Newton-Raphson

```python
from optmod import Variable, Problem, EmptyObjective, cos

x = Variable('x', value=1.)

constraints = [x*cos(x)-x*x == 0]
        
p = Problem(EmptyObjective(), constraints)

status = p.solve('nr', parameters={'quiet': False, 'feastol': 1e-10})

print(x, x.get_value())
```

The solution output is
```python
x, 0.739085
```

## License

BSD 2-clause license.
