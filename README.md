# OPTMOD

[![Build Status](https://travis-ci.org/ttinoco/OPTMOD.svg?branch=master)](https://travis-ci.org/ttinoco/OPTMOD)

## Overview

Experimental optimization modeling layer for OPTALG with automatic sparse first and second derivatives.

Expressions for derivatives are constructed once and can be subsequently evaluated multiple times. Performance is expected to be slow in Python, but I will try to move the evaluation code to C soon. 

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

p = Problem(minimize(f), constraints=constraints)

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

## License

BSD 2-clause license.
