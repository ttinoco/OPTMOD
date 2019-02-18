# Hock-Schittkowski
# Problem 71

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

p.solve(solver='ipopt', parameters={'quiet': False})

print(f.get_value())

for x in [x1, x2, x3, x4]:
    print(x, x.get_value())
