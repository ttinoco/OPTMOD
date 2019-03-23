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
