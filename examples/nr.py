from optmod import VariableScalar, Problem, EmptyObjective, cos

x = VariableScalar('x', value=1.)

constraints = [x*cos(x)-x*x == 0]
        
p = Problem(EmptyObjective(), constraints)

info = p.solve('nr', parameters={'quiet': False, 'feastol': 1e-10})

print(x, x.get_value())
