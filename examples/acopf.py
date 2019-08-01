import time
import sys
import pfnet
import optalg
from optmod import VariableDict, Problem, minimize, cos, sin

# Case
case = sys.argv[1]

# Network
net = pfnet.Parser(case).parse(case).get_copy(merge_buses=True)
net.show_components()

t = time.time()

# Generator active and reactive powers
gen_indices = [gen.index for gen in net.generators]
P0 = dict([(gen.index, gen.P) for gen in net.generators])
Q0 = dict([(gen.index, gen.Q) for gen in net.generators])
P = VariableDict(gen_indices, name='P', value=P0)
Q = VariableDict(gen_indices, name='Q', value=Q0)

# Bus voltage magnitudes and angles
bus_indices =[bus.index for bus in net.buses]
w0 = dict([(bus.index, bus.v_ang) for bus in net.buses])
v0 = dict([(bus.index, bus.v_mag) for bus in net.buses])
w = VariableDict(bus_indices, name='vang', value=w0)
v = VariableDict(bus_indices, name='vmag', value=v0)

# AC power balance
constraints = []
for bus in net.buses:
    dP = 0.
    dQ = 0.
    for gen in bus.generators:
        dP += P[gen.index]
        dQ += Q[gen.index]
    for load in bus.loads:
        dP -= load.P
        dQ -= load.Q
    for shunt in bus.shunts:
        dP -= shunt.g*v[bus.index]*v[bus.index]
        dQ += shunt.b*v[bus.index]*v[bus.index]
    for br in bus.branches_k:
        vk, vm = v[br.bus_k.index], v[br.bus_m.index]
        dw = w[br.bus_k.index]-w[br.bus_m.index]-br.phase
        dP -= (br.ratio**2.)*vk*vk*(br.g_k+br.g) - br.ratio*vk*vm*(br.g*cos(dw) + br.b*sin(dw))
        dQ -= -(br.ratio**2.)*vk*vk*(br.b_k+br.b) - br.ratio*vk*vm*(br.g*sin(dw) - br.b*cos(dw))
    for br in bus.branches_m:
        vk, vm = v[br.bus_k.index], v[br.bus_m.index]
        dw = w[br.bus_m.index]-w[br.bus_k.index]+br.phase
        dP -= vm*vm*(br.g_m+br.g) - br.ratio*vm*vk*(br.g*cos(dw) + br.b*sin(dw))
        dQ -= -vm*vm*(br.b_m+br.b) - br.ratio*vm*vk*(br.g*sin(dw) - br.b*cos(dw))
    constraints.extend([dP == 0., dQ == 0.])
    assert(abs(dP.get_value()-bus.P_mismatch) < 1e-8)
    assert(abs(dQ.get_value()-bus.Q_mismatch) < 1e-8)

# Variable limits
for gen in net.generators:
    constraints.extend([gen.P_min <= P[gen.index], gen.P_max >= P[gen.index],
                        gen.Q_min <= Q[gen.index], gen.Q_max >= Q[gen.index]])
for bus in net.buses:
    constraints.extend([bus.v_min <= v[bus.index], bus.v_max >= v[bus.index]])

# Reference angles
for bus in net.buses:
    if bus.is_slack():
        constraints.append(w[bus.index] == bus.v_ang)

# Objective
func = 0.
for gen in net.generators:
    func = func + 0.5*(P[gen.index]-gen.P)*(P[gen.index]-gen.P)

# Problem
problem = Problem(minimize(func), constraints)

time_construction = time.time()-t

# Solve
info = problem.solve(solver=optalg.opt_solver.OptSolverIpopt(), parameters={'tol': 1e-4}, fast_evaluator=True)
print("OPTMOD", func.get_value())
print(info)
print('time construction', time_construction)

t = time.time()

# PFNET and OPTALG
net.set_flags('bus',
              'variable',
              'not slack',
              'voltage angle')
net.set_flags('bus',
              ['variable', 'bounded'],
              'any',
              'voltage magnitude')
net.set_flags('generator',
              ['variable', 'bounded'],
              'any',
              ['active power', 'reactive power'])
p = pfnet.Problem(net)
p.add_function(pfnet.Function('generation redispatch penalty', 1., net))
p.add_constraint(pfnet.Constraint('AC power balance', net))
p.add_constraint(pfnet.Constraint('variable bounds', net))

time_construction = time.time()-t

t = time.time()

p.analyze()
p.show()

time_transformation = time.time()-t

t = time.time()

solver = optalg.opt_solver.OptSolverIpopt()
solver.set_parameters({'tol': 1e-4})
solver.solve(p)

time_solver = time.time()-t

print("PFNET", p.phi)
print('time construction', time_construction)
print('time transformation', time_transformation)
print('time solver', time_solver)

net.set_var_values(solver.get_primal_variables())
net.update_properties()

# Verification
net.show_properties()
for bus in net.buses:
    bus.v_mag = v[bus.index].get_value()
    bus.v_ang = w[bus.index].get_value()
for gen in net.generators:
    gen.P = P[gen.index].get_value()
    gen.Q = Q[gen.index].get_value()
net.update_properties()
net.show_properties()


