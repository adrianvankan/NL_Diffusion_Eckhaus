"""
Dedalus script simulating the 1D Eckhaus Nonlinear Diffusion equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).

"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#Q = np.load('Q.npy')
#x = np.load('x.npy')
#plt.plot(x,Q); plt.show()

load_ic = True

# Parameters
if load_ic == True:
  Q = np.load('Q.npy')
  #efun = 0*np.load('efun.npy')
  x = np.load('x.npy')
  plt.plot(Q); plt.plot(Q,'k--'); plt.show()
  Lx = np.max(x)-np.min(x) #100
  Nx = len(Q)              #512
else:
  Lx = 50
  Nx = 1024 
kappa = 1
dnl   =  0#0.05
dealias = 3/2
stop_sim_time = 1000
timestepper = d3.SBDF2
timestep = 5e-4#5e-3
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
lift_basis = xbasis.derivative_basis(3)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Fields
u = dist.Field(name='u', bases=xbasis)
tau1 = dist.Field(name='tau1')
tau2 = dist.Field(name='tau2')
tau3 = dist.Field(name='tau3')
tau4 = dist.Field(name='tau4')

# Tau polynomials
tau_basis = xbasis.derivative_basis(2)
p1 = dist.Field(bases=tau_basis)
p2 = dist.Field(bases=tau_basis)
p3 = dist.Field(bases=tau_basis)
p4 = dist.Field(bases=tau_basis)
p1['c'][-1] = 1
p2['c'][-2] = 2
p3['c'][-3] = 3
p4['c'][-4] = 4

# Substitutions
dx       = lambda A: d3.Differentiate(A, xcoord)

# Problem
problem = d3.IVP([u,tau1,tau2,tau3,tau4], namespace=locals())
problem.add_equation("dt(u) + kappa*dx(dx(dx(dx(u)))) + tau1*p1 + tau2*p2 + tau3*p3 + tau4*p4 = dx((1-3*u**2)/(1-u**2)*dx(u)) + dnl*u*dx(u)") 
problem.add_equation("dx(u)(x=0) = 0")
problem.add_equation("dx(u)(x=Lx) = 0")
problem.add_equation("dx(dx(dx(u)))(x=0)  = 0")
problem.add_equation("dx(dx(dx(u)))(x=Lx) = 0")

# Initial conditions
x = dist.local_grid(xbasis)
if load_ic == False:
  u['g'] = 0.1+0.2*(1+np.tanh((x-Lx/2)/0.5))/2#np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)
else: 
  u['g'] = Q
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
u.change_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 1000 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        u.change_scales(1); plt.clf(); plt.plot(x,u['g']); plt.ylim(0,1); plt.pause(0.001)        
    if solver.iteration % 25 == 0:
        u.change_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)

# Plot
plt.figure(figsize=(6, 4))
plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
plt.xlim(0, Lx)
plt.ylim(0, stop_sim_time)
plt.xlabel('x')
plt.ylabel('t')
#plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
plt.tight_layout()
plt.show()
#plt.savefig('kdv_burgers.pdf')
#plt.savefig('kdv_burgers.png', dpi=200)

