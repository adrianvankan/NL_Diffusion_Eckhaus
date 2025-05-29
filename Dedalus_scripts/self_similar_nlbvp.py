### This code solves the nonlinear equation derived by a similarity ansatz to the 
### nonlinear diffusion equation

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from scipy.interpolate import interp1d
logger = logging.getLogger(__name__)

# Parameters
N = 192       #resolution: number of Chebyshev polynomials
Lx = 10.      #domain size in \eta
c = 1         #constant in BC
ncc_cutoff = 1e-5    #numerical constant
restart = True       #restart from specified initial guess
tolerance = 3e-9     
dealias = 3/2      
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.Chebyshev(xcoord,size=N, bounds=(0,Lx))

# Fields
u = dist.Field(name='u', bases=xbasis)
coeff = dist.Field(name='coeff', bases=xbasis)
tau1 = dist.Field(name='tau1')
tau2 = dist.Field(name='tau2')
tau3 = dist.Field(name='tau3')
tau4 = dist.Field(name='tau4')

# Tau polynomials
tau_basis = xbasis.derivative_basis(4)
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
x = dist.local_grid(xbasis)  #this is \eta
coeff['g'] = x               
#np.save('initial_x.npy',x)

# Problem
problem = d3.NLBVP([u,tau1,tau2,tau3,tau4],namespace=locals())
problem.add_equation("tau1*p1 + tau2*p2 + tau3*p3 + tau4*p4 = -2*u**3 - 4*u**2*dx(dx(dx(dx(u)))) + coeff*u**2*dx(u) + 4*u*dx(dx(u)) - 4*dx(u)**2")
problem.add_equation("dx(u)(x=0) = 0")
problem.add_equation("dx(dx(dx(u)))(x=0)  = 0")
problem.add_equation("u(x=Lx) = c*Lx**2")
problem.add_equation("dx(dx(u))(x=Lx) = 2*c")

# Initial guess
if restart == False:
  u['g'] = c*x**4/(10+x**2) #R0**(2/(n-1)) * (1 - r**2)**2
elif restart == True:
  tmp   = np.load('initial_guess_corr.npy')
  tmp   = np.insert([tmp[0]],1,tmp)
  tmp   = np.insert([c*Lx**2],0,tmp)
  x_old = np.load('initial_x_corr.npy')
  x_old = np.insert([0.],1,x_old)
  x_old = np.insert([Lx],0,x_old)
  fun   = interp1d(x_old,tmp)
  u['g'] = fun(x) #regridded initial guess
#plt.plot(x,u['g']); plt.show()

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
pert_norm = np.inf
u.change_scales(dealias)
steps = [u['g'].ravel().copy()]

alpha = np.linspace(0.2, 1, 10000)
color = ('C0',) * (10000-1) + ('C1',)
while pert_norm > tolerance:
    #print(pert_norm)
    solver.newton_iteration()
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Perturbation norm: {pert_norm:.3e}')
    u0 = u(x=0).evaluate()
    Ri = u0 
    logger.info(f'R iterate: {Ri}')
    steps.append(u['g'].ravel().copy())
    plt.plot(x,u['g'], c=color[solver.iteration], alpha=alpha[solver.iteration], label=f"step {solver.iteration}"); 
plt.loglog(x,u['g'], c='red',lw=3); plt.ylim(ymin=0); plt.xlim(xmin=0); plt.xlabel('$\\eta$'); plt.ylabel('$H(\\eta)$')
plt.plot(x[x>3],c*x[x>3]**2,'k--')
logger.info(f'Iterations: {solver.iteration}')
logger.info(f'Final R iteration: {Ri}')

if pert_norm < 1: 
    np.save('initial_guess_corr.npy',u['g']); 
    np.save('initial_x_corr.npy',x)
plt.show()
