"""
Dedalus script computing the eigenmodes of waves on a clamped string.
This script demonstrates solving a 1D eigenvalue problem and produces
a plot of the relative error of the eigenvalues.  It should take just
a few seconds to run (serial only).

We use a Chebyshev basis to solve the EVP:
    s*u + L(u)         = 0
    dx(u)(x=0)         = 0
    dx(u)(x=Lx)        = 0
    dx(dx(dx(u)))(x=0) = 0
    dx(dx(dx(u)))(x=Lx)=0
where s is the eigenvalue and L is the linearization of the right-hand side of the nonlinear diffusion equation
about a given steady solution u0(x).

For the fourth derivative on a closed interval, we need four tau terms.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import scipy.interpolate as interpolate 
import time
ts = time.time()

plotting = True

logger = logging.getLogger(__name__)
direc = './'
fn    = ''  #'nonmon_front_clean'#'two_pulse'#'single_pulse_045'
Qold  = np.load(direc+'Q_'+fn+'.npy')
xold  = np.load(direc+'x_'+fn+'.npy'); xold-= xold[0]
if plotting: plt.plot(xold,Qold); plt.title('background_state'); plt.show()

# Parameters
Lx = np.max(xold) #np.pi
Nx = 384

kappa = 1
dtype = np.complex128

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx))

# Fields
u = dist.Field(name='u', bases=xbasis)
tau1 = dist.Field(name='tau1')
tau2 = dist.Field(name='tau2')
tau3 = dist.Field(name='tau3')
tau4 = dist.Field(name='tau4')
tau5 = dist.Field(name='tau5')
s = dist.Field(name='s')
D = dist.Field(name='D', bases=xbasis)
dDdQ = dist.Field(name='dDdQ', bases=xbasis)
ddDddQ = dist.Field(name='ddDddQ', bases=xbasis)
Q      = dist.Field(name='Q', bases=xbasis)

# Tau polynomials
tau_basis = xbasis.derivative_basis(2)
p1 = dist.Field(bases=tau_basis)
p2 = dist.Field(bases=tau_basis)
p3 = dist.Field(bases=tau_basis)
p4 = dist.Field(bases=tau_basis)
p5 = dist.Field(bases=tau_basis)
p1['c'][-1] = 1
p2['c'][-2] = 1
p3['c'][-3] = 1
p4['c'][-4] = 1

# Substitutions
dx       = lambda A: d3.Differentiate(A, xcoord)
x        = dist.local_grid(xbasis)
Qnew     = np.interp(x,xold,Qold) #fun(x) 

Q['g']      = Qnew
D['g']      = (1-3*Qnew**2)/(1-Qnew**2)
dDdQ['g']   = -4*Qnew/(1-Qnew**2)**2
ddDddQ['g'] = 4*(3*Qnew**2+1)/(Qnew**2-1)**3


#if plotting: plt.plot(x,Q['g']); plt.show()
#plt.plot(x,D['g']); plt.plot(x,dDdQ['g']); plt.plot(x,ddDddQ['g']); 

# Problem
#problem = d3.EVP([u,tau1,tau2,tau3,tau4], eigenvalue=s, namespace=locals())
problem = d3.EVP([u,tau1,tau2,tau3,tau4], eigenvalue=s, namespace=locals())
#problem.add_equation("s*u + dx(dx(dx(dx(u)))) +  tau1*p1 + tau2*p2 + tau3*p3 + tau4*p4  = 0")
problem.add_equation("s*u - ddDddQ*dx(Q)**2*u - 2*dDdQ*dx(Q)*dx(u) - dDdQ*dx(dx(Q))*u - D*dx(dx(u)) + kappa*dx(dx(dx(dx(u)))) +  tau1*p1 + tau2*p2 + tau3*p3 + tau4*p4  = 0")
problem.add_equation("dx(u)(x=0) = 0")
problem.add_equation("dx(u)(x=Lx) = 0")
problem.add_equation("dx(dx(dx(u)))(x=0)  = 0")
problem.add_equation("dx(dx(dx(u)))(x=Lx) = 0")

# Solve
solver = problem.build_solver()
solver.solve_dense(solver.subpro
##################################
# Check analytical result for constant coefficients
#print(np.sort_complex(np.sort(solver.eigenvalues))/np.pi**4)
##################################

#Plot most unstable mode
plt.figure(figsize=(6, 4),layout='constrained')
x   = dist.local_grid(xbasis)
ev_tmp = np.array([np.real(ev_i) for ev_i in solver.eigenvalues])
ev_tmp[np.isinf(ev_tmp)] = 0.0
ev_max  = np.max(ev_tmp) 
for n, idx in enumerate(np.argsort(solver.eigenvalues)[::-1][:20]):
    solver.set_state(idx, solver.subsystems[0])
    u_rescaled = u['g'].real/np.max(abs(u['g'].real));
    u_rescaled /= max(abs(u_rescaled))*u_rescaled[0]
    ev_i = solver.eigenvalues[idx]
    if (not np.isinf(ev_i)) and np.real(ev_i) > 0.99*ev_max: 
        print(ev_i)
        if plotting: plt.plot(x, u_rescaled, label='eig='+str(solver.eigenvalues[idx]))

print('runtime = '+str(time.time()-ts)+' s')
if plotting:
  plt.xlim(0, Lx)
  plt.legend(loc="lower right")
  plt.ylabel(r"mode structure")
  plt.xlabel(r"x")
  plt.tight_layout()
  plt.plot(x,np.zeros_like(x),'k--')
  plt.show()
