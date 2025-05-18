"""
Dedalus script computing the eigenmodes of waves on a clamped string.
This script demonstrates solving a 1D eigenvalue problem and produces
a plot of the relative error of the eigenvalues.  It should take just
a few seconds to run (serial only).

We use a Chebyshev basis to solve the EVP:
    s*u + dx(dx(u)) = 0
    u(x=0) = 0
    u(x=Lx) = 0
where s is the eigenvalue.

For the second derivative on a closed interval, we need two tau terms.
Here we choose to use a first-order formulation, putting one tau term
on an auxiliary first-order variable and another in the PDE, and lifting
both to the first derivative basis.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import scipy.interpolate as interpolate 
logger = logging.getLogger(__name__)

xold   = np.load('x.npy')
# Parameters
Lx = np.max(xold) #np.pi
Nx = 256

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
s = dist.Field(name='s')
D = dist.Field(name='D', bases=xbasis)
dDdQ = dist.Field(name='dDdQ', bases=xbasis)
ddDddQ = dist.Field(name='ddDddQ', bases=xbasis)
Q      = dist.Field(name='Q', bases=xbasis)

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
x = dist.local_grid(xbasis)
Qold     = np.load('Q.npy')
x   = dist.local_grid(xbasis)
print(len(xold),len(Qold))
fun = interpolate.interp1d(xold,Qold)
Qnew        = fun(x)#0.5*np.ones_like(x) #fun(x)

Q['g']      = Qnew
D['g']      = (1-3*Qnew**2)/(1-Qnew**2)
dDdQ['g']   = -4*Qnew/(1-Qnew**2)**2
ddDddQ['g'] = 4*(3*Qnew**2+1)/(Qnew**2-1)**3

# Problem
problem = d3.EVP([u,tau1,tau2,tau3,tau4], eigenvalue=s, namespace=locals())
problem.add_equation("s*u - ddDddQ*dx(Q)**2*u - 2*dDdQ*dx(Q)*dx(u) - dDdQ*dx(dx(Q))*u - D*dx(dx(u)) + kappa*dx(dx(dx(dx(u)))) +  tau1*p1 + tau2*p2 + tau3*p3 + tau4*p4 = 0") 
problem.add_equation("dx(u)(x=0) = 0")
problem.add_equation("dx(u)(x=Lx) = 0")
problem.add_equation("dx(dx(dx(u)))(x=0)  = 0")
problem.add_equation("dx(dx(dx(u)))(x=Lx) = 0")

# Solve
solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0])
evals = solver.eigenvalues
evals = evals[np.argsort(1/solver.eigenvalues)]
evals[np.isinf(evals)] = np.nan
n = 1 + np.arange(evals.size)

# Plot EVs
plt.figure(figsize=(6, 4))
plt.plot(np.sort(evals),'o')
#plt.loglog(-true_evals)
plt.ylim(-10,10)
plt.plot(x,np.zeros_like(x))
plt.xlabel("eigenvalue number")
plt.ylabel("eigenvalue")
plt.tight_layout()

#Plot most unstable mode
plt.figure(figsize=(6, 4),layout='constrained')
x   = dist.local_grid(xbasis)
for n, idx in enumerate(np.argsort(1/solver.eigenvalues)[-1:], start=1):
    solver.set_state(idx, solver.subsystems[0])
    u_rescaled = u['g'].real/np.max(abs(u['g'].real));
    u_rescaled /= -max(abs(u_rescaled))*u_rescaled[0]
    plt.plot(x, u_rescaled, label='eig='+str(solver.eigenvalues[idx]))#f"idx={idx}")

plt.xlim(10, Lx-10)
plt.legend(loc="lower right")
plt.ylabel(r"mode structure")
plt.xlabel(r"x")
plt.tight_layout()
plt.ylim(-0.075,0.075)
plt.plot(x,np.zeros_like(x),'k--')
plt.show()
