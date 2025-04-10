import matplotlib.pyplot as plt
import numpy as np

Nx = 3000
dx = 0.1
x  = np.linspace(0,Nx*dx,Nx)

Nt = 100000
dt = 0.01

def rhs(u,dx):
    #al = 0.1
    c = 0.2
    nu = 0.1 
    result = np.gradient(nu*(1-3*u**2)/(1-u**2)*np.gradient(u)/dx + c*u**2)/dx
    return result

def RK4_increment(u,dt,dx):
    k1 = rhs(u,dx)*dt
    k2 = rhs(u+k1/2,dx)*dt
    k3 = rhs(u+k2/2,dx)*dt
    k4 = rhs(u+k3,dx)*dt
    increment = (k1+2*k2+2*k3+k4)/6
    return increment

cadence = 1000
plot_1D = False

#INITIAL CONDITION
u0 = 0.2
du = 0.3
u  = u0 + du/2 * (1+np.tanh(x-2*x[-1]/3)) #0.2+0.1*np.cos(x)#0.4*np.exp(-(x-Nx*dx/2)**2)

us = np.zeros((Nx,Nt//cadence))
for i in range(Nt):
  if i % cadence == 0:
    print('i/Nt=',i/Nt)
    if plot_1D == True:
       plt.clf();  plt.plot(x,u); plt.ylim(0,1/np.sqrt(3)); plt.pause(0.01);
    us[:,i//cadence] = u
  u += RK4_increment(u,dt,dx)

  #IMPOSING NEUMANN BCs
  u[1] = u[0]
  u[-2] = u[-1]

t_subsamp = np.linspace(0,Nt*dt,Nt//cadence)
plt.clf()
plt.pcolormesh(x,t_subsamp, np.transpose(us),cmap='seismic'); plt.colorbar(); plt.show()
