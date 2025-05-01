## THIS SCRIPT IMPLEMENTS A SEMI-IMPLICIT FD SOLVER FOR THE NONLINEAR DIFFUSION EQUATION WITH TWO OPTIONS FOR THE BOUNDARY CONDITIONS
## EITHER NATURAL (u'=u'''=0 on boundary) OR Dirichlet + Neumann (NOT NATURAL, since <k> not conserved by diffusive terms)
## For an example of implicit FD solver https://math.stackexchange.com/questions/2706701/neumann-boundary-conditions-in-finite-difference

import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.sparse
import scipy.sparse.linalg

load_ic = True #switch to True to load saved ic

if load_ic:
  N  = len(np.load('x.npy')); print('N=',N)  
  dx = np.diff(np.load('x.npy'))[0]
else:
  N  = 1000  #Number of grid points
  dx = 0.05  #Space increment BEWARE: do not make dx too small or d/dt<k> due to diffusive terms does not vanish!
x  = np.linspace(0,N*dx,N)

Nt = 40000  #Number of timesteps
dt = 0.0025 #Time increment

BC = '13'   #choice of BCs: '01' (k=fixed, k'=0 at boundaries) or '13' (k'=k'''=0 at boundaries)

Dh   = 1   #hypverviscosity coefficient
c_nl = 0    #coefficient of advection term

#INITIAL CONDITION
if load_ic:     u = np.load('Q.npy')
if not load_ic: 
    u0 = 0.4
    du = 0.1
    u  = u0 + du/2 * (1+np.tanh(x-x[-1]/2))
    #u = 0.3 + 0.1/2*(np.tanh(x-N/2*dx)+1)#np.exp(-(x-N/2*dx)**2)
    u = np.array(u,np.longdouble)

#initialize field with zero slope at boundaries
u[0] = u[1]
u[-2] = u[-1]
if BC == '13':
  u[0:4] = u[0]
  u[-4:] = u[-1]

#### Neumann BCs, slope A  @ x=0, slope  B at x=N*dx
#### Dirichlet BCs, Q = k0 @ x=0, Q = k1   @  x=N*dx  
A = 0
B = 0
if BC == '01':
  k0 = u[0]  
  k1 = u[-1]
elif BC == '13':
  k0 = 0
  k1 = 0

########################################
#DEFINE MATRIX M     in     M u_n = b
M = np.diag(np.ones(N)) 

for i in range(2,N-2):
    M[i,i-2] +=  1*dt/dx**4*Dh
    M[i,i-1] += -4*dt/dx**4*Dh
    M[i,i]   +=  6*dt/dx**4*Dh
    M[i,i+1] += -4*dt/dx**4*Dh
    M[i,i+2] +=  1*dt/dx**4*Dh

M[ 1, 2]  += -1
M[-2,-3]  += -1

if BC == '13':
  M[0,0]     = 0;  M[0,:4]   += np.array([-1, 3, -3, 1])
  M[-1,-1]   = 0;  M[-1,-4:] += np.array([-1, 3, -3, 1])
##########################################
#print(M)

ts = time.time()
M_inv = np.linalg.inv(M)
print('Inverting matrix took ',time.time()-ts,' s')

#CADENCE FOR PLOTTING
cadence = 1
plot_while_running = True

#PLOT INITIAL CONDITION TO CHECK
us = np.zeros((len(x),Nt//cadence)); print(np.shape(us))

for n in range(Nt):
   #DEFINE RIGHT-HAND SIDE:
   b    = np.zeros(N)
   b[0]     = k0 
   b[1]     = - A*dx
   b[2:N-2] = u[2:N-2] 
   b[N-2]   = B*dx
   b[N-1]   = k1
   
   NL = np.zeros(N);
   dudx = np.zeros(N)
   dudx[1:N-2] = (u[3:N] - u[1:N-2])/(2*dx)
   dudx[0:2]   = 0
   dudx[-2:]   = 0
   NL_flux     = (1-3*u**2)/(1-u**2)*dudx 
   NL          = dt*np.gradient(NL_flux)/dx + dt*u*dudx*c_nl #- dt*(c_nl*(2*u0+du)/2)*dudx
   NL[0:2]     = 0
   NL[-2:]     = 0

   b += NL

   #IMPLICIT TIME STEP
   u = M_inv.dot(b)                      #THIS IS SIGNIFICANTLY FASTER IF M STAYS CONSTANT  
   #u = scipy.sparse.linalg.spsolve(M,b) #IGNORE THIS: IT IS SLOWER PROVIDED M_inv is always the same

   #PLOT AND SAVE RESULT
   if n % cadence ==0:
     print('n/N=',int(100*n/Nt),'%')
     us[:,n//cadence] = u
     print('<k>=',np.mean(u),'<k^2>=',np.mean(u**2))
     
     if plot_while_running:
        plt.clf(); plt.ylim(0,1); plt.plot(x,u); plt.xlabel('$x$'); plt.ylabel('$Q$'); plt.pause(0.01)

np.save('us.npy',us)

plt.clf(); plt.pcolormesh(x,dt*np.arange(Nt//cadence),np.transpose(us)); plt.show()
