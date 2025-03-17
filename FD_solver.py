## THIS SCRIPT IMPLEMENTS AN IMPLICIT SOLVER FOR THE HEAT EQUATION 
## WITH NEUMANN BOUNDARY CONDITIONS
## CF. https://math.stackexchange.com/questions/2706701/neumann-boundary-conditions-in-finite-difference

import matplotlib.pyplot as plt
import numpy as np
import time

N  = len(np.load('x.npy')); print('N=',N)  #5000
dx = np.diff(np.load('x.npy'))[0]#0.01
x  = np.linspace(0,N*dx,N)

Nt = 500000
dt = 0.001

#INITIAL CONDITION
u = np.load('Q.npy')

def rhs(u):
    return np.gradient((1-3*u**2)/(1-u**2)*np.gradient(u))/dx**2 - np.gradient(np.gradient(np.gradient(np.gradient(u))))/dx**4

#### Neumann BCs, slope A  @ x=0, slope  B at x=N*dx
#### Dirichlet BCs, Q = k0 @ x=0, Q = k1   @  x=N*dx  
A = 0
B = 0
k0 = u[0]
k1 = u[-1]

u[0] = u[1]
u[-2] = u[-1]

########################################
#DEFINE MATRIX M     in     M u_n = b
M = np.diag(np.ones(N)) 

for i in range(2,N-2):
    M[i,i-2] +=  1*dt/dx**4
    M[i,i-1] += -4*dt/dx**4
    M[i,i]   +=  6*dt/dx**4
    M[i,i+1] += -4*dt/dx**4
    M[i,i+2] +=  1*dt/dx**4

M[1,2]   += -1
M[-2,-3] += -1
##########################################
#np.set_printoptions(threshold=sys.maxsize)
#print(M)

#ONLY NEED TO COMPUTE INVERSE MATRIX ONCE
ts = time.time()
M_inv = np.linalg.inv(M)
print('Inverting matrix took ',time.time()-ts,' s')

#CADENCE FOR PLOTTING
cadence = 1000
plt.plot(x,u); plt.pause(0.01)
for n in range(Nt):
   #DEFINE RIGHT-HAND SIDE:
   b    = np.zeros(N)
   b[0] = k0     
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
   NL          = dt*np.gradient(NL_flux)/dx
   NL[0:2]     = 0
   NL[-2:]     = 0
   b += NL

   #IMPLICIT TIME STEP
   u = M_inv.dot(b)

   #PLOT RESULT
   if n % cadence ==0:
     print('<k>=',np.mean(u))
     plt.clf(); plt.ylim(min(u)-0.1,1); plt.plot(x,u); plt.xlabel('$x$'); plt.ylabel('$u$'); plt.pause(0.01)
