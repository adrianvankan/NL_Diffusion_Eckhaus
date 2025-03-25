####################################
###### Author: Adrian van Kan ######
####################################
# This code solves the nonlinear diffusion equation dk/dt = d/dx( (3*m-k**2)/(m-k**2) dk/dx) - d^4/dx^4 k
# using a Fourier spectral method on a periodic domain, using an implicit-explicit time stepping scheme.


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
#from numba import jit
from time import time


L = np.load('x.npy')[-1] - np.load('x.npy')[0] #10*np.pi
N  = len(np.load('x.npy')) #32*1024
x = np.linspace(0,L*(N-1)/N,N);

wn = np.zeros(N); 
wn[0:N//2] = 2*np.pi/L*np.arange(0,N//2); 
wn[N//2:]  = 2*np.pi/L*(np.arange(0,N//2)-N//2)

#########################
#TEST THAT FFT WORKS
#y = np.sin(x)
#dydx = np.real(ifft(1j*wn*fft(y)))
#plt.plot(x,dydx); plt.plot(x,np.cos(x),'k--')
#plt.show()
#########################

ts = time()
#@jit
def rhs_exp_hat(k):
    dkdx = np.real(ifft(1j*wn*fft(k)))
    return 1j*wn*fft((3-k**2)/(1-k**2)*dkdx)
#@jit
def increment(k,dt):
    tmp = (fft(k)+dt*rhs_exp_hat(k))/(1+wn**4*dt)
    return np.real(ifft(tmp))

#TOTAL NUMBER OF TIMESTEPS
Nt = 3000
#SIZE OF TIMESTEP
dt = 1e-3

#INITIAL CONDITION (CHANGE THIS TO START FROM A DIFFERENT SOLUTION)
#lamb = L/2
#k = 10.0*np.sin(2*np.pi*(x-L/2)/lamb)*np.exp(-(x-L/2)**2/(2*lamb**2))
k = np.load('Q.npy')

save_cadence  = 1
plot_cadence  = 1
ks = np.zeros((N,Nt//save_cadence))

for i in range(1,Nt):
    k = increment(k,dt) #dt*np.real(ifft(rhs_exp_hat(k,mu)/(1+wn**4*dt)))

    if i % save_cadence == 0:
       ks[:,int(i/save_cadence)] = k
       #print(max(k))
       plt.clf()
       plt.xlim(0,L); plt.ylim(0,1)
       
    if i % plot_cadence == 0:
       plt.plot(x,k)
       print(max(k))
       plt.pause(0.001)

print(str(time()-ts)+' seconds taken')
plt.figure()
ts = np.linspace(0,Nt*dt,Nt//save_cadence)
xs = np.linspace(0,L,N)
np.save('ks.npy',ks)
X,T = np.meshgrid(xs,ts)
print(np.shape(X),np.shape(ks))
plt.pcolormesh(X,T,np.transpose(ks),cmap='seismic'); plt.colorbar();

plt.figure()
_   = np.histogram(ks)
bcs = (_[0][1::]+_[0][::-1])/2
plt.plot(bcs,_[1]); 
plt.show()
