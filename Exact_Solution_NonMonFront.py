# THIS SCRIPT GENERATES AN EXACT SOLUTION OF NON-MONOTONIC FRONT-TYPE FOR THE NONLINEAR DIFFUSION EQUATION

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy import interpolate
###############################################################################
######### pyplot settings settings ############################################
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)

#set font sizes
SMALL_SIZE = 22
MEDIUM_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
#####################

def Qprime_squared(Q,C1,C2): 
   return (3*Q**2 + 2*(Q-1)*log(abs(1 - Q)) - 2*(1 + Q)*log(abs(1 + Q)))+C1*Q + C2;

def C1C2_Qp0_and_Qpp0(ka):
   C1 = 2*(log(abs(1+ka))-log(abs(ka-1))) - 6*ka
   C2 = 2*(log(abs(1+ka))+log(abs(ka-1))) + 3*ka**2
   return C1, C2

def C1C2_Qp0_both_sides(ka,kb):
   C1 = (Qprime_squared(ka,0,0) - Qprime_squared(kb,0,0))/(kb-ka)
   C2 = -Qprime_squared(ka,C1,0)                                  #-3*ka**2 - C1*ka - 2*(ka-1)*log(1-ka) + 2*(ka+1)*log(1+ka) 
   #2*(kb*(ka-1)*np.log(abs(ka-1)) - kb*(ka+1)*np.log(abs(ka+1)) + ka*(1-kb)*np.log(abs(kb-1))+ka*(kb+1)*np.log(abs(kb+1)))/(ka-kb)+ 3*ka*kb
   return C1,C2        

def Q_vs_x(C1,C2,kmin,kmax,sign,color):
  Q = linspace(kmin,kmax,5000000)
  dQdx = sqrt(Qprime_squared(Q,C1,C2)); dQdx[isnan(dQdx)] = 0
  dQdx_relevant = dQdx[dQdx>0]
  Nrel          = len(dQdx_relevant)
   
  #PLOT PHASE SPACE 
  plt.figure(1,layout='constrained'); plt.plot(Q[dQdx>0],sign*dQdx_relevant,color=color); plt.xlabel('$Q$'); plt.ylabel('$dQ/dx$'); plt.pause(0.001)
  #plt.figure(2,layout='constrained'); plt.plot(Q,Qprime_squared(Q,C1,C2),color=color); plt.ylim(ymin=-0.1); plt.xlabel('$Q$'); plt.ylabel('$(dQ/dx)^2$'); plt.plot(Q,np.zeros_like(Q),'k--'); plt.pause(0.001)

  if sign == 1:
    dQdx_extended = dQdx_relevant
    Q_extended = Q[dQdx>0]
  elif sign == -1:
    dQdx_extended = -dQdx_relevant[::-1]
    Q_extended = Q[dQdx>0][::-1]

  dQdx_extended = dQdx_relevant
  Q_extended = Q[dQdx>0]

  dQ = gradient(Q_extended)
  x = cumsum(dQ/dQdx_extended)
  x -= x[0]

  #INTERPOLATION TO EQUISPACED x-GRID
  f = interpolate.interp1d(x,Q_extended) 
  N = 5000 #NUMBER OF x-POINTS
  xnew = linspace(x[0],x[-1],N) #NEW x-AXIS
  ynew = f(xnew)   #  EVALUATE Q ON NEW x-AXIS USING `interp1d`
  return xnew,ynew

######################################################################
#Leftmost segment
k0 = 0.3
C1, C2 = C1C2_Qp0_and_Qpp0(k0)
xnew,ynew = Q_vs_x(C1,C2,k0,1,1,'black')
#Rightmost segment
k2 = 0.4
C1, C2 = C1C2_Qp0_and_Qpp0(k2)
xnew4,ynew4 = Q_vs_x(C1,C2,k2,1,-1,'blue')
#Second second from left
k4 = ynew[-1]
k1 = 0.6
C1,C2 = C1C2_Qp0_both_sides(k4,k1)
xnew2,ynew2 = Q_vs_x(C1,C2,k1,k4,-1,'red')
xnew2 += xnew[-1]
#Second segment from right
k3 = max(ynew4)
C1,C2 = C1C2_Qp0_both_sides(k1,k3)
print(Qprime_squared(k1,C1,C2),Qprime_squared(k3,C1,C2))
xnew3,ynew3 = Q_vs_x(C1,C2,k1,k3,1,'green')
xnew3 += xnew2[-1]
xnew4 += max(xnew3)

#PLOT PROFILE
plt.figure(4,layout='constrained')
plt.plot(xnew,ynew, color='black',ls='-',marker='o')
plt.plot(xnew2,ynew2[::-1],color='red',marker='o')
plt.plot(xnew3,ynew3, color='green',ls='-',marker='o')
plt.plot(xnew4,ynew4[::-1], color='blue',marker='o')
plt.xlabel('$x$'); plt.ylabel('$Q$')
plt.ylim(0,1)

x = concatenate((xnew,xnew2,xnew3,xnew4))
y = concatenate((ynew,ynew2[::-1],ynew3,ynew4[::-1]))

#INTERPOLATION TO EQUISPACED x-GRID
f = interpolate.interp1d(x,y)#Q_extended)
dx = 0.05 
N  = int((x[-1]- x[0])/dx) #NUMBER OF x-POINTS
x_sample = linspace(x[0],x[-1],N) #NEW x-AXIS
y_sample = f(x_sample)   #  EVALUATE Q ON NEW x-AXIS USING `interp1d`
plt.plot(x_sample,y_sample,color='white',ls='--',lw=2)
plt.show() 

np.save('x.npy',x_sample)
np.save('Q.npy',y_sample)


'''
for i in range(n):
  if i % 2 == 0:
    dQdx_extended[i*Nrel:(i+1)*Nrel] = dQdx_relevant#[::(-1)**i]
    Q_extended[i*Nrel:(i+1)*Nrel] = Q[dQdx>0]
  elif i % 2 == 1:
    dQdx_extended[i*Nrel:(i+1)*Nrel] = -dQdx_relevant[::-1]#[::(-1)**i]
    Q_extended[i*Nrel:(i+1)*Nrel] = Q[dQdx>0][::-1]

plt.figure(layout='constrained'); plt.plot(Q,dQdx,label='$C_1=$'+str(C1));
plt.xlabel('$Q$'); plt.ylabel('$Q\'(x)$'); plt.legend(loc='upper left'); plt.ylim(ymin=0); plt.xlim(xmin=0)
plt.figure(layout='constrained')
plt.xlabel('$x$'); plt.ylabel('$Q$')
plt.plot(x,Q_extended)
#plt.plot(x,np.gradient(Q_extended)/np.gradient(x));
dQdx_num = np.gradient(Q_extended)/np.gradient(x)
#plt.plot(x,dQdx_extended,'k--')
d2Qdx2_num = np.gradient(dQdx_extended)/np.gradient(x);
#plt.plot(x,d2Qdx2_num);Q2prime = -np.log(1+Q_extended)+np.log(1-Q_extended) +3*Q_extended +C1/2
#plt.plot(x,Q2prime,'k--');
'''
