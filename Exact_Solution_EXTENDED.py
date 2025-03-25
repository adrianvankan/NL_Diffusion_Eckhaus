import numpy as np
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

def Qprime2(Q,C1,C2):
   return 3*Q**2 + 2*(Q-1)*np.log(abs(1 - Q)) - 2*(1 + Q)*np.log(abs(1 + Q))+C1*Q + C2;

C1 = -0.5
Q = np.linspace(0,1,1000000)#000)
dQdx = np.sqrt(Qprime2(Q,C1,0)); dQdx[np.isnan(dQdx)] = 0
dQdx_relevant = dQdx[dQdx>0]
Nrel          = len(dQdx_relevant)

n = 7 #times to repeat profile

dQdx_extended = np.zeros(len(dQdx_relevant)*n);
Q_extended = np.zeros_like(dQdx_extended)

for i in range(n):
  if i % 2 == 0:
    dQdx_extended[i*Nrel:(i+1)*Nrel] = dQdx_relevant#[::(-1)**i]
    Q_extended[i*Nrel:(i+1)*Nrel] = Q[dQdx>0]
  elif i % 2 == 1:
    dQdx_extended[i*Nrel:(i+1)*Nrel] = -dQdx_relevant[::-1]#[::(-1)**i]
    Q_extended[i*Nrel:(i+1)*Nrel] = Q[dQdx>0][::-1]

#PERFORM INTEGRAL
dQ = np.gradient(Q_extended)
x = np.cumsum(dQ/dQdx_extended)
  
plt.figure(layout='constrained'); plt.plot(Q,dQdx,label='$C_1=$'+str(C1)); 
plt.xlabel('$Q$'); plt.ylabel('$Q\'(x)$'); plt.legend(loc='upper left'); plt.ylim(ymin=0); plt.xlim(xmin=0)
plt.figure(layout='constrained')
plt.xlabel('$x$'); plt.ylabel('$Q$')
plt.plot(x,Q_extended)
'''
#plt.plot(x,np.gradient(Q_extended)/np.gradient(x));
dQdx_num = np.gradient(Q_extended)/np.gradient(x)
#plt.plot(x,dQdx_extended,'k--')
d2Qdx2_num = np.gradient(dQdx_extended)/np.gradient(x);
#plt.plot(x,d2Qdx2_num); 
Q2prime = -np.log(1+Q_extended)+np.log(1-Q_extended) +3*Q_extended +C1/2
#plt.plot(x,Q2prime,'k--');
'''

#INTERPOLATION TO EQUISPACED x-GRID
f = interpolate.interp1d(x,Q_extended) 
N = n*300 #NUMBER OF x-POINTS
xnew = np.linspace(x[0],x[-1],N) #NEW x-AXIS
ynew = f(xnew)   #  EVALUATE Q ON NEW x-AXIS USING `interp1d`

plt.figure(2,layout='constrained')
plt.plot(xnew,ynew, color='red',marker='o',ls='--')
plt.xlabel('$x$'); plt.ylabel('$Q$')
np.save('x.npy',xnew)
np.save('Q.npy',ynew)

plt.show()
