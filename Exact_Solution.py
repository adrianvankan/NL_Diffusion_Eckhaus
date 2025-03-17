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

def Qprime(Q,C1,C2):
   return 3*Q**2 + 2*(Q-1)*np.log(abs(1 - Q)) - 2*(1 + Q)*np.log(abs(1 + Q))+C1*Q + C2;

C1 = -0.5
Q = np.linspace(0.4,1,1000000)#000)
dQdx = np.sqrt(Qprime(Q,C1,0)); dQdx[np.isnan(dQdx)] = 0

#PERFORM INTEGRAL
dQ = np.diff(Q)[0]
x = np.cumsum(dQ/dQdx[dQdx>0])

plt.figure(layout='constrained'); plt.plot(Q,dQdx,label='$C_1=$'+str(C1)); 
plt.xlabel('$Q$'); plt.ylabel('$Q\'(x)$'); plt.legend(loc='upper left'); plt.ylim(ymin=0); plt.xlim(xmin=0)


#INTERPOLATION TO EQUISPACED x-GRID
f = interpolate.interp1d(x,Q[dQdx>0]) 
N = 500 #NUMBER OF x-POINTS
xnew = np.linspace(x[0],x[-1],N) #NEW x-AXIS
ynew = f(xnew)   #  EVALUATE Q ON NEW x-AXIS USING `interp1d`

plt.figure(2,layout='constrained')
plt.plot(xnew,ynew, color='red',marker='o',ls='--')
plt.xlabel('$x$'); plt.ylabel('$Q$')
np.save('x.npy',xnew)
np.save('Q.npy',ynew)
plt.show()
