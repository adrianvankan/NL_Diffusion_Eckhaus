from numpy import *
import matplotlib.pyplot as plt
import time
#Compute 2nd-order-accurate differentiation matrices on `n`+1 points
#in the interval `xspan`. Returns a vector of nodes and the matrices
#for the first and second derivatives.
def diffmat2(n,xspan):
   a,b = xspan
   h = (b-a)/n
   x = [ a + i*h for i in range(0,n) ]   # nodes
    
   # Define most of D by its diagonals.
   D = zeros((n,n))
   for i in range(1,n-1):
       D[i,i-1] = -1/(2*h)
       D[i,i+1] =  1/(2*h)   
   # fix first and last rows.
   D[0,0:3] = array([-1.5,2,-0.5])/h
   D[-1,-3:] = array([0.5,-2,1.5])/h
   #Uncomment for first-order FD at boundary
   #D[0,0:2]  = array([1,-1])/h
   #D[-1,-2:] = array([1,-1])/h

   #Define most of D2 by its diagonals
   D2 = -2*diag(ones(n))/h**2
   for i in range(1,n-1):
       D2[i,i-1] =  1/h**2
       D2[i,i+1] =  1/h**2
   # fix first and last rows.
   D2[0,0:4] = array([2,-5,4,-1])/h**2
   D2[-1,-4:] = array([-1,4,-5,2])/h**2 

   #Define most of D4 by its diagonals
   D4 = zeros((n,n))#identity(n)*(-4)/h**4
   for i in range(2,n-3):
       D4[i,range(i-1,i+4)] = array([1,-4,6,-4,1])/h**4

   #fix first and last two rows
   D4[0,0:6]  = array([3, -14, 26, -24, 11, -2])/h**4
   D4[-1,-6:] = array([-2, 11, -24, 26, -14, 3])/h**4
   return x, D, D2, D4

#############################################
## TEST OF DIFFERENTIATION MATRICES (SUCCESSFUL)
#x,D,D2,D4 = diffmat2(6000,[0,2*pi])
#y = sin(x)
#dydx   = D.dot(y)
#d2ydx2 = D2.dot(y)
#d4ydx4 = D2.dot(D2.dot(y))
#d4ydx4_alt = D4.dot(y)
#plt.plot(x,y); plt.plot(x,dydx); 
#plt.plot(x,d2ydx2); 
#plt.plot(x,d4ydx4,'r-'); 
#plt.plot(x,d4ydx4_alt,'--'); plt.show()
#############################################

x =  load('x.npy')
Q0 = load('Q.npy')

x,D,D2,D4 = diffmat2(len(x),[x[0],x[-1]])

L = D.dot((1-Q0**2)*((1-3*Q0**2)*D + 6*Q0**2 *D.dot(Q0).dot(identity(len(x)))) + (1-3*Q0**2)*(2*Q0*D.dot(Q0))) - D2.dot(D2)

#BOUNDARY CONDITIONS
L[0,:] = 0; L[1,:] = 0   
L[-1,:] = 0; L[-2,:] = 0

ts = time.time()
eigenvalues, eigenvectors = linalg.eig(L)
#print(max(real(eigenvalues)),where(real(eigenvalues)==max(real(eigenvalues))))
ind = where(real(eigenvalues)==max(real(eigenvalues)))[0]
print('Finding eigenvalues/eigenvectors took',time.time()-ts,' s')

#PLOT EIGENVALUES
plt.plot(real(eigenvalues),'x'); plt.plot(zeros(len(x))); plt.ylim(-100,100); plt.ylabel('Eigenvalue'); plt.xlabel('Index')

plt.figure(); 
#PLOT EIGENFUNCTIONS OVER SOME RANGE OF INDICES
for i in range(len(x)-10,len(x)-5):
    plt.plot(x,eigenvalues[i]*eigenvectors[:,i],'k--',zorder=100)
    plt.plot(x,L.dot(eigenvectors[:,i]),'-',label='i='+str(i),lw=3); 
plt.legend()
plt.xlabel('$x$'); plt.ylabel('Eigenfunction'); plt.show()
