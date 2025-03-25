import numpy as np
import matplotlib.pyplot as plt
import os
import moviepy
###############################################################################
######### pyplot settings settings ############################################
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

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

# Check if directory exists
if not os.path.exists('frames'):
  # Create directory
  os.makedirs('frames')

us     = np.load('us.npy')
x      = np.load('x.npy')
dt_net = 0.05

plot = True
if plot:
   plt.figure(layout='constrained') 
   for i in range(len(us[:,0])):
      plt.clf(); plt.plot(x,us[:,i]); 
      plt.xlim(0,max(x)); plt.ylim(-100,100); plt.plot(x,-1*np.ones_like(x),'k--'); plt.plot(x,np.ones_like(x),'k--')
      plt.xlabel('$x$'); plt.ylabel('$Q$')
      plt.savefig('frames/profile_'+str(i).zfill(4)+'.png',dpi=100)

path = 'frames/'
dir_list = os.listdir(path); dir_list = [path+d for d in dir_list]; print(dir_list)

fps      = 30
# Create video file from PNGs
print("Producing video file...")
#filename  = os.path.join(dir_list, 'demo1.mp4')
clip      = moviepy.ImageSequenceClip(dir_list, fps=fps)
clip.write_videofile('movie.mp4')
print("Done!")
