#The analytical solution of the problem
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



#-------------------------------------------------------
# 2d cartesian cooedinates grid
def cart_grid(x_I=0., x_F=1., Nx=100, y_I=0, y_F=1, Ny=100):

    
    X = np.linspace(x_I, x_F, Nx+1)         #1d grids in each dirrection        
    Y = np.linspace(y_I, y_F, Ny+1)

    dx = (x_F - x_I)/Nx                     #step sizes
    dy = (y_F - y_I)/Ny

    x, y = np.meshgrid(X, Y)                #2d meshgrid

    return X, Y, x, y, dx, dy

#-------------------------------------------------------
#2d polar coordinates grid
def polar_grid(r_I=0., r_F=1., Nr=100, th_I=0, th_F=2*np.pi, Nth=100):

    
    R = np.linspace(r_I, r_F, Nr+1)           #1d grids in each dirrection        
    Th = np.linspace(th_I, th_F, Nth+1)

    dr = (r_F - r_I)/Nr                       #step sizes
    dth = (th_F - th_I)/Nth

    r, th = np.meshgrid(R, Th)                #2d meshgrid

    return R, Th, r, th, dr, dth


def plot(x, y, u, u_an=None, conv_hist=None):

    #Creat 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    #Scatter plot of  u
    scat = ax.scatter(x, y, u, c = 'r', label = 'numerical', alpha = 1)
    
    #Surface plot of analytical solution if provided
    if u_an is not None:
        
        surf = ax.plot_surface(x, y, u_an, cmap = cm.coolwarm, label = 'analytical', alpha = 0.8)
        #Commands to fix a bug in the legend
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
    
    #Make plot nice and beautifull
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('$u = \sin\pi x \sin\pi y + x$')

    ax.legend(loc = 2)

    if conv_hist is not None:

        fig1, ax1 = plt.subplots()

        ax1.plot(conv_hist, c='b')

        ax1.set_xlabel('$iteration$')
        ax1.set_ylabel('$\\frac{||u^{(k+1)}-u^{(k)}||_2}{||u^{(k)}||_2}$', rotation='horizontal')


        ax1.set_title('$Convergence rate$')

        ax1.set_yscale('log')
    

    plt.show()