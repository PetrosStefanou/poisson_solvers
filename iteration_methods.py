
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse


def numpy_jacobi(matrix, source, grid, init_guess=None, boundary=((0,0),(0,0)), tolerance=1.e-5, itermax=50):

    ''' 
    Solution of 2D Poisson equation using the jacobi iteration method. The solution in each grid point in the (k+1)th iteration is
    calculated by the the values of the closest neighbours in the kth iteration. 
    We assume dirichlet boundary conditions (neumann will be added in the future). This is an optimised version that uses numpy
    smart indexing instead of straightforward nested loops.

    ----------------------

    input
    matrix: (Nx*Ny)x(Nx*Ny) array. The matrix of the linear algebra problem. For the time being this parameter is                                                        
                                   redundant, we do not make use of it. 
    source: (Nx+1)x(Ny+1) array. The source term of the problem.
    grid: Tuple of the form (x, y). The grid where we solve the problem. It is provided as a meshgrid.
    init_guess: (Nx+1)x(Ny+1) array. The initial guess to start the iterative method. Default is None for which an array of ones is                                       
                                     assigned.
    boundary:tuple of two tuples of length 2. The set of Dirichlet boundary conditions in the form 
                                              ((u(x_I,y), u(x_F, y)), (u(x,y_I), u(x, y_F))). Default is zero.
    tolerance: float. The level of tolerance to set a stopping criterion. Default is 10^-5
    itermax: integer. The maximum number of iterations before returning a result if the desired tolerance has not been achieved.

    output
    u: (Nx+1)x(Ny+1) array. The solution of the problem.
    k: integer. The number of iteration that the solver performed.
    rel_diff: float. The ratio of the L2 norm of the difference of (k_final) and (k_final-1) iteration over the L2 norm of the                      
                     (k_final-1) iteration.
    conv_hist: array. An array containg all the values of rel_diff to check the convergence history.
    '''

    #read the input
    M, f, u0, B, tol, kmax = matrix, source, init_guess, boundary, tolerance, itermax

    x, y = grid

    Nx = x.shape[1] - 1
    Ny = y.shape[0] - 1
    
    dx = (x[0, -1] - x[0, 0])/Nx
    dy = (y[-1, 0] - y[0, 0])/Ny
    
    #assign zero initial guess if non provided
    if init_guess is None:

        u0 = np.ones_like(f)

    #assign dirichlet boundary conditions
    u0[:, 0] = B[0][0]                               
    u0[:, -1] = B[0][1]
    u0[0, :] = B[1][0]
    u0[-1, :] = B[1][1]

    #initial values before the iteration loop starts
    u = u0
    k = 0
    rel_diff = tol +1
    conv_hist = []
    
    #iteration loop
    while k < kmax and rel_diff > tol:

        u_next = u.copy()

        #The jacobi method using numpy indexing
        u_next[1:-1, 1:-1] = ((u[:-2, 1:-1] + u[2:, 1:-1])*dx**2 + (u[1:-1, :-2] + u[1:-1, 2:])*dy**2 -
                              dx**2*dy**2*f[1:-1, 1:-1])/(2*(dx**2+dy**2))

        #stopping criterion
        
        rel_diff = la.norm(u_next - u)/la.norm(u)
        
        conv_hist.append(rel_diff)

        k += 1

        #update the value of the solution
        u = u_next

    return u, k, rel_diff, np.array(conv_hist)

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def slow_jacobi(matrix, source, grid, init_guess = None, boundary=((0,0),(0,0)), tolerance = 1.e-5, itermax = 200):

    ''' 
    Solution of 2D Poisson equation using the jacobi iteration method. The solution in each grid point in the (k+1)th iteration is        
    calculated by the the values of the closest neighbours in the kth iteration. 
    We assume dirichlet boundary conditions (neumann will be added in the future). This is a very slow version that uses nested
    loops. It use is for comparisons or it needs to be optimised by e.g jit. 

    ----------------------

    input
    matrix: (Nx*Ny)x(Nx*Ny) array. The matrix of the linear algebra problem. For the time being this parameter is                                                        
                                   redundant, we do not make use of it. 

    source: (Nx+1)x(Ny+1) array. The source term of the problem.
    grid: Tuple of the form (x, y). The grid where we solve the problem. It is provided as a meshgrid.
    init_guess: (Nx+1)x(Ny+1) array. The initial guess to start the iterative method. Default is None for 
                                     which an array of ones is assigned.
    boundary: tuple of two tuples of length 2. The set of Dirichlet boundary conditions in the form 
                                                ((u(x_I,y), u(x_F, y)), (u(x,y_I), u(x, y_F))). Default is zero.
    tolerance: float. The level of tolerance to set a stopping criterion. Default is 10^-5
    itermax: integer. The maximum number of iterations before returning a result if the desired tolerance has not been achieved.

    output
    u: (Nx+1)x(Ny+1) array. The solution of the problem.
    k: integer. The number of iteration that the solver performed.
    rel_diff: float. The ratio of the L2 norm of the difference of (k_final) and (k_final-1) iteration over the L2 norm of the                      
                     (k_final-1) iteration.
    conv_hist: array. An array containg all the values of rel_diff to check the convergence history.
    '''
    #read the input
    M, f, u0, B, tol, kmax = matrix, source, init_guess, boundary, tolerance, itermax

    x, y = grid

    Nx = x.shape[1] - 1
    Ny = y.shape[0] - 1

    dx = (x[0, -1] - x[0, 0])/Nx
    dy = (y[-1, 0] - y[0, 0])/Ny

    #assign zero initial guess if non provided
    if init_guess is None:

        u0 = np.ones_like(f)
        

    #assign dirichlet boundary conditions
    u0[:, 0] = B[0][0]                               
    u0[:, -1] = B[0][1]
    u0[0, :] = B[1][0]
    u0[-1, :] = B[1][1]

    #initial values before the iteration loop starts
    u = u0
    rel_diff = tol + 1
    k = 0
    conv_hist = []

    #iteration loop using nested loops.
    while  k < kmax and rel_diff > tol:

        u_next = u.copy()

        for j in range(1, Ny):
            for i in range(1, Nx):
                
                u_next[j, i] = ((u[j, i-1] + u[j, i+1])*dy**2 + 
                                (u[j-1, i] + u[j+1, i])*dx**2 - f[j,i]*dx**2*dy**2)/(2*(dx**2+dy**2))

        
        rel_diff = la.norm(u_next - u)/la.norm(u)
        
        conv_hist.append(rel_diff)
        
        # L2norm = la.norm(u_temp - u, 2)
        # maxnorm = la.norm(u_temp - u, np.inf)

        # print(abs_diff, L2norm, maxnorm)


        u = u_next
        k += 1
        # print(u, '\n')
    return u, k, rel_diff, np.array(conv_hist)


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# @jit(nopython=True)
def SOR(matrix, source, grid, init_guess=None, boundary=((0, 0), (0,0)), tolerance=1.e-8, itermax=1000, omega=1.5):

    M, f, u0, B, tol, kmax, w = matrix, source, init_guess, boundary, tolerance, itermax, omega

    x, y = grid

    Nx, Ny = x.shape[1]-1, y.shape[0]-1

    dx, dy = (x[0, -1] - x[0, 0])/Nx, (y[-1, 0] - y[0, 0])/Ny

    if init_guess is None:

        u0 = np.ones_like(f)

    #assign dirichlet boundary conditions
    u0[:, 0] = B[0][0]                               
    u0[:, -1] = B[0][1]
    u0[0, :] = B[1][0]
    u0[-1, :] = B[1][1]
    
    u = u0
    k = 0
    rel_diff = tol + 1
    conv_hist = []
       
    #iteration loop using nested loops.
    while  k < kmax and rel_diff > tol:

        u_next = u.copy()

        for j in range(1, Ny):
            for i in range(1, Nx):
                
                u_next[j, i] = (1-omega)*u[j,i] + omega*((u_next[j, i-1] + u[j, i+1])*dy**2 + 
                                (u_next[j-1, i] + u[j+1, i])*dx**2 - f[j,i]*dx**2*dy**2)/(2*(dx**2+dy**2))
        
        

        rel_diff = la.norm(u_next-u)/la.norm(u)
        conv_hist.append(rel_diff)
        u = u_next
        k += 1

    return u, k, rel_diff, conv_hist