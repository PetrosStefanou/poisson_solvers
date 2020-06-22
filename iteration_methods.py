
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
    conv_hist_max = np.zeros(kmax)
        
    #iteration loop
    while k < kmax and rel_diff > tol:

        u_next = u.copy()

        #The jacobi method using numpy indexing
        u_next[1:-1, 1:-1] = ((u[:-2, 1:-1] + u[2:, 1:-1])*dx**2 + (u[1:-1, :-2] + u[1:-1, 2:])*dy**2 -
                              dx**2*dy**2*f[1:-1, 1:-1])/(2*(dx**2+dy**2))

        #stopping criterion
        
        rel_diff = la.norm(u_next - u)/la.norm(u)
        rel_diff_max = la.norm(u_next - u, ord=np.inf)/la.norm(u, ord=np.inf)

        conv_hist.append(rel_diff)
        conv_hist_max[k] = rel_diff_max

        k += 1

        #update the value of the solution
        u = u_next

    return u, k, rel_diff, np.array(conv_hist), conv_hist_max

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
    conv_hist_max = np.zeros(kmax)

    #iteration loop using nested loops.
    while  k < kmax and rel_diff > tol:

        u_next = u.copy()

        for j in range(1, Ny):
            for i in range(1, Nx):
                
                u_next[j, i] = ((u[j, i-1] + u[j, i+1])*dy**2 + 
                                (u[j-1, i] + u[j+1, i])*dx**2 - f[j,i]*dx**2*dy**2)/(2*(dx**2+dy**2))

        
        rel_diff = la.norm(u_next - u)/la.norm(u)
        rel_diff_max = la.norm(u_next - u, ord=np.inf)/la.norm(u, ord=np.inf)
        
        conv_hist.append(rel_diff)
        conv_hist_max[k] = rel_diff_max
        
        # L2norm = la.norm(u_temp - u, 2)
        # maxnorm = la.norm(u_temp - u, np.inf)

        # print(abs_diff, L2norm, maxnorm)


        u = u_next
        k += 1
        # print(u, '\n')
    return u, k, rel_diff, np.array(conv_hist), conv_hist_max




# @jit(nopython=True)
def SOR(matrix, source, grid, init_guess=None, boundary=((0, 0), (0,0)), tolerance=1.e-8, itermax=1000, omega=1.5):
    ''' 
    Solution of 2D Poisson equation using the Successive Over Relaxation iteration method in cartesian coordinates. 
    The solution in each grid point in the (k+1)th iteration is calculated by the the values of the closest 
    neighbours in their most recent available (k or k+1) iteration. 
    The method uses a linear combination of the current and next calculated value based on a parameter omega
    in order to drastically reduce the number of iterations needed to reach convergences.
    Can be configured to use dirichlet or neumann boundary conditions. This is a very slow version that uses nested
    loops. It needs to be optimised by e.g jit. 

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
    omega: float. The relaxation parameter. Its value is typically between 1<omega<2

    output
    u: (Nx+1)x(Ny+1) array. The solution of the problem.
    k: integer. The number of iteration that the solver performed.
    rel_diff: float. The ratio of the L2 norm of the difference of (k_final) and (k_final-1) iteration over the L2 norm of the                      
                     (k_final-1) iteration.
    conv_hist: array. An array containg all the values of rel_diff to check the convergence history.
    '''
    #read input
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
    
    #initial values before the iteration loop starts
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

    return u, k, rel_diff, conv_hist #, conv_hist_max

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def SOR_polar(matrix, source, grid, init_guess=None, boundary=((0, 0), (0,0)), tolerance=1.e-8, itermax=1000, omega=1.5):
    ''' 
    Solution of 2D Poisson equation using the Successive Over Relaxation iteration method in polar coordinates. 
    The solution in each grid point in the (k+1)th iteration is calculated by the the values of the closest 
    neighbours in their most recent available (k or k+1) iteration. 
    The method uses a linear combination of the current and next calculated value based on a parameter omega
    in order to drastically reduce the number of iterations needed to reach convergences.
    Can be configured to use dirichlet or neumann boundary conditions. This is a very slow version that uses nested
    loops. It needs to be optimised by e.g jit. 

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
    omega: float. The relaxation parameter. Its value is typically between 1<omega<2

    output
    u: (Nx+1)x(Ny+1) array. The solution of the problem.
    k: integer. The number of iteration that the solver performed.
    rel_diff: float. The ratio of the L2 norm of the difference of (k_final) and (k_final-1) iteration over the L2 norm of the                      
                     (k_final-1) iteration.
    conv_hist: array. An array containg all the values of rel_diff to check the convergence history.
    '''
    #read input
    M, f, u0, B, tol, kmax = matrix, source, init_guess, boundary, tolerance, itermax

    r, th = grid

    Nr, Nth = r.shape[1]-1, th.shape[0]-1

    dr, dth = (r[0, -1] - r[0, 0])/Nr, (th[-1, 0] - th[0, 0])/Nth

    if init_guess is None:

        u0 = np.ones_like(f)

    #assign dirichlet boundary conditions
    u0[:, 0] = B[0][0]                               
    u0[:, -1] = B[0][1]
    u0[0, :] = B[1][0]
    u0[-1, :] = B[1][1]
    
    #initial values before the iteration loop starts
    u = u0.copy()
    k = 0
    rel_diff = tol + 1
    conv_hist = []


    #iteration loop using nested loops.
    while  k < kmax and rel_diff > tol:

        u_next = u.copy()

        for j in range(1, Nth):
            for i in range(1, Nr):
                
                u_next[j, i] = (1-omega)*u[j,i] + omega*((u_next[j, i-1]*(r[0][i]-dr/2) + u[j, i+1]*(r[0][i]+dr/2))*r[0][i]*dth**2  
                                + (u_next[j-1, i] + u[j+1, i])*dr**2 - f[j,i]*dr**2*dth**2*r[0][i]**2)/(2*(dr**2+r[0][i]**2*dth**2))
        

        rel_diff = la.norm(u_next-u)/la.norm(u)
        
        conv_hist.append(rel_diff)
        
        u = u_next
        k += 1

    return u, k, rel_diff, conv_hist


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def grad_shaf_solver(matrix, source_term, grid, init_guess=None, 
                    boundary=((0, 0), (0,0)), tolerance=1.e-8, 
                    itermax=1000, omega=1.5, params=(1, 0, 0.5), 
                    u_analytical=None):

    #read the input
    M, u0, B, tol, kmax, uan = matrix, init_guess, boundary, tolerance, itermax, u_analytical

    #assign the grid
    r, th = grid

    R, TH = r[0], th[:,0]

    Nr, Nth = r.shape[1]-1, th.shape[0]-1

    dr, dth = (r[0, -1] - r[0, 0])/Nr, (th[-1, 0] - th[0, 0])/Nth

    
    #assign the initial guess
    if init_guess is None:

        u0 = np.ones_like(r)

    #assign dirichlet boundary conditions
    u0[1:-1, 0] = B[0][0][1:-1]                               
    # u0[:, -1] = B[0][1]
    u0[0, :] = B[1][0]
    u0[-1, :] = B[1][1]
    
    #assign extra parameters
    sigma, s, uc = params

    #assign the source term
    f = np.zeros_like(r)
    for j in range(1, Nth):
            for i in range(1, Nr):
                
                #simulate the effect of the heaviside step function
                if u0[j,i] >= uc:

                    f[j,i] = source_term(r[i], th[j], u0[j,i], params)#*np.heaviside(u0-uc, 0)
                else:
                    
                    f[j,i] = 0.


    #initial values before the iteration loop starts
    u = u0.copy()
    k = 0
    rel_diff = tol + 1
    conv_hist = []
    # err_to_an = []
    

    #iteration loop 
    while  k < kmax and rel_diff > tol:    
        
        # print the iteration number to keep track of the solver
        # if np.mod(k, 200) == 0:

        #     print(k)

        u_next = u.copy()

        
        
        #calculate the solution in the kth step
        for j in range(1, Nth):
            for i in range(1, Nr+1):
                
                #Update the source term if it is a function of the solution
                #simulate the effect of the heaviside step function
                if u[j,i] >= uc:

                    f[j,i] = source_term(r[i], th[j], u[j,i], params)
                else:
                    
                    f[j,i] = 0.

                #Robin boundary conditions at the outermost radius
                if i == Nr:

                    u_next[j,i] = (1-omega)*u[j,i] + omega/((2+2*dr/R[i])*R[i]**2*dth**2 + 2*dr**2)*(R[i]**2*dth**2*(2*u_next[j, i-1])                                                                                        + dr**2*(u[j+1,i]*(1-dth/(2*np.tan(TH[j]))) + 
                                                                                        u_next[j-1,i]*(1 + dth/(2*np.tan(TH[j])))) 
                                                                                        -f[j,i]*dr**2*dth**2*R[i]**2)
                    
                else:

                    u_next[j,i] = (1-omega)*u[j,i] + omega/(2*(R[i]**2*dth**2 + dr**2))*(R[i]**2*dth**2*(u[j,i+1] + u_next[j, i-1]) + 
                                                                                        dr**2*(u[j+1,i]*(1-dth/(2*np.tan(TH[j]))) + 
                                                                                        u_next[j-1,i]*(1 + dth/(2*np.tan(TH[j])))) 
                                                                                        -f[j,i]*dr**2*dth**2*R[i]**2)
                
        #calculate the L2 norm of the relative difference between the two last iterations   
        rel_diff = la.norm(u_next-u)/la.norm(u)
        
        #Save the convergence history
        conv_hist.append(rel_diff)
        
        #update solution for next iteration
        u = u_next

        k += 1

    return u, k, rel_diff, conv_hist


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def gs_comp_solver(matrix, source_term, grid, init_guess=None, 
                    boundary=((0, 0), (0,0)), tolerance=1.e-8, 
                    itermax=1000, omega=1.5, params=(1, 0, 0.5)):

    #read the input
    M, u0, B, tol, kmax = matrix, init_guess, boundary, tolerance, itermax

    #assign the grid
    q, th = grid

    Q, TH = q[0], th[:,0]

    Nq, Nth = q.shape[1]-1, th.shape[0]-1

    dq, dth = (q[0, -1] - q[0, 0])/Nq, (th[-1, 0] - th[0, 0])/Nth
    
    print(Nq, dq)
    
    #assign the initial guess
    if init_guess is None:

        u0 = np.ones_like(q)

    #assign dirichlet boundary conditions
    u0[:, 0] = B[0][0]                          #q_I boundary              
    u0[:, -1] = B[0][1]                         #q_F boundary
    u0[0, :] = B[1][0]                          #th_I boundary
    u0[-1, :] = B[1][1]                         #th_F boundary
    
    #assign extra parameters
    sigma, s, uc = params

    R = Q[-1]           #The radius of the star

    #assign the source term
    f = np.zeros_like(q)
    for j in range(1, Nth):
            for i in range(1, Nq):
                
                #simulate the effect of the heaviside step function
                if u0[j,i] >= uc:

                    f[j,i] = source_term(q[i], th[j], u0[j,i], params)
                else:
                    
                    f[j,i] = 0.


    #initial values before the iteration loop starts
    u = u0.copy()
    k = 0
    rel_diff = tol + 1
    conv_hist = []

    

    #iteration loop 
    while  k < kmax and rel_diff > tol:    
        
        # print the iteration number to keep track of the solver
        # if np.mod(k, 200) == 0:

        #     print(k)

        u_next = u.copy()
      
        #calculate the solution in the kth step
        for j in range(1, Nth):
            for i in range(1, Nq):
                
                #Update the source term if it is a function of the solution
                #simulate the effect of the heaviside step function
                if u[j,i] >= uc:

                    f[j,i] = source_term(q[i], th[j], u[j,i], params)
                else:
                    
                    f[j,i] = 0.


                #update the solution using SOR method
                u_next[j,i] = (1-omega)*u[j,i] + omega/(2*(Q[i]**2*dth**2 + dq**2))*  \
                              (dth**2*Q[i]*(u[j,i+1]*(Q[i]+dq) + u_next[j,i-1]*(Q[i]-dq)) 
                               +dq**2*(u[j+1,i]*(1-dth/(2*np.tan(TH[j]))) + u_next[j-1,i]*(1+dth/(2*np.tan(TH[j])))) 
                               -(dq*dth*R**2/Q[i])**2*f[j,i])
                
        #calculate the L2 norm of the relative difference between the two last iterations   
        rel_diff = la.norm(u_next-u)/la.norm(u)
        
        #Save the convergence history
        conv_hist.append(rel_diff)

        #update solution for next iteration
        u = u_next

        k += 1

    return u, k, rel_diff, conv_hist