###################################################### 
# This file includes Wilson-theta algorithms for     #                                    
# structural dynamics                                #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np 
from scipy.linalg import ldl
from scipy.linalg import solve_triangular

'''
Wilson-theta Method for Structural Dynamics.
This algorithm is an extension of the liear-acceleration method.
No need to convert 2nd order ODE to two 1st order ODE.

No need of inversing matrix.
Utilizes Cholesky decomposition (LDL) or LU decomposition.

Theta>=1

Scheme for Wilson theta:  
    (1) Assume linear change for acceleration:
        * a_(t+tau) = a_t + tau/(theta*dt) * (a_(t+theta*dt) - a_t)
    (2) Integrate (with respect to tau) to get the change for velocity:
        * v_(t+tau) = v_t + tau*a_t + tau**2/(2*theta*dt)*(a_(t+theta*dt)-a_t)
    (3) Integrate (2) to get the expression for displacement:
        * u_(t+tau) = u_t + tau*v_t + tau**2/2 * a_t + tau**3/(6*theta*dt)*(a_(t+theta*dt)-a_t)
    (4) plug tau = theta * dt in:
        Get expression of u_(t+theta*dt) & v_(t+theta*dt)
    (5) write acceleration & velocity as a function of displacement
    (6) state EOM at t+theta*dt
    (7) Become a function for u_(t+theta*dt), solve it. (In a form of Ax=b)
    (8) Get acceleration & velocity at t+theta*dt
    (9) Back to t+dt
    
#Inputs:
# [K_a] -> Stiffness matrix for active dofs
# [M_a] -> Mass matrix for active dofs
# [C_a] -> Propotional damping matrix for active dofs
# f_t -> External loads acting on active dofs. 
Each element is a time-dependent function or values at discrete time steps (earthquake).
1. {f_t} is a vector of functions f_t(t)
2. [f_t] is a matrix, records of ground motion
# {u_0} -> Initial value for displacement
# {v_0} -> Initial value for velocity
# h -> Time step for integration
# t_a,N start time of time integration,steps
# dt_ft time step within discrete ft
# theta: coefficient in the method
'''
def Wilson_theta(theta,K_a,M_a,C_a,f_t,u_0,v_0,t_a,N,h,dt_ft = 0):
     
    n_dof = K_a.shape[0] # Dimension
    
    #Storing array
    u_t = np.zeros((n_dof,N+1))
    v_t = np.zeros((n_dof,N+1))
    a_t = np.zeros((n_dof,N+1))
    
    #Initial value
    u_t[:,0] = u_0
    v_t[:,0] = v_0
    
    if dt_ft == 0:
        b = f_t(0) - C_a@v_0 - K_a@u_0
    else:
        b = f_t[:,0] - C_a@v_0 - K_a@u_0

    a_t[:,0] = np.linalg.solve(M_a,b)
    
    
    #Start of Wilson-theta
    
    #Parameters for effective stiffness & load
    a0 = 6/(theta*h)**2
    a1 = 3/(theta*h)
    a2 = 2 *a1
    a3 = theta*h/2
    a4 = a0/theta
    a5 = -a2/theta
    a6 = 1-3/theta
    a7 = h/2
    a8 = h**2/6
    
    #Effective stiffness matrix and decomposition
    K_ef = K_a + a0 * M_a + a1 * C_a
    L,D,perm = ldl(K_ef,lower = 0) #LDL decomposition
    
    #Case1: Inputs are sets of functions      
    if dt_ft == 0:
        for i in range(N):
            t_i = i * h
            F_ef_temp = f_t(t_i) + theta * (f_t(t_i + h) - f_t(t_i)) + M_a @ (a0 * u_t[:,i] + a2 * v_t[:,i] + 2 * a_t[:,i]) + C_a @ (a1 * u_t[:,i] + 2 * v_t[:,i] + a3 * a_t[:,i])

            u_temp = np.linalg.solve(K_ef,F_ef_temp)
            
            #Get displacement, velocity and acceleration at t + h
            a_t[:,i+1] = a4 * (u_temp - u_t[:,i]) + a5 * v_t[:,i] + a6 * a_t[:,i]
            v_t[:,i+1] = v_t[:,i] + a7 * (a_t[:,i+1] + a_t[:,i])
            u_t[:,i+1] = u_t[:,i] + h * v_t[:,i] + a8 * (a_t[:,i+1] + 2*a_t[:,i])
    
    #Case2: Inputs are discrete datasets       
    else:
        for i in range(N):
            t_i = i * h
            f_left_index_i = int(t_i/dt_ft)
            f_left_dis_i = (t_i)%dt_ft
            ft_temp = f_t[:,f_left_index_i] + (f_t[:,f_left_index_i + 1]-f_t[:,f_left_index_i]) * f_left_dis_i/dt_ft
            
            f_left_index_ip1 = int((t_i+h)/dt_ft)
            f_left_dis_ip1 = (t_i+h)%dt_ft
            ftp1_temp = f_t[:,f_left_index_ip1] + (f_t[:,f_left_index_ip1 + 1]-f_t[:,f_left_index_ip1]) * f_left_dis_ip1/dt_ft
            
            F_ef_temp = ft_temp + theta * (ftp1_temp - ft_temp) + M_a @ (a0 * u_t[:,i] + a2 * v_t[:,i] + 2 * a_t[:,i]) + C_a @ (a1 * u_t[:,i] + 2 * v_t[:,i] + a3 * a_t[:,i])
            
            #Solve for u_(t+theta*h)
            u_temp = np.linalg.solve(K_ef,F_ef_temp)
            
            #Get displacement, velocity and acceleration at t + h
            a_t[:,i+1] = a4 * (u_temp - u_t[:,i]) + a5 * v_t[:,i] + a6 * a_t[:,i]
            v_t[:,i+1] = v_t[:,i] + a7 * (a_t[:,i+1] + a_t[:,i])
            u_t[:,i+1] = u_t[:,i] + h * v_t[:,i] + a8 * (a_t[:,i+1] + 2*a_t[:,i])
    return u_t,v_t,a_t
    
    

    




    

 






