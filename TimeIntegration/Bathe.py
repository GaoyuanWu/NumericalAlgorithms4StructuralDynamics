###################################################### 
# This file includes Wilson-theta algorithms for     #                                    
# structural dynamics                                #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np

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
def Bathe(K_a,M_a,C_a,f_t,u_0,v_0,t_a,N,h,dt_ft = 0):
     
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
    
    
    #Start of Bathe
    
    #Parameters for effective stiffness & load
    a1 = 16/(h**2)
    a2 = 4/h
    a3 = 9/(h**2)
    a4 = 3/h
    a5 = 8/h
    a6 = 4/h
    a7 = 12/(h**2)
    a8 = 3/(h**2)
    a9 = 1/h
    
    
    #Case1: Inputs are sets of functions      
    if dt_ft == 0:
        for i in range(N):
            t_i = i * h
            
            # At t_i + 0.5h
            K_eff_1 = a1 * M_a + a2 * C_a + K_a
            F_eff_1 = f_t(t_i + 0.5 *h) + (a1*u_t[:,i] + a5*v_t[:,i] + a_t[:,i])@M_a + (a6*u_t[:,i] + v_t[:,i]) @ C_a
            u_half = np.linalg.solve(K_eff_1,F_eff_1)
            v_half = (u_half - u_t[:,i]) * a6 - v_t[:,i]
            
            # At t_i + h
            K_eff_2 = a3 * M_a + a4 * C_a + K_a
            F_eff_2 = f_t(t_i + h) + (a7 * u_half - a8*u_t[:,i] + a6*v_half-a9*v_t[:,i])@M_a + C_a@(a6*u_half - a9*u_t[:,i])
            
            u_temp = np.linalg.solve(K_eff_2,F_eff_2)
            #Get displacement, velocity and acceleration at t + h
            v_t[:,i+1] = a9 * u_t[:,i] - a2 * u_half + a4*u_temp
            a_t[:,i+1] = a9 * v_t[:,i] - a2 * v_half + a4*v_t[:,i+1]
            u_t[:,i+1] = u_temp
    
    #Case2: Inputs are discrete datasets       
    else:
        for i in range(N):
            t_i = i * h
            f_left_index = int((t_i+h)/dt_ft)
            f_left_dis = (t_i+h)%dt_ft
            f_tpdt = f_t[:,f_left_index] + (f_t[:,f_left_index + 1]-f_t[:,f_left_index]) * f_left_dis/dt_ft
            
            f_left_half_index = int((t_i + 0.5*h)/dt_ft)
            f_left_half_dis = (t_i + 0.5 * h)%dt_ft
            f_tphalf = f_t[:,f_left_half_index] + (f_t[:,f_left_half_index + 1]-f_t[:,f_left_half_index]) * f_left_half_dis/dt_ft
            # At t_i + 0.5h
            K_eff_1 = a1 * M_a + a2 * C_a + K_a
            F_eff_1 = f_tphalf + (a1*u_t[:,i] + a5*v_t[:,i] + a_t[:,i])@M_a + (a6*u_t[:,i] + v_t[:,i]) @ C_a
            u_half = np.linalg.solve(K_eff_1,F_eff_1)
            v_half = (u_half - u_t[:,i]) * a6 - v_t[:,i]
            
            # At t_i + h
            K_eff_2 = a3 * M_a + a4 * C_a + K_a
            F_eff_2 = f_tpdt + (a7 * u_half - a8*u_t[:,i] + a6*v_half-a9*v_t[:,i])@M_a + C_a@(a6*u_half - a9*u_t[:,i])
            
            u_temp = np.linalg.solve(K_eff_2,F_eff_2)
            #Get displacement, velocity and acceleration at t + h
            v_t[:,i+1] = a9 * u_t[:,i] - a2 * u_half + a4*u_temp
            a_t[:,i+1] = a9 * v_t[:,i] - a2 * v_half + a4*v_t[:,i+1]
            u_t[:,i+1] = u_temp

    return u_t,v_t,a_t
    
    

    




    

 






