###################################################### 
# This file includes Newmark-beta algorithms for     #                                    
# structural dynamics                                #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np 

'''
Newmark-beta Method for Structural Dynamics.
This algorithm is an extension of the liear-acceleration method.
No need to convert 2nd order ODE to two 1st order ODE.
No need of inversing matrix.

delta,beta are parameters
    
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
# delta, beta : coefficients

#Recommened: delta >= 0.5, beta >= (0.5 + delta)**2
'''
def Newmark_beta(delta,beta,K_a,M_a,C_a,f_t,u_0,v_0,t_a,N,h,dt_ft = 0):
    
    n_dof = K_a.shape[0] # Dimension
    
    #Storing array
    u_t = np.zeros((n_dof,N+1),float)
    v_t = np.zeros((n_dof,N+1),float)
    a_t = np.zeros((n_dof,N+1),float)
    
    #Initial value
    u_t[:,0] = u_0
    v_t[:,0] = v_0
    
    #Start of Newmark-beta
    
    #Coefficients for effective stiffness & load
    a0 = 1/(beta*(h**2))
    a1 = delta/(beta*h)
    a2 = 1/(beta*h)
    a3 = (1/(2*beta)) - 1
    a4 = (delta/beta) - 1
    a5 = (h/2) * ((delta/beta) -2)
    a6 = h * (1 - delta)
    a7 = delta * h
    
    #Effective stiffness matrix and decomposition
    K_ef = K_a + a0 * M_a + a1 * C_a
    #L,D,perm = ldl(K_ef,lower = 0) #LDL decomposition
    
    #Case1: Inputs are sets of functions      
    if dt_ft == 0:
        for i in range(N):
            t_i = i * h
            temp_1 = a0 * u_t[:,i] + a2 * v_t[:,i] + a3 * a_t[:,i]
            temp_M = M_a @ temp_1
            temp_2 = a1 * u_t[:,i] + a4 * v_t[:,i] + a5 * a_t[:,i]
            temp_C = C_a @ temp_2
            F_ef_temp = f_t(t_i + h) + temp_M + temp_C
            
            #Solve for u_(t+theta*h)
            u_t[:,i+1] = np.linalg.solve(K_ef,F_ef_temp)
            
            #Get velocity and acceleration at t + h
            a_t[:,i+1] = a0 * (u_t[:,i+1] - u_t[:,i]) - a2 * v_t[:,i] - a3 * a_t[:,i]
            v_t[:,i+1] = v_t[:,i] + a6 * a_t[:,i] + a7 * a_t[:,i+1]
    
    #Case2: Inputs are discrete datasets       
    else:
        for i in range(N):
            t_i = i * h
            f_left_index = int((t_i+h)/dt_ft)
            f_left_dis = (t_i+h)%dt_ft
            f_tpdt = f_t[:,f_left_index] + (f_t[:,f_left_index + 1]-f_t[:,f_left_index]) * f_left_dis/dt_ft
            temp_1 = a0 * u_t[:,i] + a2 * v_t[:,i] + a3 * a_t[:,i]
            temp_M = M_a @ temp_1
            temp_2 = a1 * u_t[:,i] + a4 * v_t[:,i] + a5 * a_t[:,i]
            temp_C = C_a @ temp_2
            F_ef_temp = f_tpdt + temp_M + temp_C
            
            #Solve for u_(t+theta*h)
            u_t[:,i+1] = np.linalg.solve(K_ef,F_ef_temp)
            
            #Get velocity and acceleration at t + h
            a_t[:,i+1] = a0 * (u_t[:,i+1] - u_t[:,i]) - a2 * v_t[:,i] - a3 * a_t[:,i]
            v_t[:,i+1] = v_t[:,i] + a6 * a_t[:,i] + a7 * a_t[:,i+1]

    return u_t,v_t,a_t
    
    

    




    

 






