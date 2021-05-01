###################################################### 
# This file helps you to apply boundary conditions,  # 
#  transfering initial M & K into new matrices;      #               
# which can then be used in time integration         #                                    
#                                                    #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np 

'''
Runge-Kutta for Structural Dynamics.
Utilizes Strong-stability preserving Runge-Kunte with 3 stages (SSP RK3)

Scheme for RK3:
    * dU/dt = F(U,t)
    
    (1) U_1 = U_n + h(F(U_n,t_n))
    (2) U_2 = 3/4*U_n + 1/4(F(U_1,t_(n+1))
    (3) U_(n+1) = 1/3 U_n + 2/3(F(U_2,t_(n+1/2)))

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
'''
def SSP_RK3(K_a,M_a,C_a,f_t,u_0,v_0,t_a,N,h,dt_ft = 0):
    
    if dt_ft != 0:
        if(h%dt_ft != 0):
            print("Please choose anothe time step h that is in accordance with time step in ground motion records")
        
    
    U_0 = np.hstack((u_0.T,v_0.T)) #Initial Value
    M_inv = np.linalg.inv(M_a) #Inverse of M_a
    n_dof = K_a.shape[0] # Dimension
    
    
    
    A_r1c1 = np.zeros((n_dof,n_dof))
    A_r1c2 = np.identity(n_dof)
    A_r2c1 = -M_inv@K_a
    A_r2c2 = -M_inv@C_a
    A_r1 = np.hstack((A_r1c1,A_r1c2))
    A_r2 = np.hstack((A_r2c1,A_r2c2))
    A = np.vstack((A_r1,A_r2)) #The big matrix
        
    # Start SSP-RK3
    U_t = np.zeros((2*n_dof,N+1))
    U_t[:,0] = U_0
    
    #Case1: Inputs are sets of functions
    if dt_ft == 0:
        for i in range(N):
            t_i = i * h
            F_t_i = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t(t_i)).T))
            F_t_ip1 = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t(t_i + h)).T))
            F_t_iph = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t(t_i + 0.5*h)).T))        
            U_i = U_t[:,i]
            #Stage 1
            U1 = U_i + h * (A@U_i + F_t_i)
            #Stage 2
            U2 = 3/4 * U_i + 1/4 * (U1 + h * (A@U1 + F_t_ip1))
            #Final Stage
            U_t[:,i+1] = 1/3 * U_i + 2/3 * (U2 + h * (A@U2 + F_t_iph))
    else:
        for i in range(N):
            t_i = i * h
            F_t_i = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t[:,int(t_i/dt_ft)]).T))
            F_t_ip1 = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t[:,int(t_i/dt_ft) + 1]).T))
            
            #t_i + 0.5 h, just take the mid point of the record
            f_t_iph = 0.5 * (f_t[:,int(t_i/dt_ft)] + f_t[:,int(t_i/dt_ft) + 1])
            F_t_iph = np.hstack((np.zeros((n_dof)).T,(M_inv@f_t_iph).T))  
            
            U_i = U_t[:,i]
            #Stage 1
            U1 = U_i + h * (A@U_i + F_t_i)
            #Stage 2
            U2 = 3/4 * U_i + 1/4 * (U1 + h * (A@U1 + F_t_ip1))
            #Final Stage
            U_t[:,i+1] = 1/3 * U_i + 2/3 * (U2 + h * (A@U2 + F_t_iph))
        
    return U_t
    
    

    




    

 






