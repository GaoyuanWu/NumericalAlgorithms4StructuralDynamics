###################################################### 
# This file helps you to apply boundary conditions,  # 
#  transfering initial M K into new matrices;      #               
# which can then be used in time integration         #                                    
#                                                    #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

'''
Specify boundary conditions at "Original Node".
1: Fixed
2: Pinned

Transform original node to "meshed node" index.
Reduce the size of global M & K.
Get relationship between global M & K and M_bc & K_bc

bc is a dictionary storing "original node" & "bc" (node number starts from 0)
ini_to_new represents relationship between initial node-index and new node-index 
'''

import numpy as np

def MK_Reduction(M,K,bc,ini_to_new):
    active_dof = list(np.arange(M.shape[0])) #Storing active dof index
    inact_dof = [] #Storing inactive dof index (BC)
    inact_num = 0 # Number of inactive dofs
    glb_to_bc = np.arange(M.shape[0]) #Storing relationship of global dof to new dof
            
    for i in range(len(bc)):
        temp_node_index = int(ini_to_new[list(bc.keys())[i]]) # BC node
        #Fixed, all dofs constrained
        if list(bc.values())[i] == 1:
            active_dof.remove(temp_node_index*3)
            active_dof.remove(temp_node_index*3 + 1)
            active_dof.remove(temp_node_index*3 + 2)
            inact_dof.append(temp_node_index*3) #x
            inact_dof.append(temp_node_index*3 + 1) #y
            inact_dof.append(temp_node_index*3 + 2) #Rotation
            inact_num += 3
            
        elif list(bc.values())[i] == 2:
            active_dof.remove(temp_node_index*3)
            active_dof.remove(temp_node_index*3 + 1)
            inact_dof.append(temp_node_index*3) #x
            inact_dof.append(temp_node_index*3 + 1)#y
            inact_num += 2

    for i in range(len(active_dof)):
        glb_to_bc[active_dof[i]] = i
    for i in range(len(inact_dof)):
        glb_to_bc[inact_dof[i]] = len(active_dof) + i    

    M_aa = M[active_dof,:]    
    M_aa = M_aa[:,active_dof]
    M_ai = M[active_dof,:]
    M_ai = M_ai[:,inact_dof]
    M_ia = M[inact_dof,:]
    M_ia = M_ia[:,active_dof]   
    M_ii = M[inact_dof,:]
    M_ii = M_ii[:,inact_dof]
    
    K_aa = K[active_dof,:]   
    K_aa = K_aa[:,active_dof]
    K_ai = K[active_dof,:]
    K_ai = K_ai[:,inact_dof]   
    K_ia = K[inact_dof,:]
    K_ia = K_ia[:,active_dof]   
    K_ii = K[inact_dof,:]
    K_ii = K_ii[:,inact_dof]
    
    M_bc1 = np.hstack((M_aa,M_ai))
    M_bc2 = np.hstack((M_ia,M_ii))
    M_bc = np.vstack((M_bc1,M_bc2))
    K_bc1 = np.hstack((K_aa,K_ai))
    K_bc2 = np.hstack((K_ia,K_ii))
    K_bc = np.vstack((K_bc1,K_bc2))
    
    return M_bc,K_bc,inact_num,glb_to_bc
    
    
            
    
    
    


