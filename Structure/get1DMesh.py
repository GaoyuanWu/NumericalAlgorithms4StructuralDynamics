###################################################### 
# This file gets structural geometry input from      #
# users and then discretizes the structural elements #               
# into smaller 1D meshes according to users'         #
# input of mesh size.                                # 
#                                                    #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np 

'''
# Get 1D mesh for the 2D plane frame
# node_coord -> initial node coordinates for the 2D frame; 2D array
# B -> Connectivity matrix representing the relationship between nodes & elements
# h -> mesh size; num = np.floor(length / h)
'''

def get1DMesh(Ini_nodecoord,B,h):
    Ini_nodeNum = Ini_nodecoord.shape[0] # Initial node numbers
    Ini_eleNum = B.shape[0] # Initial element numbers
    
    Ini_ele_u = np.zeros((Ini_eleNum))
    Ini_ele_v = np.zeros((Ini_eleNum))
    Ini_ele_l = np.zeros((Ini_eleNum)) # Array storing initial element parameters
    
    ele_subNum = np.zeros((Ini_eleNum)) # Storing sub-division numbers for initial ele
    
    New_eleNum = 0
    New_nodeNum = Ini_nodeNum
    
    for i in range(Ini_eleNum):
        Ini_ele_u[i] = Ini_nodecoord[B[i,1]-1,0] - Ini_nodecoord[B[i,0]-1,0] # Ini_ele X-difference
        Ini_ele_v[i] = Ini_nodecoord[B[i,1]-1,1] - Ini_nodecoord[B[i,0]-1,1] # Ini_ele Y-difference
        Ini_ele_l[i] = np.sqrt(Ini_ele_u[i]**2 + Ini_ele_v[i]**2) # Ini_ele length
        
        ele_subNum[i] = np.floor(Ini_ele_l[i]/h)
        New_eleNum += int(ele_subNum[i])
        New_nodeNum += int(ele_subNum[i] - 1)
        
    New_nodecoord = np.zeros((New_nodeNum,2)) #Storing new node coordinates
    New_B = np.zeros((New_eleNum,2)).astype(int) # Storing new connectivity matrix
    
    New_node_index = 0
    New_ele_index = 0
    Ini_node_index = [] # List storing old node already in new node list
    Ini_to_New = np.zeros((Ini_nodeNum)) # Representing relationship between initial node-index and new node-index
    
    # Creating new B and new Node_coord
    for i in range(Ini_eleNum):
        temp_node_index_new = np.zeros(int(ele_subNum[i]) + 1)
        
        # Assignment for new_node_coord & new_B
        for j in range(int(ele_subNum[i]) + 1):
            if j == 0: # Check whether local node1 in list already or not
                if B[i,0] not in Ini_node_index:
                    Ini_node_index.append(B[i,0])
                    (New_nodecoord[New_node_index])[0] = Ini_nodecoord[B[i,0]-1,0] # x, local node1
                    (New_nodecoord[New_node_index])[1] = Ini_nodecoord[B[i,0]-1,1] # y, local node1
                    Ini_to_New[B[i,0]-1] = New_node_index
                    temp_node_index_new[j] = New_node_index
                    New_node_index += 1
                else:
                    temp_node_index_new[j] = Ini_to_New[B[i,0]-1]
            elif j == ele_subNum[i]: # Check whether local node2 in list already or not
                if B[i,1] not in Ini_node_index:
                    Ini_node_index.append(B[i,1])
                    (New_nodecoord[New_node_index])[0] = Ini_nodecoord[B[i,1]-1,0] # x, local node2
                    (New_nodecoord[New_node_index])[1] = Ini_nodecoord[B[i,1]-1,1] # y, local node2
                    Ini_to_New[B[i,1]-1] = New_node_index
                    temp_node_index_new[j] = New_node_index
                    New_B[New_ele_index,0] = temp_node_index_new[j-1]
                    New_B[New_ele_index,1] = temp_node_index_new[j] # New_B
                    New_ele_index += 1
                    New_node_index += 1
                else:
                    temp_node_index_new[j] = Ini_to_New[B[i,1]-1]
                    New_B[New_ele_index,0] = temp_node_index_new[j-1]
                    New_B[New_ele_index,1] = temp_node_index_new[j]
                    New_ele_index += 1
            else:
                (New_nodecoord[New_node_index])[0] = Ini_nodecoord[B[i,0]-1,0] + Ini_ele_u[i]/ele_subNum[i] * j  # x
                (New_nodecoord[New_node_index])[1] = Ini_nodecoord[B[i,0]-1,1] + Ini_ele_v[i]/ele_subNum[i] * j  # y
                temp_node_index_new[j] = New_node_index
                New_B[New_ele_index,0] = temp_node_index_new[j-1]
                New_B[New_ele_index,1] = temp_node_index_new[j]
                New_ele_index += 1
                New_node_index += 1
    return New_nodecoord,New_B + 1, Ini_to_New #+1: Back to physical index..starting from 1
       
            

   
     
            
        
        
        
        
        
        
        
        





	

 






