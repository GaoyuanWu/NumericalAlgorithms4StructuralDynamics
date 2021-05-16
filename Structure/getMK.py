###################################################### 
# This file helps you to get stiffness/mass          # 
#  matrices for a 2-D shear building                 #               
# (plane frames)                                     #
# Then matrices can then be used for dynamic analysis# 
# of structures.                                     #
#                                                    #
# Created by Gaoyuan WU, Phd Student @ Princeton     # 
######################################################

import numpy as np 

'''
# Method 1, lumped-mass method
# Features: 
# * simple & easy to use;
# * enjoys popularity in engineering practice;
# * fair results but not that accurate;
# * need to simplyfy the structures by experienced engineers beforehand

#Inputs:
# n -> Number of stories
# {k} -> Vector for stiffness of each floor; 1-D array; top floor to bottom floor
# {m} -> Vector for lumped-mass of each floor; 1-D array
'''
def getMK_SDOF(k,m):
    return np.array([k]),np.array([m])

def getMK_lumpedmass(n,k,m):
    M = np.diag(m)# Mass Matrix
    K = np.zeros([n,n])   # Initialized Stiffness Matrix
    for i in range(n):
        for j in range(n):
            if j == 0:
                if i == 0:
                    K[i,j] = k[0]
                elif i == 1:
                    K[i,j] = -k[0]
            if j == n-1:
                if i == n-1:
                    K[i,j] = k[n-1] + k[n-2]
                elif i == n-2:
                    K[i,j] = -k[n-2]
            elif i == n-1 and j == n-2:
                K[i,j] = K[j,i]
            else:
                if i == j:
                    K[i,j] = k[j-1] + k[j-2]
                elif i == j+1:
                    K[i,j] = -k[j-1]
                elif i == j-1:
                    K[i,j] = -k[j-2]
    
    return M,K # End of Lumped Mass


'''
# Method 2, Finite-element method (FEM)
# Features: 
# * more accurate;
# * able to represent axial deformation & bending deformation of each element;
# * utilizes "shape function", which can be viewed as "Interpolation" within the element;
# * results are senstive to number of elemens

#Inputs:
# joint_coord -> Global coordinates of joints; 2-D array; nj(number of joints) x 2
# B -> Connectivity matrix between joints & elements; 2-D array; ne(number of elements) x 2
# ele_m -> mass per unit length of each element;1-D array
# ele_E -> Young's modulus; 1-D array
# ele_A -> Area of the element; 1-D array
# ele_I -> moment of inertia of the elemennt; 1-D array
'''

#Local mass matrix
def local_M(m,l):
    row_1 = np.array([140,0,0,70,0,0])
    row_2 = np.array([0,156,22*l,0,54,-13*l])
    row_3 = np.array([0,22*l,4*l**2,0,13*l,-3*l**2])
    row_4 = np.array([70,0,0,140,0,0])
    row_5 = np.array([0,54,13*l,0,156,-22*l])
    row_6 = np.array([0,-13*l,-3*l**2,0,-22*l,4*l**2])
    local_M = np.vstack([row_1,row_2,row_3,row_4,row_5,row_6])
    return m*l/420*local_M

#Local stifness matrix
def local_K(E,I,A,l):
    row_1 = np.array([A*l**2/I,0,0,-A*l**2/I,0,0])
    row_2 = np.array([0,12,6*l,0,-12,6*l])
    row_3 = np.array([0,6*l,4*l**2,0,-6*l,2*l**2])
    row_4 = np.array([-A*l**2/I,0,0,A*l**2/I,0,0])
    row_5 = np.array([0,-12,-6*l,0,12,-6*l])
    row_6 = np.array([0,6*l,2*l**2,0,-6*l,4*l**2])
    local_K = np.vstack([row_1,row_2,row_3,row_4,row_5,row_6])
    return E*I*local_K/(l**3)

#Coordinate transformation matrix
def trans_T(beta):
    T = np.zeros([6,6])
    sine = np.sin(beta)
    cosine = np.cos(beta)
    T[0,0] = cosine
    T[0,1] = sine
    T[1,0] = -sine
    T[1,1] = cosine
    T[2,2] = 1
    T[3,3] = cosine
    T[3,4] = sine
    T[4,3] = -sine
    T[4,4] = cosine
    T[5,5] = 1
    return T


def getMK_FEM(joint_coord,B,ele_m,ele_E,ele_A,ele_I):
    jo_num = joint_coord.shape[0] #Number of joints
    ele_num = B.shape[0] #Number of elements    
    
    Glo_K = np.zeros((jo_num * 3,jo_num * 3)) #2-D array storing "Global Stiffness Matrices" 
    Glo_M = np.zeros((jo_num * 3,jo_num * 3)) #2-D array storing "Global Mass Matrices"
    ele_K = np.zeros((ele_num,6,6)) #3-D array storing "Element Stiffness Matrices"
    ele_M = np.zeros((ele_num,6,6)) #3-D array storing "Element Mass Matrices"
    
    ele_u = np.zeros((ele_num))
    ele_v = np.zeros((ele_num))
    ele_l = np.zeros((ele_num))
    ele_uni = np.zeros((ele_num,2)) # Array storing element unit vector

    Global_x = np.array([1,0]) # Global unit_X
    ele_beta = np.zeros(ele_num) #Initialized angle between local-1 to global-x

    
    for i in range(ele_num):
        ele_u[i] = joint_coord[B[i,1]-1,0] - joint_coord[B[i,0]-1,0] # ele X-difference
        ele_v[i] = joint_coord[B[i,1]-1,1] - joint_coord[B[i,0]-1,1] # ele Y-difference
        ele_l[i] = np.sqrt(ele_u[i]**2 + ele_v[i]**2) # ele length
        ele_uni[i,:] = (np.vstack([ele_u[i],ele_v[i]]).T) / ele_l[i] # ele unit vector
        if ele_v[i] >= 0:
            ele_beta[i] = np.arccos(np.dot(Global_x,ele_uni[i]))
        else:
            ele_beta[i] = 2 * np.pi - np.arccos(np.dot(Global_x,ele_uni[i]))

        ele_K[i,:,:] = local_K(ele_E[i],ele_I[i],ele_A[i],ele_l[i])
        ele_M[i,:,:] = local_M(ele_m[i],ele_l[i])
        trans_temp = trans_T(ele_beta[i]) #Coordinate Transform
        ele_K[i,:,:] = (trans_temp.T @ ele_K[i,:,:]) @ trans_temp # To Global K for ele
        ele_M[i,:,:] = (trans_temp.T @ ele_M[i,:,:]) @ trans_temp # To Global M for ele

        #Put ele-matrix into global matrix
        nodei_temp = B[i,0] - 1
        nodej_temp = B[i,1] - 1

        Glo_K[3*nodei_temp:3*nodei_temp+3, 3*nodei_temp:3*nodei_temp+3] += ele_K[i,:3,:3]
        Glo_K[3*nodej_temp:3*nodej_temp+3, 3*nodei_temp:3*nodei_temp+3] += ele_K[i,3:,:3]
        Glo_K[3*nodej_temp:3*nodej_temp+3, 3*nodej_temp:3*nodej_temp+3] += ele_K[i,3:,3:]
        Glo_K[3*nodei_temp:3*nodei_temp+3, 3*nodej_temp:3*nodej_temp+3] += ele_K[i,:3,3:]

        Glo_M[3*nodei_temp:3*nodei_temp+3, 3*nodei_temp:3*nodei_temp+3] += ele_M[i,:3,:3]
        Glo_M[3*nodej_temp:3*nodej_temp+3, 3*nodei_temp:3*nodei_temp+3] += ele_M[i,3:,:3]
        Glo_M[3*nodej_temp:3*nodej_temp+3, 3*nodej_temp:3*nodej_temp+3] += ele_M[i,3:,3:]
        Glo_M[3*nodei_temp:3*nodei_temp+3, 3*nodej_temp:3*nodej_temp+3] += ele_M[i,:3,3:]
    
    return Glo_K,Glo_M




    

 






