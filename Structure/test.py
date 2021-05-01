import sys
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/Structure')
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/TimeIntegration')
import get1DMesh as get1DMesh
import getPlot as getPlot
import getMK as getMK
import getBC as getBC
import numpy as np
import SSPRK3 as SSPRK3
import matplotlib.pyplot as plt
import pandas as pd

#Initial structure
test_nodecoord = np.array([[0,0],[100*np.cos(np.pi/4),100*np.cos(np.pi/4)],[100*np.cos(np.pi/4)+100,100*np.cos(np.pi/4)]])
test_B = np.array ([[1,2],[2,3]])

#Fixed BC
bc = {0:1,2:1} 

#Plot
getPlot.getPlot(test_nodecoord,test_B,True,True,10)

# 1D MESH
h = 100 #mesh size
new_nodecoord,new_B,itn = get1DMesh.get1DMesh(test_nodecoord, test_B, h)
getPlot.getPlot(new_nodecoord,new_B,True,True,10)     
        
# Global Matrix
ele_A = 6 * np.ones(new_B.shape[0])
ele_m = 4.2 * np.ones(new_B.shape[0])
ele_I = 100 * np.ones(new_B.shape[0])
ele_E = 10**7 * np.ones(new_B.shape[0])

K,M = getMK.getMK_FEM(new_nodecoord,new_B, ele_m, ele_E, ele_A, ele_I)
#print("stiffness matrix: ", K)
#print("mass matrix: ", M)     
#print("old to new ", itn)  


#Transformed Matrix
M_bc,K_bc,inact_num,glb_to_bc = getBC.MK_Reduction(M, K, bc, itn)
act_num = K_bc.shape[0] - inact_num
K_a = K_bc[:act_num,:act_num]
M_a = M_bc[:act_num,:act_num]
C_a = C_bc[:act_num,:act_num]
u_0 = np.zeros(act_num)
v_0 = np.zeros(act_num)    


#Propotional Damping
alpha = 0
beta = 0
C = alpha * M + beta * K
C_bc = alpha * M_bc + beta * K_bc


#External loading
#Test External loading 1: acting on ini_node 2, dir x
def f_cons(x):
    return 100000
def test_f1(x):
    f_list = np.array([[f_cons,glb_to_bc[itn[1]*3]]])
    vec = np.zeros(act_num)
    for i in range(f_list.shape[0]):
        pos = f_list[i,1]
        vec[pos] = f_list[i,0](x)
    return vec    

#Test External loading 2: Bi-sections. 100000 from t = 0
#to t = 0.25; drops linearly to 0 till t = 0.5s
def f_bi(x):
    if x <= 0.25:
        return 100000
    elif x<= 0.5:
        return 100000-400000*(x-0.25)
    else:
        return 0
def test_f2(x):
    f_list = np.array([[f_bi,glb_to_bc[itn[1]*3]]])
    vec = np.zeros(act_num)
    for i in range(f_list.shape[0]):
        pos = f_list[i,1]
        vec[pos] = f_list[i,0](x)
    return vec

#Test External loading 3: Ground Motion-Kern County(1952), LA - Hollywood Stor FF
#PGA = 0.1g
#dt = 0.005
#Read xlsx

GM_tab = pd.read_excel('GroundMotion.xlsx')
GM_data = np.array(GM_tab.values)
dt_ft = 0.005
test_f3 = - M_a @ (np.ones((act_num,1))@(GM_data.T))




# Algorithm 1: SSP-RK3 #

t_a = 0
h = 0.005
N = 10000

#Test 1:
U_test1 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f1,u_0,v_0,t_a,N,h)
u1 = U_test1[0,:]
u2 = U_test1[1,:]
u3 = U_test1[2,:]

#fig, ax = plt.subplots()
#plt.ylabel("u1")
#plt.plot(np.arange(0,5.005,0.005),u1,color = 'b')

#fig, ax = plt.subplots()
#plt.ylabel("u2")
#plt.plot(np.arange(0,5.005,0.005),u2,color = 'b')

#fig, ax = plt.subplots()
#plt.ylabel("u3")
#plt.plot(np.arange(0,5.005,0.005),u3,color = 'b')

#Test 2:
U_test2 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f2,u_0,v_0,t_a,N,h)
u1 = U_test2[0,:]
u2 = U_test2[1,:]
u3 = U_test2[2,:]

#fig, ax = plt.subplots()
#plt.ylabel("u1")
#plt.plot(np.arange(0,5.005,0.005),u1,color = 'b')

#fig, ax = plt.subplots()
#plt.ylabel("u2")
#plt.plot(np.arange(0,5.005,0.005),u2,color = 'b')

#fig, ax = plt.subplots()
#plt.ylabel("u3")
#plt.plot(np.arange(0,5.005,0.005),u3,color = 'b')

#Test 3:
U_test3 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f3,u_0,v_0,t_a,N,h,dt_ft)
u1 = U_test3[0,:]
u2 = U_test3[1,:]
u3 = U_test3[2,:]

fig, ax = plt.subplots()
plt.ylabel("u1")
plt.plot(np.arange(0,50.005,0.005),u1,color = 'b')

fig, ax = plt.subplots()
plt.ylabel("u2")
plt.plot(np.arange(0,50.005,0.005),u2,color = 'b')

fig, ax = plt.subplots()
plt.ylabel("u3")
plt.plot(np.arange(0,50.005,0.005),u3,color = 'b')


