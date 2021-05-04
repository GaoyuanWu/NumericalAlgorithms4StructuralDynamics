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
import Wilson as Wilson
import Newmark as Newmark

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

#Propotional Damping
alpha = 0
beta = 0.0001
C = alpha * M + beta * K


#Transformed Matrix
M_bc,K_bc,inact_num,glb_to_bc = getBC.MK_Reduction(M, K, bc, itn)
C_bc = alpha * M_bc + beta * K_bc
act_num = K_bc.shape[0] - inact_num
K_a = K_bc[:act_num,:act_num]
M_a = M_bc[:act_num,:act_num]
C_a = C_bc[:act_num,:act_num]
u_0 = np.zeros(act_num)
v_0 = np.zeros(act_num)    


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

GM_tab = pd.read_excel('GM.xlsx')
GM_data = np.array(GM_tab.values)
dt_ft = 0.005
test_f3 = - M_a @ (np.ones((act_num,1))@(GM_data.T))




# Algorithm 1: SSP-RK3 #

t_a = 0
h = 0.005
N = 1000

#Test 1:
U_test1 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f1,u_0,v_0,t_a,N,h)
u1_ssp1 = U_test1[0,:]
u2_ssp1 = U_test1[1,:]
u3_ssp1 = U_test1[2,:]

#Test 2:
U_test2 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f2,u_0,v_0,t_a,N,h)
u1_ssp2 = U_test2[0,:]
u2_ssp2 = U_test2[1,:]
u3_ssp2 = U_test2[2,:]

#Test 3:
U_test3 = SSPRK3.SSP_RK3(K_a,M_a,C_a,test_f3,u_0,v_0,t_a,N,h,dt_ft)
u1_ssp3 = U_test3[0,:]
u2_ssp3 = U_test3[1,:]
u3_ssp3 = U_test3[2,:]

# Algorithm 2: Wilson-theta #
theta = 1.38
#Test 1:
u,v,a = Wilson.Wilson_theta(theta,K_a,M_a,C_a,test_f1,u_0,v_0,t_a,N,h)
u1_w1 = u[0,:]
u2_w1 = u[1,:]
u3_w1 = u[2,:]

#Test 2:
u,v,a = Wilson.Wilson_theta(theta,K_a,M_a,C_a,test_f2,u_0,v_0,t_a,N,h)
u1_w2 = u[0,:]
u2_w2 = u[1,:]
u3_w2 = u[2,:]

#Test 3:
u,v,a = Wilson.Wilson_theta(theta,K_a,M_a,C_a,test_f3,u_0,v_0,t_a,N,h,dt_ft)
u1_w3 = u[0,:]
u2_w3 = u[1,:]
u3_w3 = u[2,:]

# Algorithm 3: Newmark-beta #
delta = 0.5
beta = 0.25
#Test 1:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f1,u_0,v_0,t_a,N,h)
u1_nm1 = u[0,:]
u2_nm1 = u[1,:]
u3_nm1 = u[2,:]

#Test 2:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f2,u_0,v_0,t_a,N,h)
u1_nm2 = u[0,:]
u2_nm2 = u[1,:]
u3_nm2 = u[2,:]

#Test 3:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f3,u_0,v_0,t_a,N,h,dt_ft)
u1_nm3 = u[0,:]
u2_nm3 = u[1,:]
u3_nm3 = u[2,:]

# Algorithm 4: HHT-alpha #
alpha = -0.1
delta = 0.5 * (1-2*alpha)
beta = 0.25 * (1-alpha)**2
#Test 1:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f1,u_0,v_0,t_a,N,h)
u1_hht1 = u[0,:]
u2_hht1 = u[1,:]
u3_hht1 = u[2,:]

#Test 2:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f2,u_0,v_0,t_a,N,h)
u1_hht2 = u[0,:]
u2_hht2 = u[1,:]
u3_hht2 = u[2,:]

#Test 3:
u,v,a = Newmark.Newmark_beta(delta,beta,K_a,M_a,C_a,test_f3,u_0,v_0,t_a,N,h,dt_ft)
u1_hht3 = u[0,:]
u2_hht3 = u[1,:]
u3_hht3 = u[2,:]


##Comparison for different algorithms, test1

fig, ax = plt.subplots()
plt.ylabel("u1")
plt.plot(np.arange(0,5.005,0.005),u1_w2,color = 'b',label = "Wilson")
plt.plot(np.arange(0,5.005,0.005),u1_ssp2,color = 'r',label = "SSP-RK3")
plt.plot(np.arange(0,5.005,0.005),u1_nm2,color = 'y',label = "Newmark")
plt.plot(np.arange(0,5.005,0.005),u1_hht2,color = 'c',label = "HHT")
plt.legend()

fig, ax = plt.subplots()
plt.ylabel("u2")
#plt.plot(np.arange(0,5.005,0.005),u2_w3,color = 'b',label = "Wilson")
#plt.plot(np.arange(0,5.005,0.005),u2_ssp3,color = 'r',label = "SSP-RK3")
plt.plot(np.arange(0,5.005,0.005),u2_nm1,color = 'y',label = "Newmark")
#plt.plot(np.arange(0,5.005,0.005),u2_hht1,color = 'c',label = "HHT")
plt.legend()

fig, ax = plt.subplots()
plt.ylabel("u3")
#plt.plot(np.arange(0,5.005,0.005),u3_w3,color = 'b',label = "Wilson")
#plt.plot(np.arange(0,5.005,0.005),u3_ssp3,color = 'r',label = "SSP-RK3")
plt.plot(np.arange(0,5.005,0.005),u3_nm1,color = 'y',label = "Newmark")
#plt.plot(np.arange(0,5.005,0.005),u3_hht1,color = 'c',label = "HHT")
plt.legend()
