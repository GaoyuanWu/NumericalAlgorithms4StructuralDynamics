import sys
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/Structure')
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/TimeIntegration')
import getMK as getMK
import numpy as np
import SSPRK3 as SSPRK3
import matplotlib.pyplot as plt
import Wilson as Wilson
import Newmark as Newmark
import Bathe as Bathe
import get1DMesh as get1DMesh
import getPlot as getPlot
import getBC as getBC
import pandas as pd
import time
#%%
'''
This is a study of different algorithms on MDOF system.
The spatial discretization utilizes finite-element-method (FEM).

Algorithms:
    1. SSP-RK3
    2. Wilson-theta (theta=1.4,1.6)
    3. Newmark-beta (delta=0.5,beta=0.25; delta=0.5,beta=0.2; delta= 0.6,beta=0.2)


Study will do an analysis on the "displacement of a node" versus "mesh size".

The prototype problem is a 3-story shear building with 3 blocks.

External loading:
    1.Constant load
    2.Ground Motion

'''


#Initial structure
test_nodecoord = np.array([[0,0],[10,0],[20,0],[30,0],[0,5],[10,5],[20,5],[30,5],[10,10],[20,10],[10,15],[20,15]])
test_B = np.array ([[1,5],[2,6],[3,7],[4,8],[5,6],[6,7],[7,8],[6,9],[7,10],[9,10],[9,11],[10,12],[11,12]])

#Specify Column, Beam elements
beam_ini_ele = [4,5,6,9,12]
col_ini_ele = [0,1,2,3,7,8,10,11]


#Fixed BC
#Node 1-4 are fixed
bc = {0:1,1:1,2:1,3:1} 

#Plot
getPlot.getPlot(test_nodecoord,test_B,True,True,15)

# 1D MESH: Mesh list
h_list = [0,5,2.5,1,0.5] #mesh size , 0 denotes original node 

# Structural member property, unit:m, KN
col_A =  1 
beam_A = 0.24
E = 3.79 * 10**7
col_I =  1/12
beam_I = 0.4 * 0.6**3/12
m_unit = 2.4473 

#Propotional Damping coefficients
alpha = 0
beta = 0

#External loading
#functional input or time-step input
# dof_index : 0:x, 1:y, 2:rz

#Test External loading 1: acting on node 11 (10 in py), postive x direction
def f1(x):
    return 1000
f1_ini_node = 8
f1_dof_index = 0

#Test External loading 2: Ground Motion-Kern County(1952), LA - Hollywood Stor FF
#PGA = 0.1g
# -> 10 times -> To g
#dt = 0.005
#Read xlsx
GM_tab = pd.read_excel('GM.xlsx')
GM_data = 10 * np.array(GM_tab.values)
dt_ft = 0.005

#%%



###########
#Algorithm setup 
###########


# Test f1, t_a = 0, t_b = 5
t_a1 = 0
t_b1 = 5
N_list_f1 = np.linspace(10,1010,5,endpoint=True)
h_list_f1 = (t_b1 - t_a1)/N_list_f1

# Test f2, t_a = 0 ,t_b = 20
t_a2 = 0
t_b2 = 20
N_list_f2 = np.linspace(40,4040,5,endpoint=True)
h_list_f2 = (t_b2 - t_a2)/N_list_f2

method_list = [SSPRK3.SSP_RK3,Wilson.Wilson_theta,Newmark.Newmark_beta,Bathe.Bathe]
coef = np.array([[1.4,0],[0.5,0.25]]) #Coefficients for different algorithms
color_list = ['b','brown','r','y']
label_list = ['SSP-RK3',r'Wilson, $\theta = 1.4$',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.25','Bathe']


###### Accuracy study for h_space 
# For study for spatial discretization, N_time is set as 4000
# u_max at node 11(10 in py) versus h
u_max_li_f1 = np.zeros((len(h_list),len(method_list)))
u_max_li_f2 = np.zeros((len(h_list),len(method_list)))

u_li_f1 = np.zeros((len(h_list),len(method_list),4001))
u_li_f2 = np.zeros((len(h_list),len(method_list),4001))               

runtime = np.zeros(len(method_list)-1)
#%%
for i in range(len(h_list)):
    if h_list[i] == 0:
        new_nodecoord = test_nodecoord
        new_B = test_B
        itn = np.linspace(0,test_nodecoord.shape[0],test_nodecoord.shape[0],dtype = np.int32)
        itn_ele = np.linspace(0,test_B.shape[0],test_B.shape[0],dtype = np.int32)
    else:
        h_space = h_list[i]
        new_nodecoord,new_B,itn,itn_ele = get1DMesh.get1DMesh(test_nodecoord, test_B, h_space) #1D MESH
    ele_A = np.zeros(new_B.shape[0])
    ele_m = np.zeros(new_B.shape[0])
    ele_I = np.zeros(new_B.shape[0])
    ele_E = E * np.ones(new_B.shape[0])
    for l in range(itn_ele.shape[0]):
        if itn_ele[l] in beam_ini_ele:
            temp_A = beam_A
            ele_A[l] = temp_A
            ele_m[l] = temp_A * m_unit
            ele_I[l] = beam_I
        else:
            temp_A = col_A
            ele_A[l] = temp_A
            ele_m[l] = temp_A * m_unit
            ele_I[l] = col_I
    K,M = getMK.getMK_FEM(new_nodecoord,new_B, ele_m, ele_E, ele_A, ele_I) #FEM
    C = alpha * M + beta * K
    M_bc,K_bc,inact_num,glb_to_bc = getBC.MK_Reduction(M, K, bc, itn) #Impose Boundary Conditions
    C_bc = alpha * M_bc + beta * K_bc
    act_num = K_bc.shape[0] - inact_num
    K_a = K_bc[:act_num,:act_num]
    M_a = M_bc[:act_num,:act_num]
    
    
    
    C_a = C_bc[:act_num,:act_num]
    u_0 = np.zeros(act_num) # IV
    v_0 = np.zeros(act_num) # IV 
            
    node = int(glb_to_bc[itn[10] * 3])#Node 11 (10 in PY) in new sequence
    def test_f(x,itn = itn,glb_to_bc = glb_to_bc,f = f1,f_ini_node = f1_ini_node,dof_index = f1_dof_index,act_num = act_num):
        f_list = np.array([[f,glb_to_bc[itn[f1_ini_node]*3 + dof_index]]])
        vec = np.zeros(act_num)
        for i in range(f_list.shape[0]):
            pos = f_list[i,1]
            vec[pos] = f_list[i,0](x)
        return vec  

    def test_gm(M_a,act_num = act_num,GM_data = GM_data):
        return - M_a @ (np.ones((act_num,1))@(GM_data.T))
    test_f2 = test_gm(M_a)
            
    N_t = 4000
    for j in range(2):
        for k in range(len(method_list)):
            if j == 0:
                h_t = (t_b1-t_a1)/N_t
                if k == 0 :
                    U_temp = method_list[k](K_a,M_a,C_a,test_f,u_0,v_0,t_a1,N_t,h_t)
                    u_li_f1[i,k,:] = U_temp[node,:]
                elif k == 1:
                    U_temp = (method_list[k](coef[0,0],K_a,M_a,C_a,test_f,u_0,v_0,t_a1,N_t,h_t))[0]
                    u_li_f1[i,k,:] = U_temp[node,:]
                elif k == 2:
                    U_temp = (method_list[k](coef[1,0],coef[1,1],K_a,M_a,C_a,test_f,u_0,v_0,t_a1,N_t,h_t))[0] 
                    u_li_f1[i,k,:] = U_temp[node,:]
                else:
                    U_temp = (method_list[k](K_a,M_a,C_a,test_f,u_0,v_0,t_a1,N_t,h_t))[0]
                    u_li_f1[i,k,:] = U_temp[node,:]
                    
                temp_max = max(np.abs(U_temp[node,:]))  
                u_max_li_f1[i,k] = temp_max
            if j == 1:
                h_t = (t_b2-t_a2)/N_t
                if k == 0:
                    U_temp = method_list[k](K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft)
                    u_li_f2[i,k,:] = U_temp[node,:]
                elif k == 1:
                    start_time = time.time()
                    U_temp = (method_list[k](coef[0,0],K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
                    if i == len(h_list) - 1:
                        runtime[k-1] = time.time() - start_time
                    u_li_f2[i,k,:] = U_temp[node,:]
                elif k == 2:
                    start_time = time.time()
                    U_temp = (method_list[k](coef[1,0],coef[1,1],K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
                    if i == len(h_list) - 1:
                        runtime[k-1] = time.time() - start_time
                    u_li_f2[i,k,:] = U_temp[node,:]
                else:
                    start_time = time.time()
                    U_temp = (method_list[k](K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
                    if i == len(h_list) - 1:
                        runtime[k-1] = time.time() - start_time
                    u_li_f2[i,k,:] = U_temp[node,:]
                    
                u_max_li_f2[i,k] = max(np.abs(U_temp[node,:]))  
            print(i ,j , k)


#%%
'''
Plots of MAX disp during earthquake versus h_FEM for different algorithms.
'''

#Implementation & Plot
fig,ax = plt.subplots(figsize = (20,20))
ax.set_title ('Maximum displacement at node 11 during earthquake',fontsize = 20)
ax.set_xlabel('h(m)',fontsize = 20)
ax.set_ylabel(r'$u_{max}^{11} (cm)$',fontsize = 20)
ax.set_ylim(0.9,2.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
h_l_plt = [7.5,5,2.5,1,0.5]
for i in range(len(method_list)):
    plt.plot(h_l_plt[:],u_max_li_f2[:,i] * 100 ,marker = "x" ,color = color_list[i],label = label_list[i])
    plt.legend(fontsize = 20)
#%%
'''
Plots of the error of MAX disp during earthquake versus h_FEM for different algorithms.
'''
#Implementation & Plot
benchmark_max = 0.01682
fig,ax = plt.subplots(figsize = (20,20))
ax.set_title ('Error of maximum displacement at node 11',fontsize = 20)
ax.set_xlabel('h(m)',fontsize = 20)
ax.set_ylabel(r'Error: $u_{max}^{11} (cm)$',fontsize = 20)
ax.set_ylim(10**-5,10**-1)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.yscale('log')
plt.xscale('log')
h_l_plt = np.array([7.5,5,2.5,1,0.5])
for i in range(len(method_list)):
    plt.plot(h_l_plt[:],abs(u_max_li_f2[:,i] -0.01682)  ,color = color_list[i],label = label_list[i])
    plt.legend(fontsize = 20)
plt.plot(h_l_plt[:], (h_l_plt[:]/100) ** 2 ,linestyle = "dashed")
ax.text(0.5,(0.5/100)**2,str( r'$ h^2$'),fontsize = 20)
#%%
#Get SAP2000 result
sap_tab = pd.read_excel('Disp.xlsx')
sap_data = np.array(sap_tab.values) [:,1]
#%%

'''
Plot of the time history during earthquake
'''
fig,ax = plt.subplots(figsize = (20,20))
ax.set_title ('Time history of node 11 during earthquake',fontsize = 20)
ax.set_xlabel('t(s)',fontsize = 20)
ax.set_ylabel(r'$u^{11} (cm)$',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.plot(np.linspace(0,20,4001,endpoint=True),100*u_li_f2[4,1] ,color = 'b',label = label_list[1])
plt.plot(np.linspace(0,20,4001,endpoint=True),100*u_li_f2[4,2] ,color = color_list[2],label = label_list[2])
plt.plot(np.linspace(0,20,4001,endpoint=True),100*u_li_f2[4,3] ,color = color_list[3],label = label_list[3])
plt.plot(np.linspace(0,20,4001,endpoint=True),100*sap_data ,color = 'k',label = 'SAP 2000,Modal Superposition')

plt.legend(fontsize=20)



#%%
'''
Study on grid size for time integration

'''
###### Accuracy study for h_time
# N_fem is set as 0.5
# u_max at node 11(10 in py) versus h
N_list_f2 = np.array([20,400,800,1200,2000,4000,10000]).astype(int)
h_list_f2 = (t_b2 - t_a2)/N_list_f2

u_max_time_f2 = np.zeros((int(h_list_f2.shape[0]),len(method_list)))

u_time_f2 = np.zeros((h_list_f2.shape[0],len(method_list),10001))    
#%%
for i in range(h_list_f2.shape[0]):
    for k in range(len(method_list)):
        h_space = 0.5
        new_nodecoord,new_B,itn,itn_ele = get1DMesh.get1DMesh(test_nodecoord, test_B, h_space) #1D MESH
        ele_A = np.zeros(new_B.shape[0])
        ele_m = np.zeros(new_B.shape[0])
        ele_I = np.zeros(new_B.shape[0])
        ele_E = E * np.ones(new_B.shape[0])
        for l in range(itn_ele.shape[0]):
            if itn_ele[l] in beam_ini_ele:
                temp_A = beam_A
                ele_A[l] = temp_A
                ele_m[l] = temp_A * m_unit
                ele_I[l] = beam_I
            else:
                temp_A = col_A
                ele_A[l] = temp_A
                ele_m[l] = temp_A * m_unit
                ele_I[l] = col_I
        K,M = getMK.getMK_FEM(new_nodecoord,new_B, ele_m, ele_E, ele_A, ele_I) #FEM
        C = alpha * M + beta * K
        M_bc,K_bc,inact_num,glb_to_bc = getBC.MK_Reduction(M, K, bc, itn) #Impose Boundary Conditions
        C_bc = alpha * M_bc + beta * K_bc
        act_num = K_bc.shape[0] - inact_num
        K_a = K_bc[:act_num,:act_num]
        M_a = M_bc[:act_num,:act_num]
        C_a = C_bc[:act_num,:act_num]
        u_0 = np.zeros(act_num) # IV
        v_0 = np.zeros(act_num) # IV 
        
        node = int(glb_to_bc[itn[10] * 3])#Node 11 (10 in PY) in new sequence

        def test_gm(M_a,act_num = act_num,GM_data = GM_data):
            return - M_a @ (np.ones((act_num,1))@(GM_data.T))
        test_f2 = test_gm(M_a)
            
        N_t = N_list_f2[i]
        h_t = h_list_f2[i]
        if k == 0:
            U_temp = method_list[k](K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft)
            u_time_f2[i,k,:N_list_f2[i] +1] = U_temp[node,:]
        elif k == 1:
            U_temp = (method_list[k](coef[0,0],K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
            u_time_f2[i,k,:N_list_f2[i] +1] = U_temp[node,:]
        elif k == 2:
            U_temp = (method_list[k](coef[1,0],coef[1,1],K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
            u_time_f2[i,k,:N_list_f2[i] +1] = U_temp[node,:]
        else:
            U_temp = (method_list[k](K_a,M_a,C_a,test_f2,u_0,v_0,t_a1,N_t,h_t,dt_ft))[0]
            u_time_f2[i,k,:N_list_f2[i] +1] = U_temp[node,:]
            
        u_max_time_f2[i,k] = max(np.abs(U_temp[node,:]))  
        print(i, k)
#%%
'''
Plots of MAX disp during earthquake versus h_time for different algorithms.
'''
benchmark_max = 0.01682

#Implementation & Plot
fig,ax = plt.subplots(figsize = (14,14))
ax.set_title ('Error of maximum displacement at node 11',fontsize = 20)
ax.set_xlabel('h(s)',fontsize = 20)
ax.set_ylabel(r'Error:$u_{max}^{11} (cm)$',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax.set_ylim(10**-6,0.1)
plt.yscale('log')
plt.xscale('log')
for i in range(len(method_list)):
    plt.plot(h_list_f2,abs(u_max_time_f2[:,i]-benchmark_max)  ,marker='x',color = color_list[i],label = label_list[i])
    plt.legend(fontsize = 20)
