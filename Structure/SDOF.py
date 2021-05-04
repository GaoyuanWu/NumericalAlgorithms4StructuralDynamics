import sys
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/Structure')
sys.path.append('D:/GitHubRepo/NumericalAlgorithms4StructuralDynamics/TimeIntegration')
import getMK as getMK
import numpy as np
import SSPRK3 as SSPRK3
import matplotlib.pyplot as plt
import Wilson as Wilson
import Newmark as Newmark

'''
This is a study of different algorithms on SDOF system.
Algorithms:
    1. SSP-RK3
    2. Wilson-theta (theta=1.4,1.6)
    3. Newmark-beta (delta=0.5,beta=0.25; delta=0.5,beta=0.2)
    4. HHT

Accuracy & stability analysis is conducted.

External loading:
    1.Sin wave (different frequency chosen to show resonance)
    2.Free vibration with different damping

'''

#SDOF m,k
k = 800 * 10**3 * np.ones(1) #800 kN/m
m = 6000 * np.ones(1) #kg
n = 1

K,M = getMK.getMK_SDOF(k,m)
print("Stiffness matrix: ", K)
print("Mass matrix: ", M)

#Propotional Damping
alpha = 0
beta = 0
C = alpha * M + beta * K
print("Damping: ", C)


#####################################
#Test 1
####################################
#Test load 1 : Sine wave, 1000 cos(wt) KN
def f_sin(w,x):
    return 10**6 * np.sin(w*x)
def test_f1(x):
    f_list = np.array([[f_sin,0]])
    vec = np.zeros(n)
    for i in range(f_list.shape[0]):
        pos = f_list[i,1]
        vec[pos] = f_list[i,0](w,x)
    return vec 
#
def ex_sin(w,x):
    u_st = 10**6/k
    
#Parameters
u0 = np.zeros(n)
v0 = np.zeros(n)

t_a = 0
t_b = 10 
h_list = [0.5,0.1,0.02]
N_list = [20,100,200]

w_list = [11]

method_list = [SSPRK3.SSP_RK3,Wilson.Wilson_theta,Wilson.Wilson_theta,Newmark.Newmark_beta,Newmark.Newmark_beta,Newmark.Newmark_beta]

coef = np.array([[1.4,0],[1.6,0],[0.5,0.25],[0.5,0.2],[0.6,0.2]]) #Coefficients for different algorithms

color_list = ['b','k','r','y','c','m']
label_list = ['SSP-RK3',r'Wilson, $\theta = 1.4$',r'Wilson, $ \theta = 1.6$',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.25',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.2',r'Newmark, $\delta$ = 0.6, $\beta$ = 0.2']

#Implementation
for i in range(len(h_list)):
    h = h_list[i] #Time step
    N = N_list[i]
    f = test_f1
    for i_1 in range(len(w_list)):
        w = w_list[i_1]
        fig,ax = plt.subplots()
        ax.set_title(r'$f(t) = 1000cos(wt), w = {}$, h = {}'.format(w,h))
        ax.set_xlabel('t(s)')
        ax.set_ylabel('Displacement (m)')
        plt.plot(np.linspace(t_a,t_b,N+1),u_temp.T[:],color = "bisque",label = "exact")
        for j in range(len(method_list)):
            if j == 0:
                u_temp,v_temp = method_list[j](K,M,C,f,u0,v0,t_a,N,h)
            elif j <= 2:
                theta = coef[j-1,0]
                u_temp,v_temp,a_temp = method_list[j](theta,K,M,C,f,u0,v0,t_a,N,h)
            else:
                delta = coef[j-1,0]
                beta = coef[j-1,1]
                u_temp,v_temp,a_temp = method_list[j](delta,beta,K,M,C,f,u0,v0,t_a,N,h)
                
            plt.plot(np.linspace(t_a,t_b,N+1),u_temp.T[:],color = color_list[j],label = label_list[j])
            plt.legend()

            


