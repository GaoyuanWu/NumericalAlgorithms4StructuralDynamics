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
#%%
'''
This is a study of different algorithms on SDOF system.
Algorithms:
    1. SSP-RK3
    2. Wilson-theta (theta=1.4,1.6)
    3. Newmark-beta (delta=0.5,beta=0.25; delta=0.5,beta=0.2; delta= 0.6,beta=0.2)

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
c = alpha * m + beta * k
C = alpha * M + beta * K
print("Damping: ", C)


#####################################
#Test 1
####################################
#Test load 1 : Sine wave, 1000 cos(wt) KN
w = 2
p0 = 10**5
def f_sin(t,w = w,p0 = p0):
    return p0 * np.sin(w*t)
f_t = f_sin
def test_f1(x):
    f_list = np.array([[f_sin,0]])
    vec = np.zeros(n)
    for i in range(f_list.shape[0]):
        pos = f_list[i,1]
        vec[pos] = f_list[i,0](w,x)
    return vec 
#Exact solution
def ex_sdof_sin(w,t,u0,v0,k,m,c,p0):
    u_st = p0/k
    w_n = np.sqrt(k/m)
    c_r = c/(2*m*w_n)
    w_d = w_n*np.sqrt(1-c_r**2)
    beta = w/w_n
    C = u_st * (1-beta**2)/((1-beta**2)**2 + (2*c_r*beta)**2)
    D = u_st * (-2*c_r*beta)/((1-beta**2)**2 + (2*c_r*beta)**2)
    B = u0 - D
    A = (v0-C*w)/w_d
    Transient = np.exp(-c_r*w_n*t)*(A * np.sin(w_d *t) + B*np.cos(w_d*t))
    Steady = C * np.sin(w*t) + D * np.cos(w*t)
    return Transient + Steady
    
    
    
    
#Parameters
u0 = np.zeros(n)
v0 = np.zeros(n)

t_a = 0
t_b = 5 
h_list = [0.2,0.1,0.005]
N_list = [25,50,1000]

w_list = [5.5,11.5]

method_list = [SSPRK3.SSP_RK3,Wilson.Wilson_theta,Wilson.Wilson_theta,Newmark.Newmark_beta,Newmark.Newmark_beta,Newmark.Newmark_beta,Bathe.Bathe]

coef = np.array([[1.4,0],[2,0],[0.5,0.25],[0.5,1/6],[11/20,3/10]]) #Coefficients for different algorithms

color_list = ['b','brown','r','y','c','m','k','peachpuff']
label_list = ['SSP-RK3',r'Wilson, $\theta = 1.4$',r'Wilson, $ \theta = 2$',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.25',r'Newmark, $\delta$ = 0.5, $\beta$ = $\frac{1}{6}$',r'Newmark, $\delta$ = $\frac{11}{20}$, $\beta$ = $ \frac{3}{10}$ ','Bathe']

#%%
'''
Plots of time history for displacement for different h & frequency.
Showing SSP-RK3 is not unconditionally stable.
Showing resonance.
'''

#Implementation & Plot
for i in range(len(h_list)):
    h = h_list[i] #Time step
    N = N_list[i]
    for i_1 in range(len(w_list)):
        w = w_list[i_1]
        p0 = 10**5
        f = test_f1
        fig,ax = plt.subplots()
        ax.set_title(r'$f(t) = 1000cos(\omega t), \omega = {}$, h = {}'.format(w,h),fontsize = 20)
        ax.set_xlabel('t(s)',fontsize = 20)
        ax.set_ylabel('Displacement (m)',fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.plot(np.linspace(t_a,t_b,2000),ex_sdof_sin(w,np.linspace(t_a,t_b,2000),u0,v0,k,m,c,p0),marker = 'x',label = "exact")
        for j in range(len(method_list)):
            if j == 0:
                u_temp,v_temp = method_list[j](K,M,C,f,u0,v0,t_a,N,h)
            elif j <= 2:
                theta = coef[j-1,0]
                u_temp,v_temp,a_temp = method_list[j](theta,K,M,C,f,u0,v0,t_a,N,h)
            elif j<6:
                delta = coef[j-1,0]
                beta = coef[j-1,1]
                u_temp,v_temp,a_temp = method_list[j](delta,beta,K,M,C,f,u0,v0,t_a,N,h)
            else:
                u_temp,v_temp,a_temp = method_list[j](K,M,C,f,u0,v0,t_a,N,h)
                
            plt.plot(np.linspace(t_a,t_b,N+1),u_temp.T[:],color = color_list[j],label = label_list[j])
            plt.legend(fontsize = 15)
            
#%%
'''
Do a convergence test.
w = 5.5, p0 = 1000
Error of displacement at t_b for different algorithms versus grid spacing h
h = np.arange(10,2010,400)
'''

# Settings
u0 = np.zeros(n)
v0 = np.zeros(n)
t_a = 0
t_b = 5 
N = np.arange(100,10000,2000)
N_num = len(N)
method_list = [SSPRK3.SSP_RK3,Wilson.Wilson_theta,Wilson.Wilson_theta,Newmark.Newmark_beta,Newmark.Newmark_beta,Newmark.Newmark_beta,Bathe.Bathe]
coef = np.array([[1.4,0],[1.6,0],[0.5,0.25],[0.5,1/6],[11/20,3/10]]) #Coefficients for different algorithms
algo_err = np.zeros((len(method_list),N_num)) #Storing errors

p0 = 10**5
w = 5.5

# Implementation
for i in range (N_num):
    for j in range (len(method_list)):
        N_temp = int(N[i])
        h_temp = (t_b - t_a)/N_temp
        exa = ex_sdof_sin(w,t_b,u0,v0,k,m,c,p0)
        if j == 0:
            u_temp = method_list[j](K,M,C,f,u0,v0,t_a,N_temp,h_temp)[0]
            numerical = u_temp[N_temp]
        elif j <= 2:
            theta = coef[j-1,0]
            u_temp = method_list[j](theta,K,M,C,f,u0,v0,t_a,N_temp,h_temp)[0]
            numerical = u_temp[0,N_temp]
        elif j<6:
            delta = coef[j-1,0]
            beta = coef[j-1,1]
            u_temp = method_list[j](delta,beta,K,M,C,f,u0,v0,t_a,N_temp,h_temp)[0]
            numerical = u_temp[0,N_temp]
        else:
            u_temp = method_list[j](K,M,C,f,u0,v0,t_a,N_temp,h_temp)[0]
            numerical = u_temp[0,N_temp]
        algo_err[j,i] = np.abs(exa-numerical)
        
# Plot

fig,ax = plt.subplots(figsize=(10,14))
ax.set_title(r'Convergence test for SDOF system, $\omega = {}$'.format(w),fontsize = 20)
ax.set_xlabel('h,Grid Spacing',fontsize = 20)
ax.set_ylabel(r'Error for $u(5)$',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
for i in range (len(method_list)):
    plt.plot((t_b - t_a)/N,algo_err[i,:],color = color_list[i],label = label_list[i])
plt.plot((t_b - t_a)/N,((t_b - t_a)/N*4)**2,linestyle = "dashed")
plt.plot((t_b - t_a)/N,((t_b - t_a)/N*5)**3 ,linestyle = "dashed")
ax.text(0.05,(0.05*4)**2,str( r'$ h^2$'),fontsize = 20)
ax.text(0.05,(0.05*4)**3,str( r'$ h^3$'),fontsize = 20)
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(fontsize = 15)   
#%%

#####################################
#Test 2
####################################
#Test 2: Free vibration with different damping ratio



def ex_sdof_free(t,u0,v0,k,m,c_r):
    w_n = np.sqrt(k/m)
    w_d = w_n*np.sqrt(1-c_r**2)
    pt_1 = u0 * np.cos(w_d * t)
    pt_2 = (v0 + c_r * w_n * u0) * np.sin(w_d * t)/w_d
    return np.exp(-c_r * w_n * t) * (pt_1 + pt_2)
def test_f2(x):
    vec = np.zeros(n)
    return vec


#SDOF m,k
k = 800 * 10**3 * np.ones(1) #800 kN/m
m = 6000 * np.ones(1) #kg
n = 1

K,M = getMK.getMK_SDOF(k,m)

#Damping ratio
c_ratio_list = np.array([0, 0.05,0.1])
c_list = c_ratio_list * 2 * m * np.sqrt(k/m)

    
#Parameters
u0 = np.ones(n)
v0 = np.zeros(n)

t_a = 0
t_b = 5 
h_list = [0.5,0.01,0.005]
N_list = [10,500,1000]

method_list = [SSPRK3.SSP_RK3,Wilson.Wilson_theta,Wilson.Wilson_theta,Newmark.Newmark_beta,Newmark.Newmark_beta,Newmark.Newmark_beta]

coef = np.array([[1.4,0],[1.6,0],[0.5,0.25],[0.5,0.2],[0.6,0.2]]) #Coefficients for different algorithms

color_list = ['b','brown','r','y','c','m','pink']
label_list = ['SSP-RK3',r'Wilson, $\theta = 1.4$',r'Wilson, $ \theta = 1.6$',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.25',r'Newmark, $\delta$ = 0.5, $\beta$ = 0.2',r'Newmark, $\delta$ = 0.6, $\beta$ = 0.2']

#%%
'''
Plots of time history for displacement for different damping ratio.
Showing SSP-RK3 is not unconditionally stable.
'''

#Implementation & Plot
for i in range(len(h_list)):
    h = h_list[i] #Time step
    N = N_list[i]
    for i_1 in range(c_list.shape[0]):
        c = c_list[i_1]
        c_ratio = c_ratio_list[i_1]
        C = np.array([[c]])
        fig,ax = plt.subplots()
        ax.set_title(r'$ Free vibration, \xi = {}, h = {} $'.format(c_ratio,h),fontsize = 20)
        ax.set_xlabel('t(s)',fontsize = 20)
        ax.set_ylabel('Displacement (m)',fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.plot(np.linspace(t_a,t_b,2000),ex_sdof_free(np.linspace(t_a,t_b,2000),u0,v0,k,m,c_ratio),linewidth = 2,color = "k",label = "exact")
        for j in range(len(method_list)):
            if j == 0:
                u_temp,v_temp = method_list[j](K,M,C,test_f2,u0,v0,t_a,N,h)
            elif j <= 2:
                theta = coef[j-1,0]
                u_temp,v_temp,a_temp = method_list[j](theta,K,M,C,test_f2,u0,v0,t_a,N,h)
            else:
                delta = coef[j-1,0]
                beta = coef[j-1,1]
                u_temp,v_temp,a_temp = method_list[j](delta,beta,K,M,C,test_f2,u0,v0,t_a,N,h)
                
            plt.plot(np.linspace(t_a,t_b,N+1),u_temp.T[:],color = color_list[j],label = label_list[j])
            plt.legend(fontsize = 20)

 
