import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = 10e6
k = 10e3

G = 10000*k    # Shear modulus 
K = 20000*k    # Bulk modulus 

steady_state_angle = 30 * np.pi/180
peak_dilatance_angle = 5*np.pi/180


PLOT_FOLDER = "plots"

A = 1 

M_f = (6 * np.sin(steady_state_angle))/(3-np.sin(steady_state_angle))
M_Q = (6 * np.sin(peak_dilatance_angle))/(3-np.sin(peak_dilatance_angle))


print(f"steady-angle {steady_state_angle}, peak_angle {peak_dilatance_angle}")
print(f"M_f: {M_f}, M_Q {M_Q}")

epsilon_py = 0.04   # ulitimate strength 
p_psi = 400*k       # mean stress
p_nodil = 200*k     # mean stress for no dilantancy 

def chi(x: float) -> float: 
    
    result = (2*np.sqrt(x))/(1+x)
    
    return result

def M_F(x: float) -> float : 
    
    result = chi(x)*M_f
    
    return result

def delta_M_over_delta_epsilon_q(x: float, epsilon_a: float) -> float: 
    
    result = (M_f/epsilon_a)*((1-x)/(np.sqrt(x)*(1+x)**2))
    
    return result 


def delta_p () -> float : 
    
    result = (3*G*K*M_Q)/(A+(K*M_f*M_Q)+3*G)

    return result 


def delta_q() -> float: 
    
    result = (3*G-(9*G**2)/(A+(K*M_f*M_Q)+3*G))
    
    return result 



x = np.linspace(0.001,1.00,num=24)

data_colums = ["x","chi","epsilon_p_q", "m_f","delta_MP/delta_epsilon"]


data = []
for i in x:
    chi_i = chi(i)
    eps_p_q = i*epsilon_py
    m_f = i*M_f
    delta_m_p_over_delta_epsilon = delta_M_over_delta_epsilon_q(i,epsilon_py)
    new_data_row = [i,chi_i,eps_p_q,m_f,delta_m_p_over_delta_epsilon]
    data.append(new_data_row)
    
df = pd.DataFrame(data,columns=data_colums)   
print(f"df \n {df}")


plt.clf()
df.plot(x="x",y="chi",kind="line",figsize=(10,10))
fig_name = Path.cwd()/PLOT_FOLDER/"normalized_hardening_curve.png"
plt.grid()
plt.xlabel("Relative plastic strain, x")
plt.ylabel("Relative mobilized q/p' ratio")
plt.title("Normalized hardening curve")
plt.savefig(fig_name)

plt.clf()
df.plot(x="epsilon_p_q",y="m_f",kind="line",figsize=(10,10))
plt.grid()
plt.xlabel("Accumulated distortional strain, x")
plt.ylabel("Mobilized M_f=q/p' ratio")
plt.title("Normalized hardening curve")
fig_name = Path.cwd()/PLOT_FOLDER/"hardening_curve.png"
plt.savefig(fig_name)







# rho = 1 
# psi = 1 
# N_F = (1+np.sin(rho))/(1-np.sin(rho))
# N_Q = (1+np.sin(psi))/(1-np.sin(psi))

# def calculate_yield(sigma1: float, sigma3, N_f:float): 
    
    
    
#     return 


# def elastic_range(delta_sigma1: float, delta_sigma3: float) -> list[float,float]: 
    
#     g_matrix = np.matrix([[(3*K+4*G)/(12*K+4*G),((-3*K+2*G)/(12*K+4*G))],[(-3*K+2*G)/(12*K+4*G),(3*K+4*G)/(12*K+4*G)]])
        
#     D = (1/G)*g_matrix 
    
#     delta_sigma = np.array([delta_sigma1,delta_sigma3])
    
#     delta_epsilon = np.matmul(D,delta_sigma) 
    
#     delta_epsilon1 = delta_epsilon[0]
#     delta_epsilon3 = delta_epsilon[1]
    
#     return delta_epsilon1, delta_epsilon3


# def elastic_plastic_range(delta_sigma1: float, delta_sigma3: float) -> list[float,float]: 
    
#     g_matrix = np.matrix([[(3*K+4*G)/(12*K+4*G),((-3*K+2*G)/(12*K+4*G))],[(-3*K+2*G)/(12*K+4*G),(3*K+4*G)/(12*K+4*G)]])
    
#     a_matrix = np.matrix([[1,-N_F],[-N_Q,N_F*N_Q]])
    
#     D = (1/G)*g_matrix + (1/A)*a_matrix
    
#     delta_sigma = np.array([delta_sigma1,delta_sigma3])
    
#     delta_epsilon = np.matmul(D,delta_sigma) 
    
#     delta_epsilon1 = delta_epsilon[0]
#     delta_epsilon3 = delta_epsilon[1]
    
#     return delta_epsilon1, delta_epsilon3