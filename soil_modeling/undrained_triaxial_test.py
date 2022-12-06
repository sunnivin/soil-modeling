import math
import sys

import numpy as np


def M(theta: float,type: str)-> float: 

    if type=="compression":
        denominator =  3-np.sin(theta)
    elif type=="extension":
        denominator =  3+np.sin(theta)
    else:
        sys.error("unknown ")

    M = 6*np.sin(theta)/denominator
    
    return M 


def Delta_p0(p0: float, p: float, delta_p: float, kappa: float, Lambda : float): 
    
    delta_p0 = p0*(((p+delta_p)/p)**(-kappa/(Lambda-kappa))-1)
    
    return delta_p0 


def Q_and_delta_q(M: float, p: float, delta_p: float, p0:float, delta_p0:float): 
    
    
    q_and_delta_q = M*np.sqrt((p+delta_p)*(p0+delta_p0-(p+delta_p)))
    
    return q_and_delta_q

def Delta_epsilon_p(Lambda: float, kappa: float, p0:float, delta_p0:float)-> float: 
    
    delta_epsilon_p = -(Lambda-kappa)*math.log((p0+delta_p0)/p0)
    
    return delta_epsilon_p
theta = 25.33
m = M(theta,type="compression")
print(f"m: {round(m,3)}")

m=1 
p0 = 300
p = 100
delta_p = 10
Lambda = 0.125 
kappa = 0.0025

delta_p0 = Delta_p0(p0=p0,p=p,delta_p=delta_p,kappa=kappa,Lambda=Lambda)
print(f"delta_p0: {round(delta_p0,3)}")

q_and_delta_q = Q_and_delta_q(M=m,p=p,delta_p=delta_p,p0=p0,delta_p0=delta_p0)
print(f"q_and_delta_q: {round(q_and_delta_q,3)}")

delta_epsilon_p = Delta_epsilon_p(Lambda=Lambda,kappa=kappa,p0=p0,delta_p0=delta_p0)
print(f"delta_epsilon_p: {round(delta_epsilon_p,6)}")