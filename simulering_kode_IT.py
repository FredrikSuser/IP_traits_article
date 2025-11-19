# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:58:45 2025

@author: fse01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 100
alpha = 1
beta = 1
alpha_values = [0.01, 0.1, 0.5, 0.9,1.2]
beta_values = [0.20, 0.24, 0.25, 0.28]
mu = 0.1
tmax = 15000

# vary beta and keep alpha fixed as well.

# def construct_tridiagonal_matrix(N, alpha, beta):
#     A = np.zeros((N, N))

#     for k in range(1, N + 1):
#         if k > 1:
#             A[k-1, k-2] = (alpha * (k - 1) * (N - (k - 1)) * (N - beta *( k - 1))) / (N**2 * (N - 1))
        
#         A[k-1, k-1] = (N *(N-1) * (N - beta * k) - alpha * (k * N * (N - k) + beta * k * (2 * k**2 + 1 + N - 2 * k * (N + 1)))) / (N**2 * (N - 1))
        
#         if k < N:
#             A[k-1, k] = (beta * (k + 1) * (-alpha * k * (N - k) + N * (N - 1))) / (N**2 * (N - 1))
    
#     return A

# def construct_new_tridiagonal_mtx(N,alpha, beta):
#     Q = np.zeros((N,N))
    
#     for k in range(1, N + 1):
#         if k > 1:
#             Q[k-1, k-2] = beta*(k)
        
#         Q[k-1, k-1] = -beta*k-alpha*k*(N-k)/(N-1)
        
#         if k< N:
#             Q[k-1, k] = alpha*k*(N-k)/(N-1)
    
#     print(np.linalg.norm(Q.sum(axis=1)))
    
#     return Q


def construct_new_tridiagonal_mtx(N,alpha):
    Q = np.zeros((N,N))
    
    for k in range(1, N + 1):
        if k > 1:
            Q[k-1, k-2] = k
        
        Q[k-1, k-1] = -k-alpha*k*(N-k)/(N-1)
        
        if k< N:
            Q[k-1, k] = alpha*k*(N-k)/(N-1)
    
    #print(np.linalg.norm(Q.sum(axis=1)))
    
    return Q




def compute_S(N, alpha, beta, mu, tmax):
    A = construct_tridiagonal_matrix(N, alpha, beta)
    f = np.zeros(N)
    mu_and_zeros = np.zeros(N)
    mu_and_zeros[0] = mu
    S = np.zeros(tmax)
    for t in range(1,tmax):
        f = np.dot(A, f) + mu_and_zeros
        S[t] = f.sum()    
    return S



def compute_S_new(N, alpha, mu, tmax):
    q = construct_new_tridiagonal_mtx(N, alpha)  # Q matrix
    eigvals = np.linalg.eigvals(q)
    max_real_part = max(np.real(eigvals))
#    convergence_rate = np.exp(max_real_part)
#    print(f"max eigenvalue = {max(np.real(eigvals))}")
    mu_and_zeros = np.zeros(N)
    mu_and_zeros[0] = mu
    f0 = np.zeros(N)


    def odesys(t, f):
        return q.T @ f + mu_and_zeros[:, None] * N
    t_eval = np.arange(tmax)  # Ensures 100 time points
    sol = solve_ivp(odesys, (0, tmax - 1), f0, t_eval=t_eval, vectorized=True, jac=q.T, method= 'BDF')

    S = np.sum(sol.y, axis=0)  # shape = (tmax,)
    return S, max_real_part
    


def compute_f_new(alpha, N, mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha *k/(k+1)*(N-k)/(N-1)
    f_1 = mu *N
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f

def compute_S_eq(N, alpha, mu):
    f_new =  compute_f_new(alpha, N, mu)
    return f_new.sum()

def compute_loss_rate(N,alpha, beta, mu, tmax):
    for t in range(1,tmax):
        S = compute_S(N, alpha, beta, mu, tmax)
        f_1 = mu *N/beta
        t_prime = (f_1*beta) /S
    return t_prime

    
def plot_S(alpha_values, N, beta, mu, tmax):
    plt.figure(figsize=(8, 5))
    x = np.arange(tmax)

    for alpha in alpha_values:
        S = compute_S(N, alpha, beta, mu, tmax)
        S_eq = compute_S_eq(N, alpha, beta, mu)
        plt.plot(x, S, linestyle='-', label=f'$S$ (α ={alpha: .2f})')
        #plt.plot(x, S_eq * np.ones_like(x), linestyle='--', label=f'$S_{{eq}}$ (α/β={alpha/beta: .2f})')
        
        
    
    #for beta in beta_values:
        #S = compute_S(N, alpha, beta, mu, tmax)
        #S_eq = compute_S_eq(N, alpha, beta, mu)
        #plt.plot(x, S, linestyle='-', label=f'$S$ (α={alpha})')
        #plt.plot(x, S_eq * np.ones_like(x), linestyle='--', label=f'$S_{{eq}}$ (α={alpha})')
        
    
    plt.xlabel('time t')
    plt.ylabel('Expected amount $S$ of culture')
    plt.title('')
    plt.legend()
    plt.grid()
    plt.legend(fontsize='small', labelspacing=0.3, handlelength=1.5, borderpad=0.5)
    plt.show()
    

def plot_S_new(alpha_values, N, beta, mu, tmax):
    plt.figure(figsize=(8, 5))
    

    for alpha in alpha_values:
        S = compute_S_new(N, alpha, beta, mu, tmax)
        x = np.arange(len(S))
        #S_eq = compute_S_eq(N, alpha, beta, mu)
        plt.plot(x, S, linestyle='-', label=f'$S$ (α ={alpha: .2f})')
        #plt.plot(x, S_eq * np.ones_like(x), linestyle='--', label=f'$S_{{eq}}$ (α/β={alpha: .2f})')
    
    plt.xlabel('time t')
    plt.ylabel('Expected amount $S$ of culture')
    plt.title('')
    plt.legend()
    plt.grid()
    plt.legend(fontsize='small', labelspacing=0.3, handlelength=1.5, borderpad=0.5)
    plt.savefig('plotalpha1.eps', format= 'eps')
    plt.show()

def plot_loss_rate(alpha, N, beta, mu, tmax):
    plt.figure(figsize=(8, 5))
    x = np.arange(tmax)
    t_primes = compute_loss_rate(N, alpha, beta, mu, tmax)
    plt.plot(x, t_primes, linestyle='-', label='loss rate')
    
    plt.xlabel('time t')
    plt.ylabel('Loss rate')
    plt.title()
    plt.legend()
    plt.grid()
    plt.show()
    
def time_to_equilibrium(S, S_eq, tol): # Original tolerance tol=1e-2
    """
    Find the first time index where S is within tol of equilibrium.
    Returns len(S)-1 if never reached.
    """
    print(f"S_eq = {S_eq}")
    for t, val in enumerate(S):
        if t % 1000 == 0:
            print(f"   {val}")
            print(f"   abs = {abs(val - S_eq)}")
            print(f"   tol * S_eq = {tol * S_eq}")
        if abs(val - S_eq) <= tol * S_eq:
            return t
    return len(S) - 1


def plot_time_to_equilibrium(N, mu, tmax, alpha_min=0, alpha_max=2, n_points=50, tol=np.exp(-3)):
    """
    Plot time to equilibrium vs alpha.
    """
    alpha_values = np.linspace(alpha_min, alpha_max, n_points)
    times = []
    times2 = []

    for alpha in alpha_values:
        # compute trajectory and equilibrium
        S, max_real_part = compute_S_new(N, alpha, mu, tmax)
        S_eq = compute_S_eq(N, alpha, mu)
        t_eq = time_to_equilibrium(S, S_eq, tol=tol)
        times.append(t_eq)
        times2.append(-3/max_real_part)

    # plot
    plt.figure(figsize=(8, 5))
    #plt.plot(alpha_values, np.array(times)/np.array(times2), "--", label="time to equilibrium")
    plt.plot(alpha_values, times2, "o-", label="time to equilibrium")
    plt.xlabel(r"social earning efficiency $\alpha$")
    plt.ylabel("time to equilibrium")
    plt.title("Time to reach equilibrium vs α")
    #plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_time_to_equilibrium(N, mu, tmax)
    

#plot_S(alpha_values, N, beta, mu, tmax)
#plot_loss_rate(alpha, N, beta, mu, tmax)
#plot_S_new(alpha_values,N, beta, mu , tmax)



