import numpy as np
import matplotlib.pyplot as plt


def compute_S(alpha, beta, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * k * (N - k) * (N - beta * k) / (beta * (k + 1) * (N * (N - 1) - alpha * k * (N - k)))
    f_1 = mu *N/beta
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    S = f.sum()
    return S

def compute_new_S(alpha, beta, N, mu):
    f_new =  compute_f_new(alpha, beta, N, mu)
    return f_new.sum()

def compute_f(alpha, beta, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * k * (N - k) * (N - beta * k) / (beta * (k + 1) * (N * (N - 1) - alpha * k * (N - k)))
    f_1 = mu *N/beta
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f

def compute_f_new(alpha, beta, N, mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha/beta *k/(k+1)*(N-k)/(N-1)
    f_1 = mu *N/beta
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f

def compute_f_old(alpha, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * (N - k) *  k / ( (N - 1- alpha*k) * (k+1))
    f_1 = mu *N
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f


def compute_S_from_N(*,alpha,beta,mu,Nvalues):
    S = np.array([compute_S(alpha, beta, N, mu) for N in Nvalues])
    return S


def compute_S_from_alpha(*, alphavalues, beta, mu, N):
    S = np.array([compute_new_S(alpha, beta, N, mu) for alpha in alphavalues])
    return S

def compute_S_from_beta(*,betavalues,alpha, mu,N):
    S = np.array([compute_S(alpha, beta, N, mu) for beta in betavalues])
    return S


# Approximation function S tilde

def compute_S_tilde_from_N(*,alpha,beta,mu,Nvalues):
    S_tilde =  (mu * Nvalues / alpha) * np.log(1 / (1 - alpha / beta)) 
    return S_tilde

def compute_S_tilde_from_N_alpha_greater_than_one(*,alpha,mu,Nvalues):
    numerator = mu * np.sqrt(2 * np.pi * Nvalues)
    denominator = 1 - (1 / alpha)
    base = alpha * np.exp(-(1 - (1 / alpha)))
    S_tilde =  (numerator / denominator) * (base ** (Nvalues - 1))
    return S_tilde
    
    #make s tilde for alpha2 from prop2
def compute_S_tilde_from_alpha(*,alphavalues,beta,mu,N):
    S_tilde =  (mu * N / alphavalues) * np.log(1 / (1 - alphavalues / beta)) 
    return S_tilde
    
def compute_S_hat_old(alpha, beta,N,mu):
    # Generate the values of i from 1 to N-1
    i_values = np.arange(1, N)
    
    # Compute the product term
    numerators = N - beta * i_values
    denominators = (N * (N - 1)) / (N - i_values) - alpha * i_values
    product_term = np.prod(numerators / denominators)
    
    # Compute the final value of S_hat
    S_hat = mu* (alpha**(N - 1)) / (beta**N) * product_term
    return S_hat   

def compute_S_hat(alpha, beta,N,mu):
    first_factor = mu * np.sqrt(2*np.pi*N) / (beta * (1 - beta/alpha))
    second_factor = np.power(alpha/beta * np.exp(beta/alpha - 1), N-1)
    S_hat = first_factor * second_factor
    return S_hat
    
def compute_trait_per_individual(*,alpha, beta, N, mu):
    f = compute_f_new(alpha, beta, N, mu)
    return np.sum(f * (1 + np.arange(N))) / N

def compute_trait_per_individual_from_alpha(*, alphavalues, beta, mu, N):
    R = np.array([compute_trait_per_individual(alpha=alpha, beta=beta, N=N, mu=mu) for alpha in alphavalues])
    return R


def compute_R_infty_hat(alpha, beta, N,mu):
    first_factor = mu * np.sqrt(2*np.pi*N) / beta
    second_factor = np.power(alpha/beta * np.exp(beta/alpha - 1), N-1)
    R_infty_hat = first_factor * second_factor
    return R_infty_hat

def compute_R_infty_hat_alpha_less_beta(alpha, beta, N,mu):
    first_factor = mu 
    second_factor = 1/(beta - alpha)
    R_infty_hat = first_factor * second_factor
    return R_infty_hat

def compute_S_hat_from_alpha(*, alphavalues, beta, mu, N):
    S_hat = np.array([compute_S_hat(alpha, beta, N, mu) for alpha in alphavalues])
    return S_hat

def compute_S_hat_from_beta(*, alpha, betavalues, mu, N):
    S_hat = np.array([compute_S_hat(alpha, beta, N, mu) for beta in betavalues])
    return S_hat
    

def plot_S(S,S_approx,x,xlabel, s_approx_label):
    plt.figure(figsize=(8, 5))
    plt.plot(x, S, linestyle='-', color='b', label='$E(S)$')
    if S_approx is not None:  
        plt.plot(x, S_approx, linestyle='--', color='r', label=s_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel("Expected amount of culture at equilibrium")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.show()
    
    
def plot_S2(S_hat, tvalues, alpha, beta, mu, N):
    plt.plot(S_hat, tvalues ,alpha, beta, mu,N)    
    plt.figure(figsize=(8, 5))
    plt.plot(tvalues, S_hat, linestyle='-', color='b', label='$S$')  # Added tvalues as x-axis
    plt.xlabel("t")  # Added x-axis label
    plt.ylabel("Expected amount $S$ of culture at equilibrium")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.show()
    
    

    
def plot_S_test(S,x,xlabel):
    plt.figure(figsize=(8, 5))
    plt.plot(x, S, linestyle='-', color='b', label='$S$')
    plt.xlabel(xlabel)
    plt.ylabel("Expected amount $S$ of culture at equilibrium")
    plt.title("S from formula")
    plt.legend()
    plt.grid()
    plt.show()

def plot_Z(Z,Z_approx,x,xlabel, Z_approx_label):
    plt.figure(figsize=(8, 5))
    plt.plot(x, Z, linestyle='-', color='b', label='$Z$')
    if Z_approx is not None:  
        plt.plot(x, Z_approx, linestyle='--', color='r', label=Z_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel("Expected amount of culture per individual")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_f(f, k_top,width_top, height_top):
    k = np.arange(1,len(f)+1)
    delta_k = k - k_top
    v = delta_k / width_top
    
    plt.figure(figsize=(8, 5))
    plt.plot(k, f, linestyle='-', color='b', label='$f$')
    #plt.plot([k_top,k_top], [0,np.max(f)], linestyle='-', color='r', label='top')
    #plt.plot(k, height_top*np.exp(-v*v/2),linestyle='-', color='g', label='gauss' )
    plt.ylabel(" $f$ of culture at equilibrium")
    plt.xlabel("Trait Popularity $k$")
    plt.title("f from CAM model")
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_f_with_approx(f, f_approx):
    k = np.arange(1,len(f)+1)    
    plt.figure(figsize=(8, 5))
    plt.plot(k, f, linestyle='-', color='b', label='$f$')
    plt.plot(k, f_approx, linestyle='-', color='g', label='f_approx' )
    plt.ylabel(" $f$ of culture at equilibrium")
    plt.title("f from formula")
    plt.legend()
    plt.grid()
    plt.show()

def plot_R(R,R_approx,x,xlabel, s_approx_label):
    plt.figure(figsize=(8, 5))
    plt.plot(x, R, linestyle='-', color='b', label='$E(R)/E(S)$')
    if R_approx is not None:  
        plt.plot(x, R_approx, linestyle='--', color='r', label=s_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel("Proportion of expected total culture carried by each agent")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.savefig('plotRalphalessthanbeta1.png', dpi=500)
    plt.show()

    
def generate_figure_S_and_approx_when_alpha_less_than_beta():
    beta = 1
    mu = 0.1
    N =100
    alphavalues = np.arange(0, 1, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu=mu, N=N)
    S_tilde = compute_S_tilde_from_alpha(alphavalues=alphavalues, beta= beta, mu=mu, N=N)
    plot_S(S=S,S_approx = S_tilde,x=alphavalues,xlabel = '$ \\alpha $',s_approx_label= 'approximation') 

def generate_figure_S_and_approx_when_alpha_greater_than_beta():
    beta = 1
    mu = 0.1
    N =100
    alphavalues = np.arange(1.01, 1.5, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu=mu, N=N)
    S_hat = compute_S_hat(alpha=alphavalues, beta= beta, mu=mu, N=N)
    plot_S(S=S,S_approx = S_hat,x=alphavalues,xlabel = 'alpha/beta $ alpha/beta $',s_approx_label=None) 
    
def generate_figure_popularity_distribution_alpha_greater_than_beta():
    alpha = 0.999
    beta = 0.001
    mu = 0.1
    N =100
    f_old = compute_f_old(alpha=alpha, N=N, mu=mu)
    #f_new = compute_f_new(alpha=alpha, beta = beta, N=N, mu=mu)
    #f = compute_f(alpha=alpha, beta=beta, N=N, mu=mu)
    k_top= (1- beta/alpha)*(N-1)+1
    height_top = 1/k_top *mu*N/beta * (alpha/(np.exp(1)*beta) * np.exp(beta/alpha))**(N-1) * np.sqrt(alpha/beta) 
    width_top = np.sqrt((N-1)*beta/alpha)
    plot_f(f_old, k_top , width_top,height_top)

def generate_figure_popularity_distribution_alpha_less_than_beta():
    alpha = 0.8
    beta = 1
    mu = 0.1
    N =100
    f_new = compute_f_new(alpha=alpha, beta = beta, N=N, mu=mu)
    k = np.arange(1,N+1)
    f_approx = mu* N / (k * beta) * (alpha/beta)**(k-1)
    plot_f_with_approx(f_new, f_approx)


def generate_trait_per_individual_when_alpha_greater_than_beta():
    beta = 1
    mu = 0.1
    N =1000
    alphavalues = np.arange(1.01, 1.5, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu= mu, N=N)
    R_infty = compute_trait_per_individual_from_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
    R_over_S_approx = 1 - 1/alphavalues
    plot_R(R = R_infty/S, R_approx = R_over_S_approx, x=alphavalues, xlabel = '$\\alpha$', s_approx_label='approximation')    
    
    
def generate_trait_per_individual_when_alpha_less_than_beta():
    beta = 1
    mu = 0.1
    N =1000
    alphavalues = np.arange(0.01, 1, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu= mu, N=N)
    R_infty = compute_trait_per_individual_from_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
    R_over_S_approx = 1/(N * (1/alphavalues - 1) * np.log(1/(1-alphavalues)))
    plot_R(R = R_infty/S, R_approx = R_over_S_approx, x=alphavalues, xlabel = '$\\alpha$', s_approx_label='approximation')    

def generate_combined_trait_per_individual():
    beta = 1
    mu = 0.1
    N = 1000
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division by zero

    # true values
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu=mu, N=N)
    R_infty = compute_trait_per_individual_from_alpha(alphavalues=alphavalues, beta=beta, N=N, mu=mu)
    R_over_S_true = R_infty / S

    # approximations
    mask_less = alphavalues < 1
    mask_greater = alphavalues > 1

    R_over_S_approx_less = 1 / (
        N * (1 / alphavalues[mask_less] - 1) * np.log(1 / (1 - alphavalues[mask_less]))
    )
    R_over_S_approx_greater = 1 - 1 / alphavalues[mask_greater]

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(alphavalues, R_over_S_true, label=r"$R_\infty / S$ (true)", color="black", linewidth=2)
    plt.plot(
        alphavalues[mask_less],
        R_over_S_approx_less,
        "--",
        label=r"approx ($\alpha < 1$)",
        color="blue",
    )
    plt.plot(
        alphavalues[mask_greater],
        R_over_S_approx_greater,
        "--",
        label=r"approx ($\alpha > 1$)",
        color="red",
    )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$R_\infty / S$")
    plt.title(r"Trait per individual: true vs approximations")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def generate_combined_figure():
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division issues

    # true S
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)

    # approximations
    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1

    S_tilde_less = compute_S_tilde_from_alpha(
        alphavalues=alphavalues[mask_less_than_1], mu=mu, N=N
    )
    S_tilde_greater = compute_S_tilde_from_N_alpha_greater_than_one(
        alpha=alphavalues[mask_greater_than_1], mu=mu, Nvalues=N
    )

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(alphavalues, S, label="$S$", color="black", linewidth=2)
    plt.plot(
        alphavalues[mask_less_than_1],
        S_tilde_less,
        "--",
        label="$\\tilde{S}$ (for $\\alpha<1$)",
        color="blue",
    )
    plt.plot(
        alphavalues[mask_greater_than_1],
        S_tilde_greater,
        "--",
        label="$\\tilde{S}$ (for $\\alpha>1$)",
        color="red",
    )

    plt.xlabel("social earning efficiency $\\alpha$")
    plt.ylabel("$S$")
    plt.title("True $S$ and approximations $\\tilde{S}$")
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    #generate_figure_S_and_approx_when_alpha_less_than_beta()
    #generate_trait_per_individual_when_alpha_greater_than_beta()
    generate_combined_trait_per_individual()
    return
    alpha = 1.7
    beta = 1
    mu = 0.1
    N =100
    N_test_values = np.arange(0,100)
    Nvalues = np.arange(2,101)
    tvalues = np.arange(0,10000)
    #R = mu / (1-(alpha/beta))
    #alphavalues, betavalues
    alphavalues = np.arange(1.01, 1.5, 0.01)
    #alphavalues = np.arange(0.0, 0.15, 0.01)
    betavalues = np.arange(0.0, 1.1, 0.001)  
    #S = compute_S(alpha=alpha, beta=beta, N=N, mu=mu)
    S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu=mu, N=N)
    #Z = R/S
    #Z_approx = compute_trait_per_individual_alpha(alphavalues=alphavalues, beta=beta, N=N)
    #S = compute_S_from_beta(alpha=alpha, betavalues=betavalues, mu=mu, N=N)
#    f = compute_f(alpha = alpha, beta = beta, N = N, mu = mu)
    #f = compute_f_old(alpha = alpha, N = N, mu = mu)
    f_new = compute_f_new(alpha=alpha, beta = beta, N=N, mu=mu)
    #S = compute_S_from_N(alpha=alpha, beta= beta, mu=mu, Nvalues=N_test_values)
#    print(f_new.sum())
    #S_tilde = compute_S_tilde_from_N(alpha=alpha, beta= beta, mu=mu, Nvalues=N_test_values)
    #S_hat = compute_S_hat_from_alpha(alphavalues=alphavalues, beta= beta, mu=mu, N=N)
    #S_hat = compute_S_hat_from_beta(alpha=alpha, betavalues= betavalues, mu=mu, N=N)

    k_top= (1- beta/alpha)*(N-1)+1
    height_top = 1/k_top *mu*N/beta * (alpha/(np.exp(1)*beta) * np.exp(beta/alpha))**(N-1) * np.sqrt(alpha/beta) 
    width_top = np.sqrt((N-1)*beta/alpha)
    plot_f(f_new, k_top, width_top,height_top)
 
    #plot_Z(Z, Z_approx, x=alphavalues, xlabel= 'learning efficiency $\\alpha $ ', Z_approx_label= 'Z approx')
#    S_tilde = compute_S_tilde_from_alpha(alphavalues=alphavalues, beta= beta, mu=mu, N=N)
#    S_hat = compute_S_hat(alpha=alphavalues, beta= beta, mu=mu, N=N)
#    plot_S(S=S,S_approx = S_hat,x=alphavalues,xlabel = 'alpha/beta $ alpha/beta $',s_approx_label=None) 
    #plot_S(S=np.log(S)-np.log(S_hat), S_approx=None, x=betavalues,xlabel='$\\beta$', s_approx_label = '$\\hat{S}$')
    #plot_S_test(S=S, x=N_test_values,xlabel=' Population size $N$')
    #plot_S2(S_hat, tvalues, alpha, beta, mu, N)
    
    
if __name__ == '__main__':
    main()


