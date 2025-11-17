import numpy as np
import matplotlib.pyplot as plt
import os


def compute_S(alpha, beta, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * k * (N - k) * (N - beta * k) / (beta * (k + 1) * (N * (N - 1) - alpha * k * (N - k)))
    f_1 = mu *N/beta
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    S = f.sum()
    return S

def compute_new_S(alpha, N, mu):
    f_new =  compute_f_new(alpha, N, mu)
    return f_new.sum()

def compute_f(alpha, beta, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * k * (N - k) * (N - beta * k) / (beta * (k + 1) * (N * (N - 1) - alpha * k * (N - k)))
    f_1 = mu *N/beta
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f

def compute_f_new(alpha, N, mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha *k/(k+1)*(N-k)/(N-1)
    f_1 = mu *N
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f

def compute_f_old(alpha, N,mu):
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 =   alpha * (N - k) *  k / ( (N - 1- alpha*k) * (k+1))
    f_1 = mu *N
    f = np.cumprod(np.concatenate(([f_1], coeff_from_k_to_kplus1)))
    return f


def compute_S_from_N(*,alpha,mu,Nvalues):
    S = np.array([compute_new_S(alpha, N, mu) for N in Nvalues])
    return S


def compute_S_from_alpha(*, alphavalues, mu, N):
    S = np.array([compute_new_S(alpha, N, mu) for alpha in alphavalues])
    return S

def compute_S_from_beta(*,betavalues,alpha, mu,N):
    S = np.array([compute_S(alpha, beta, N, mu) for beta in betavalues])
    return S


# Approximation function S tilde

def compute_S_tilde_from_N(*,alpha,mu,Nvalues):
    S_tilde =  (mu * Nvalues / alpha) * np.log(1 / (1 - alpha)) 
    return S_tilde

def compute_S_tilde_from_N_alpha_greater_than_one(*,alpha,mu,Nvalues):
    numerator = mu * np.sqrt(2 * np.pi * Nvalues)
    denominator = 1 - (1 / alpha)
    base = alpha * np.exp(-(1 - (1 / alpha)))
    S_tilde =  (numerator / denominator) * (base ** (Nvalues - 1))
    return S_tilde
    
    #make s tilde for alpha2 from prop2
def compute_S_tilde_from_alpha(*,alphavalues,mu,N):
    S_tilde =  (mu * N / alphavalues) * np.log(1 / (1 - alphavalues)) 
    return S_tilde

def compute_S_tilde_from_alpha_greater_than_one(*, alphavalues, mu, N):
    numerator = mu * np.sqrt(2 * np.pi * N)
    denominator = 1 - (1 / alphavalues)
    base = alphavalues * np.exp(-(1 - (1 / alphavalues)))
    S_tilde =  (numerator / denominator) * (base ** (N - 1))
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
    
def compute_trait_per_individual(*,alpha, N, mu):
    f = compute_f_new(alpha, N, mu)
    return np.sum(f * (1 + np.arange(N))) / N

def compute_trait_per_individual_alpha(*, alphavalues, N,mu):
    R_infty = np.array([compute_trait_per_individual(alpha=alpha, N=N,mu=mu) for alpha in alphavalues])
    return R_infty

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
    plt.plot(x, S, linestyle='-', color='b', label='$S$')
    if S_approx is not None:  
        plt.plot(x, S_approx, linestyle='--', color='r', label=s_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel(" $E(S)$ at equilibrium")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.savefig('plotalphagreaterone.eps', format= 'eps')
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
    
def plot_f(f, k_top, width_top, height_top):
    k = np.arange(1, len(f) + 1)
    delta_k = k - k_top
    v = delta_k / width_top

    plt.figure(figsize=(8, 5))
    plt.plot(k, f, linestyle='-', color='b', label='$f$')
    plt.plot([k_top, k_top], [0, np.max(f)], linestyle='-', color='r', label='top')
    plt.plot(k, height_top * np.exp(-v * v / 2), linestyle='-', color='g', label='gauss')
    plt.ylabel("$f$ of culture at equilibrium")
    plt.xlabel("Trait Popularity $k$")
    plt.title("f from CAM model")
    plt.legend()
    plt.grid()

    # ✅ Save the plot as a PDF in the user's Downloads folder
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "f_from_CAM_model.pdf")
    plt.savefig(downloads_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {downloads_path}")

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
    plt.plot(x, R, linestyle='-', color='b', label='$S$')
    if R_approx is not None:  
        plt.plot(x, R_approx, linestyle='--', color='r', label=s_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel("Expected amount $S$ of culture at equilibrium")
    plt.title('')
    plt.legend()
    plt.grid()
    plt.savefig('plotRalphalessthanbeta1.png', dpi=500)
    plt.show()

    
def generate_figure_S_and_approx_when_alpha_less_than_beta():
    mu = 0.1
    N =100
    alphavalues = np.arange(1, 1.5, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    S_tilde = compute_S_tilde_from_alpha_greater_than_one(alphavalues=alphavalues, mu=mu, N=N)
    plot_S(S=S,S_approx = S_tilde,x=alphavalues,xlabel = 'social earning efficiency $\\alpha $',s_approx_label= '$\\tilde{S}$') 

def generate_figure_S_and_approx_when_alpha_greater_than_beta():
    mu = 0.1
    N =100
    alphavalues = np.arange(0.01, 1, 0.01)
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    #S_hat = compute_S_hat(alpha=alphavalues, beta= beta, mu=mu, N=N)
    S_tilde = compute_S_tilde_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    plot_S(S=S,S_approx = S_tilde,x=alphavalues,xlabel = '$\\alpha/\\beta$',s_approx_label= '$\\tilde{S}$') 
    

def generate_figure_S_based_on_population_when_alpha_smaller_than_one():
    alpha = 0.3
    mu = 0.1
    Nvalues = np.arange(0,100)
    S = compute_S_from_N(alpha=alpha, mu=mu, Nvalues=Nvalues)
    S_tilde = compute_S_tilde_from_N(alpha=alpha, mu=mu, Nvalues=Nvalues)
    plot_S(S=S, S_approx=S_tilde, x = Nvalues, xlabel = 'population size $N, (\\alpha < 1)$, ', s_approx_label= '$\\tilde{S}$')


def generate_figure_S_based_on_population_when_alpha_larger_than_one():
    alpha = 1.5
    mu = 0.1
    Nvalues = np.arange(0,100)
    S = compute_S_from_N(alpha=alpha, mu=mu, Nvalues=Nvalues)
    S_tilde = compute_S_tilde_from_N_alpha_greater_than_one(alpha=alpha, mu=mu, Nvalues=Nvalues)
    plot_S(S=S, S_approx=S_tilde, x = Nvalues, xlabel = 'population size $N, (\\alpha > 1)$, ', s_approx_label= '$\\tilde{S}$')

    
def generate_figure_popularity_distribution_alpha_greater_than_beta():
    alpha = 2
    beta = 1
    mu = 0.1
    N =100
    #f_old = compute_f_old(alpha=alpha, N=N, mu=mu)
    f_new = compute_f_new(alpha=alpha, N=N, mu=mu)
    #f = compute_f(alpha=alpha, beta=beta, N=N, mu=mu)
    k_top= (1- beta/alpha)*(N-1)+1
    height_top = 1/k_top *mu*N/beta * (alpha/(np.exp(1)*beta) * np.exp(beta/alpha))**(N-1) * np.sqrt(alpha/beta) 
    width_top = np.sqrt((N-1)*beta/alpha)
    plot_f(f_new, k_top , width_top,height_top)

def generate_figure_popularity_distribution_alpha_less_than_beta():
    alpha = 2
    beta = 1
    mu = 0.1
    N =100
    f_new = compute_f_new(alpha=alpha, N=N, mu=mu)
    k = np.arange(1,N+1)
    f_approx = mu* N / (k * beta) * (alpha/beta)**(k-1)
    plot_f_with_approx(f_new, f_approx)
    

def generate_trait_per_individual_when_alpha_greater_than_beta():
    beta = 1
    mu = 0.1
    N =100
    alphavalues = np.arange(1.01, 1.5, 0.01)
    R_infty = compute_trait_per_individual_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
    R_infty_hat = compute_R_infty_hat(alpha=alphavalues, beta=beta, N=N, mu=mu)
    plot_R(R = R_infty, R_approx = R_infty_hat, x=alphavalues, xlabel = '$\\alpha/\\beta$', s_approx_label='$\\hat{R}$')    
    
    
def generate_trait_per_individual_when_alpha_less_than_beta():
    beta = 1
    mu = 0.1
    N =100
    alphavalues = np.arange(0.01, 1, 0.01)
    R_infty = compute_trait_per_individual_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
    R_infty_hat = compute_R_infty_hat_alpha_less_beta(alpha =alphavalues, beta=beta, N=N, mu=mu)
    plot_R(R = R_infty, R_approx = R_infty_hat, x=alphavalues, xlabel = '$\\alpha/\\beta$', s_approx_label='$\\hat{R}$')
    
def generate_side_by_side_figures():
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 1.5, 200)  # avoid 0 to prevent division issues
    
    
    # true S
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    
    # masks for splitting domain
    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1
    
    # approximations
    S_tilde_less = compute_S_tilde_from_alpha(
        alphavalues=alphavalues[mask_less_than_1], mu=mu, N=N
    )
    S_tilde_greater = compute_S_tilde_from_N_alpha_greater_than_one(
        alpha=alphavalues[mask_greater_than_1], mu=mu, Nvalues=N
    )
    
    # create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: α in (0,1)
    axes[0].plot(
        alphavalues[mask_less_than_1],
        S[mask_less_than_1],
        label="$S$",
        color="black",
        linewidth=2,
    )
    axes[0].plot(
        alphavalues[mask_less_than_1],
        S_tilde_less,
        "--",
        label="$\\tilde{S}$ (for $\\alpha<1$)",
        color="blue",
    )
    axes[0].set_xlabel("social earning efficiency $\\alpha$")
    axes[0].set_ylabel("$S$")
    axes[0].set_title("True $S$ and $\\tilde{S}$ for $\\alpha<1$")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Right plot: α in (1,2)
    axes[1].plot(
        alphavalues[mask_greater_than_1],
        S[mask_greater_than_1],
        label="$S$",
        color="black",
        linewidth=2,
    )
    axes[1].plot(
        alphavalues[mask_greater_than_1],
        S_tilde_greater,
        "--",
        label="$\\tilde{S}$ (for $\\alpha>1$)",
        color="red",
    )
    axes[1].set_xlabel("social earning efficiency $\\alpha$")
    axes[1].set_ylabel("$E(S)$")
    axes[1].set_title("True $E(S)$ and $\\tilde{S}$ for $\\alpha>1$")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


    


def generate_side_by_side_trait_per_individual():
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division by zero

    # true values
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    R_infty = compute_trait_per_individual_alpha(alphavalues=alphavalues, N=N, mu=mu)
    R_over_S_true = R_infty / S
    
    # masks
    mask_less = alphavalues < 1
    mask_greater = alphavalues > 1
    
    # approximations
    R_over_S_approx_less = 1 / (
        N * (1 / alphavalues[mask_less] - 1) * np.log(1 / (1 - alphavalues[mask_less]))
    )
    R_over_S_approx_greater = 1 - 1 / alphavalues[mask_greater]
    
    # create two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: α in (0,1)
    axes[0].plot(
        alphavalues[mask_less_than_1],
        R_over_S_true[mask_less_than_1],
        label=r"$R_\infty / S$ (true)",
        color="black",
        linewidth=2,
    )
    axes[0].plot(
        alphavalues[mask_less_than_1],
        R_over_S_approx_less,
        "--",
        label=r"approx ($\alpha < 1$)",
        color="blue",
    )
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$R_\infty / S$")
    axes[0].set_title(r"Trait per individual for $\alpha < 1$")
    axes[0].legend()
    axes[0].grid(True)
    
    # Right panel: α in (1,2)
    axes[1].plot(
        alphavalues[mask_greater],
        R_over_S_true[mask_greater],
        label=r"$R_\infty / S$ (true)",
        color="black",
        linewidth=2,
    )
    axes[1].plot(
        alphavalues[mask_greater],
        R_over_S_approx_greater,
        "--",
        label=r"approx ($\alpha > 1$)",
        color="red",
    )
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$R_\infty / S$")
    axes[1].set_title(r"Trait per individual for $\alpha > 1$")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()

    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "trait_and_alpha_plots.pdf")
    fig.savefig(downloads_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {downloads_path}")

    plt.show()
    
    
def generate_trait_and_alpha_plots():
    mu = 0.1
    beta = 1
    N_large = 1000
    N_mid = 100
    alphavalues = np.linspace(0.01, 1.5, 200)

    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1

    # ===== Row 1: Trait per individual =====
    S_trait = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N_large)
    R_infty = compute_trait_per_individual_alpha(alphavalues=alphavalues, beta=beta, N=N_large, mu=mu)
    R_over_S_true = R_infty / S_trait

    R_over_S_approx_less = 1 / (
        N_large * (1 / alphavalues[mask_less_than_1] - 1) *
        np.log(1 / (1 - alphavalues[mask_less_than_1]))
    )
    R_over_S_approx_greater = 1 - 1 / alphavalues[mask_greater_than_1]

    # ===== Row 2: S vs alpha =====
    S_alpha = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N_mid)
    S_tilde_less = compute_S_tilde_from_alpha(
        alphavalues=alphavalues[mask_less_than_1], mu=mu, N=N_mid
    )
    S_tilde_greater = compute_S_tilde_from_N_alpha_greater_than_one(
        alpha=alphavalues[mask_greater_than_1], mu=mu, Nvalues=N_mid
    )

    # ===== Plotting (2 rows × 2 columns) =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.4)

    # Row 1 left
    axes[0, 0].plot(alphavalues[mask_less_than_1], R_over_S_true[mask_less_than_1], lw=2, label='True R∞/E(S)')
    axes[0, 0].plot(alphavalues[mask_less_than_1], R_over_S_approx_less, '--', label='Approx (α<1)')
    axes[0, 0].set_ylabel('R∞/E(S)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Row 1 right
    axes[0, 1].plot(alphavalues[mask_greater_than_1], R_over_S_true[mask_greater_than_1], lw=2, label='True R∞/E(S)')
    axes[0, 1].plot(alphavalues[mask_greater_than_1], R_over_S_approx_greater, '--', label='Approx (α>1)')
    axes[0, 1].set_ylabel('R∞/E(S)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Row 2 left
    axes[1, 0].plot(alphavalues[mask_less_than_1], S_alpha[mask_less_than_1], lw=2, label='True E(S)')
    axes[1, 0].plot(alphavalues[mask_less_than_1], S_tilde_less, '--', label='S~ (α<1)')
    axes[1, 0].set_xlabel('α')
    axes[1, 0].set_ylabel('E(S)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Row 2 right
    axes[1, 1].plot(alphavalues[mask_greater_than_1], S_alpha[mask_greater_than_1], lw=2, label='True E(S)')
    axes[1, 1].plot(alphavalues[mask_greater_than_1], S_tilde_greater, '--', label='S~ (α>1)')
    axes[1, 1].set_ylabel('E(S)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "combinedalphaplot.pdf")
    fig.savefig(downloads_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {downloads_path}")
    plt.show()

def generate_population_plot():
    mu = 0.1
    N_small = np.arange(0, 100)

    alpha_less = 0.3
    alpha_greater = 1.5

    S_N_less = compute_S_from_N(alpha=alpha_less, mu=mu, Nvalues=N_small)
    S_tilde_N_less = compute_S_tilde_from_N(alpha=alpha_less, mu=mu, Nvalues=N_small)

    S_N_greater = compute_S_from_N(alpha=alpha_greater, mu=mu, Nvalues=N_small)
    S_tilde_N_greater = compute_S_tilde_from_N_alpha_greater_than_one(alpha=alpha_greater, mu=mu, Nvalues=N_small)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)

    # Left: alpha < 1
    axes[0].plot(N_small, S_N_less, lw=2, label='True E(S)')
    axes[0].plot(N_small, S_tilde_N_less, '--', label='S~ (α=0.3)')
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('E(S)')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # Right: alpha > 1
    axes[1].plot(N_small, S_N_greater, lw=2, label='True E(S)')
    axes[1].plot(N_small, S_tilde_N_greater, '--', label='S~ (α=1.5)')
    axes[1].set_xlabel('N')
    axes[1].set_ylabel('E(S)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "trait_and_alpha_plots.pdf")
    fig.savefig(downloads_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {downloads_path}")


    plt.show()



def generate_side_by_side_S_and_R_over_S_vs_alpha():
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division by zero
    

    # true S
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    
    # masks for splitting domain
    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1
    
    # approximations
    S_tilde_less = compute_S_tilde_from_alpha(
        alphavalues=alphavalues[mask_less_than_1], mu=mu, N=N
    )
    S_tilde_greater = compute_S_tilde_from_N_alpha_greater_than_one(
        alpha=alphavalues[mask_greater_than_1], mu=mu, Nvalues=N
    )

    # true values
    S = compute_S_from_alpha(alphavalues=alphavalues, mu=mu, N=N)
    R_infty = compute_trait_per_individual_alpha(alphavalues=alphavalues, N=N, mu=mu)
    R_over_S_true = R_infty / S
    
    # masks
    mask_less = alphavalues < 1
    mask_greater = alphavalues > 1
    
    # approximations
    R_over_S_approx_less = 1 / (
        N * (1 / alphavalues[mask_less] - 1) * np.log(1 / (1 - alphavalues[mask_less]))
    )
    R_over_S_approx_greater = 1 - 1 / alphavalues[mask_greater]
    
  
  
  
  
    # create two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    


    # Left plot: α in (0,1)
    axes[0,0].plot(
        alphavalues[mask_less],
        S[mask_less],
        label="$E(S)$",
        color="black",
        linewidth=2,
    )
    axes[0,0].plot(
        alphavalues[mask_less],
        S_tilde_less,
        "--",
        label="Approx $E(S)$",
        color="blue",
    )
#    axes[0,0].set_xlabel("social earning efficiency $\\alpha$")
    axes[0,0].set_xticklabels([])
    axes[0,0].set_ylabel("$E(S)$")
    axes[0,0].set_title("True and approximative $E(S)$ for $\\alpha<1$")
    axes[0,0].legend()
    axes[0,0].grid(True)
    axes[0,0].set_yscale('log')
    
    # Right plot: α in (1,2)
    axes[0,1].plot(
        alphavalues[mask_greater_than_1],
        S[mask_greater_than_1],
        label="$E(S)$",
        color="black",
        linewidth=2,
    )
    axes[0,1].plot(
        alphavalues[mask_greater_than_1],
        S_tilde_greater,
        "--",
        label="approx $E(S)$",
        color="blue",
    )
#    axes[0,1].set_xlabel("social earning efficiency $\\alpha$")
    axes[0,1].set_xticklabels([])
    axes[0,1].set_ylabel("$E(S)$")
    axes[0,1].set_title("True and approx $E(S)$")
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].set_yscale('log')



    # Left panel: α in (0,1)
    axes[1,0].plot(
        alphavalues[mask_less],
        R_over_S_true[mask_less],
        label=r"$E(R) / E(S)$ (true)",
        color="black",
        linewidth=2,
    )
    axes[1,0].plot(
        alphavalues[mask_less],
        R_over_S_approx_less,
        "--",
        label=r"$E(R) / E(S)$ (approx)",
        color="blue",
    )
    axes[1,0].set_xlabel(r"$\alpha$")
    axes[1,0].set_ylabel(r"$E(R) / E(S)$")
    axes[1,0].set_title(r"Proportion of traits per individual")
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Right panel: α in (1,2)
    axes[1,1].plot(
        alphavalues[mask_greater],
        R_over_S_true[mask_greater],
        label=r"$E(R) / E(S)$ (true)",
        color="black",
        linewidth=2,
    )
    axes[1,1].plot(
        alphavalues[mask_greater],
        R_over_S_approx_greater,
        "--",
        label=r"approx",
        color="blue",
    )
    axes[1,1].set_xlabel(r"$\alpha$")
    axes[1,1].set_ylabel(r"$E(R) / E(S)$")
    axes[1,1].set_title(r"Proportion of traits per individual")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()





def main():
    generate_figure_S_based_on_population_when_alpha_smaller_than_one()
    return
    #generate_side_by_side_trait_per_individual()
    #return
    generate_side_by_side_S_and_R_over_S_vs_alpha()
    return
    generate_side_by_side_figures()
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
    #S = compute_S_from_alpha(alphavalues=alphavalues, beta=beta, mu=mu, N=N)
    #Z = R/S
    #Z_approx = compute_trait_per_individual_alpha(alphavalues=alphavalues, beta=beta, N=N)
    #S = compute_S_from_beta(alpha=alpha, betavalues=betavalues, mu=mu, N=N)
#    f = compute_f(alpha = alpha, beta = beta, N = N, mu = mu)
    #f = compute_f_old(alpha = alpha, N = N, mu = mu)
    f_new = compute_f_new(alpha=alpha, N=N, mu=mu)
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


