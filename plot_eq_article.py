import numpy as np
import matplotlib.pyplot as plt
import os

def compute_f(N, alpha, mu):
    alpha = np.asarray(alpha)[..., np.newaxis]
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 = alpha * k/(k+1) * (N-k)/(N-1)
    f_1 = np.full_like(alpha, mu * N)
    f = np.cumprod(np.concatenate((f_1, coeff_from_k_to_kplus1), axis=-1), axis=-1)
    return f

def compute_S(*, N, alpha, mu):
    f =  compute_f(N=N, alpha=alpha, mu=mu)
    return f.sum(axis=-1)

def compute_S_from_N(*, Nvalues, alpha, mu):
    S = np.array([compute_S(N=N, alpha=alpha, mu=mu) for N in Nvalues])
    return S

# Approximation function S tilde

def compute_S_approx_for_alpha_smaller_than_one(*, N, alpha, mu):
    assert np.all(alpha < 1)
    return (mu * N / alpha) * np.log(1 / (1 - alpha))

def compute_S_approx_for_alpha_larger_than_one(*, N, alpha, mu):
    assert np.all(alpha > 1)
    numerator = mu * np.sqrt(2 * np.pi * N)
    denominator = 1 - (1 / alpha)
    base = alpha * np.exp(-(1 - (1 / alpha)))
    return (numerator / denominator) * (base ** (N - 1))

def compute_S_approx(*, N, alpha, mu):
    if np.all(alpha < 1):
        return compute_S_approx_for_alpha_smaller_than_one(N=N, alpha=alpha, mu=mu)
    elif np.all(alpha > 1):
        return compute_S_approx_for_alpha_larger_than_one(N=N, alpha=alpha, mu=mu)
    assert False
    
def compute_trait_per_individual(*, N, alpha, mu):
    f = compute_f(N, alpha, mu)
    return np.sum(f * (1 + np.arange(N)), axis=-1) / N

def compute_R_infty_approx(N, alpha, mu):
    if np.all(alpha < 1):
        return mu / (1 - alpha)
    elif np.all(alpha > 1):
        return (1 - 1 / alpha) * compute_S_approx_for_alpha_larger_than_one(N=N, alpha=alpha, mu=mu)
    assert False

def plot_S(S, S_approx, x, xlabel, s_approx_label):
    plt.figure(figsize=(8, 5))
    plt.plot(x, S, linestyle='-', color='b', label='$S$')
    if S_approx is not None:  
        plt.plot(x, S_approx, linestyle='--', color='r', label=s_approx_label)
    plt.xlabel(xlabel)
    plt.ylabel("$E(S)$ at equilibrium")
    plt.title('')
    plt.legend()
    plt.grid()
    #plt.savefig('plotalphagreaterone.eps', format= 'eps')
    plt.show()
    
    
# def plot_S2(S_hat, tvalues, alpha, beta, mu, N):
#     plt.plot(S_hat, tvalues ,alpha, beta, mu,N)    
#     plt.figure(figsize=(8, 5))
#     plt.plot(tvalues, S_hat, linestyle='-', color='b', label='$S$')  # Added tvalues as x-axis
#     plt.xlabel("t")  # Added x-axis label
#     plt.ylabel("Expected amount $S$ of culture at equilibrium")
#     plt.title('')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
    

    
# def plot_S_test(S,x,xlabel):
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, S, linestyle='-', color='b', label='$S$')
#     plt.xlabel(xlabel)
#     plt.ylabel("Expected amount $S$ of culture at equilibrium")
#     plt.title("S from formula")
#     plt.legend()
#     plt.grid()
#     plt.show()

# def plot_Z(Z,Z_approx,x,xlabel, Z_approx_label):
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, Z, linestyle='-', color='b', label='$Z$')
#     if Z_approx is not None:  
#         plt.plot(x, Z_approx, linestyle='--', color='r', label=Z_approx_label)
#     plt.xlabel(xlabel)
#     plt.ylabel("Expected amount of culture per individual")
#     plt.title('')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
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

        
def _generate_figure_S_vs_N(*, Nmax, alpha, mu):
    Nvalues = np.arange(0, Nmax)
    S = compute_S_from_N(Nvalues=Nvalues, alpha=alpha, mu=mu)
    S_tilde = compute_S_approx(N=Nvalues, alpha=alpha, mu=mu)
    plot_S(S=S, S_approx=S_tilde, x = Nvalues, xlabel = f'population size $N, (\\alpha = {alpha})$, ', s_approx_label= '$\\tilde{S}$')

def generate_figure_S_vs_N_when_alpha_smaller_than_one():
    Nmax = 100
    alpha = 0.3
    mu = 0.1
    _generate_figure_S_vs_N(Nmax=Nmax, alpha=alpha, mu=mu)

def generate_figure_S_vs_N_when_alpha_larger_than_one():
    Nmax = 100
    alpha = 1.5
    mu = 0.1
    _generate_figure_S_vs_N(Nmax=Nmax, alpha=alpha, mu=mu)
    
def generate_figure_popularity_distribution_alpha_greater_than_beta():
    alpha = 2
    beta = 1
    mu = 0.1
    N =100
    f = compute_f(N=N, alpha=alpha, mu=mu)
    k_top= (1- beta/alpha)*(N-1)+1
    height_top = 1/k_top *mu*N/beta * (alpha/(np.exp(1)*beta) * np.exp(beta/alpha))**(N-1) * np.sqrt(alpha/beta) 
    width_top = np.sqrt((N-1)*beta/alpha)
    plot_f(f, k_top , width_top,height_top)

def generate_figure_popularity_distribution_alpha_less_than_beta():
    alpha = 2
    beta = 1
    mu = 0.1
    N =100
    f = compute_f(N=N, alpha=alpha, mu=mu)
    k = np.arange(1,N+1)
    f_approx = mu* N / (k * beta) * (alpha/beta)**(k-1)
    plot_f_with_approx(f, f_approx)
    

# def generate_trait_per_individual_when_alpha_greater_than_beta():
#     beta = 1
#     mu = 0.1
#     N =100
#     alphavalues = np.arange(1.01, 1.5, 0.01)
#     R_infty = compute_trait_per_individual_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
#     R_infty_hat = compute_R_infty_hat(alpha=alphavalues, beta=beta, N=N, mu=mu)
#     plot_R(R = R_infty, R_approx = R_infty_hat, x=alphavalues, xlabel = '$\\alpha/\\beta$', s_approx_label='$\\hat{R}$')    
    
    
# def generate_trait_per_individual_when_alpha_less_than_beta():
#     beta = 1
#     mu = 0.1
#     N =100
#     alphavalues = np.arange(0.01, 1, 0.01)
#     R_infty = compute_trait_per_individual_alpha(alphavalues = alphavalues, beta=beta, N=N,mu=mu)
#     R_infty_hat = compute_R_infty_hat_alpha_less_beta(alpha =alphavalues, beta=beta, N=N, mu=mu)
#     plot_R(R = R_infty, R_approx = R_infty_hat, x=alphavalues, xlabel = '$\\alpha/\\beta$', s_approx_label='$\\hat{R}$')
    

def generate_trait_and_alpha_plots():
    mu = 0.1
    beta = 1
    N_large = 1000
    N_mid = 100
    alphavalues = np.linspace(0.01, 1.5, 200)

    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1

    # ===== Row 1: Trait per individual =====
    S_trait = compute_S(N=N_large, alpha=alphavalues, mu=mu)
    R_infty = compute_trait_per_individual(N=N_large, alpha=alphavalues, mu=mu)
    R_over_S_true = R_infty / S_trait

    R_over_S_approx_less = 1 / (
        N_large * (1 / alphavalues[mask_less_than_1] - 1) *
        np.log(1 / (1 - alphavalues[mask_less_than_1]))
    )
    R_over_S_approx_greater = 1 - 1 / alphavalues[mask_greater_than_1]

    # ===== Row 2: S vs alpha =====
    S_alpha = compute_S(N=N_mid, alpha=alphavalues, mu=mu)
    S_tilde_less = compute_S_approx(
        N=N_mid, alpha=alphavalues[mask_less_than_1], mu=mu
    )
    S_tilde_greater = compute_S_approx(
        N=N_mid, alpha=alphavalues[mask_greater_than_1], mu=mu
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
    #plt.show()


def generate_side_by_side_S_and_R_over_S_vs_alpha():
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division by zero
    

    # true S
    S = compute_S(N=N, alpha=alphavalues, mu=mu)
    
    # masks for splitting domain
    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1
    
    # approximations
    S_tilde_less = compute_S_approx(
        N=N, alpha=alphavalues[mask_less_than_1], mu=mu
    )
    S_tilde_greater = compute_S_approx(
        N=N, alpha=alphavalues[mask_greater_than_1], mu=mu
    )

    # true values
    S = compute_S(N=N, alpha=alphavalues, mu=mu)
    R_infty = compute_trait_per_individual(N=N, alpha=alphavalues, mu=mu)
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
#    plt.show()





def main():
    #generate_figure_popularity_distribution_alpha_greater_than_beta()
    generate_figure_popularity_distribution_alpha_less_than_beta()
    ###generate_figure_S_vs_N_when_alpha_smaller_than_one()
    ###generate_figure_S_vs_N_when_alpha_larger_than_one()
    ###generate_side_by_side_S_and_R_over_S_vs_alpha() # Fin figur
    ###generate_trait_and_alpha_plots() # Likadan som den fina figuren men lite mindre fin
    plt.show()
    return    
    
if __name__ == '__main__':
    main()


