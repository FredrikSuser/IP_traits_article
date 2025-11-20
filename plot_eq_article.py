import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import path

IMAGE_FOLDER = path.Path("generated_images")
EXACT_LABEL = "exact value"
EXACT_COLOR = "black"
APPROXIMATION_LABEL = "approximation"
APPROXIMATION_COLOR = "blue"

def _wrap_funcions_by_case(
        *,
        func_alpha_small,
        func_alpha_large
    ):
    def func(*, N: int | np.ndarray, alpha: float | np.ndarray, mu: float) -> np.ndarray:
        alpha = np.asarray(alpha)
        result_shape = np.broadcast(N, alpha, mu).shape
        mask_alpha_small = alpha < 1
        mask_alpha_large = alpha > 1
        mask_alpha_small_broadcast = np.broadcast_to(mask_alpha_small, result_shape)
        mask_alpha_large_broadcast = np.broadcast_to(mask_alpha_large, result_shape)
        result = np.full(result_shape, np.nan)
        if np.any(mask_alpha_small):
            result[mask_alpha_small_broadcast] = func_alpha_small(N=N, alpha=alpha[mask_alpha_small], mu=mu)
        if np.any(mask_alpha_large):
            result[mask_alpha_large_broadcast] = func_alpha_large(N=N, alpha=alpha[mask_alpha_large], mu=mu)
        return result
    return func

def compute_f(N: int, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    alpha = np.asarray(alpha)[..., np.newaxis]
    k = np.arange(1,N)
    coeff_from_k_to_kplus1 = alpha * k/(k+1) * (N-k)/(N-1)
    f_1 = np.full_like(alpha, mu * N)
    f = np.cumprod(np.concatenate((f_1, coeff_from_k_to_kplus1), axis=-1), axis=-1)
    return f

def compute_f_approx(N: int, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    k = np.arange(1, N + 1)
    if alpha < 1:
        return mu * N * alpha**(k-1) / k
    elif alpha > 1:
        m = (1 - 1 / alpha) * N
        sigma2 = N / alpha
        C = mu * alpha**(3/2) / (alpha - 1) * (alpha * np.exp(-(1 - 1 / alpha))) ** (N - 1)
        return C * np.exp(-(k - m)**2 / (2 * sigma2))
    assert False

def compute_S(*, N: int, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    f =  compute_f(N=N, alpha=alpha, mu=mu)
    return f.sum(axis=-1)

def compute_S_from_N(*, Nvalues: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    S = np.array([compute_S(N=N, alpha=alpha, mu=mu) for N in Nvalues])
    return S

def _compute_S_approx_alpha_smaller_than_one(*, N: int | np.ndarray, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    return (mu * N / alpha) * np.log(1 / (1 - alpha))

def _compute_S_approx_alpha_larger_than_one(*, N: int | np.ndarray, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    numerator = mu * np.sqrt(2 * np.pi * N)
    denominator = 1 - (1 / alpha)
    base = alpha * np.exp(-(1 - (1 / alpha)))
    return (numerator / denominator) * (base ** (N - 1))    

compute_S_approx = _wrap_funcions_by_case(
    func_alpha_small=_compute_S_approx_alpha_smaller_than_one,
    func_alpha_large=_compute_S_approx_alpha_larger_than_one
)
    
def compute_R_over_S(*, N: int, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    f = compute_f(N=N, alpha=alpha, mu=mu)
    R = np.sum(f * (1 + np.arange(N)), axis=-1) / N
    S = compute_S(N=N, alpha=alpha, mu=mu)
    return R / S

def _compute_R_over_S_approx_alpha_smaller_than_one(N: int | np.ndarray, alpha: float | np.ndarray, mu: float) -> np.ndarray:
    return mu / (1 - alpha) / _compute_S_approx_alpha_smaller_than_one(N=N, alpha=alpha, mu=mu) 

def _compute_R_over_S_approx_alpha_larger_than_one(N: int | np.ndarray, alpha: float | np.ndarray, mu: float) -> float | np.ndarray:
    return 1 - 1 / alpha

compute_R_over_S_approx = _wrap_funcions_by_case(
    func_alpha_small=_compute_R_over_S_approx_alpha_smaller_than_one,
    func_alpha_large=_compute_R_over_S_approx_alpha_larger_than_one
)

def save_figure(filename: str) -> None:
    path = IMAGE_FOLDER / filename
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot to: {path}")

def plot_S(S, S_approx, x):
    plt.figure(figsize=(8, 5))
    plt.plot(x, S, label=EXACT_LABEL, color=EXACT_COLOR, linewidth=2)
    if S_approx is not None:
        plt.plot(x, S_approx, linestyle='--', label=APPROXIMATION_LABEL, color=APPROXIMATION_COLOR)
    plt.xlabel('population size $N$')
    plt.ylabel('expected amount of culture $E(S)$')
    plt.title('')
    plt.legend()
    plt.grid()
    plt.tight_layout()
            
def plot_f_with_approx(f, f_approx):
    assert len(f) == len(f_approx)
    k = np.arange(1, len(f) + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    plt.plot(k, f, label=EXACT_LABEL, color=EXACT_COLOR, linewidth=2)
    plt.plot(k, f_approx, linestyle='--', label=APPROXIMATION_LABEL, color=APPROXIMATION_COLOR)
    plt.xlabel("popularity $k$")
    plt.ylabel("expected frequency $E(f_k)$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
 
def _generate_figure_S_vs_N(*, Nmax: int, alpha: float, mu: float) -> None:
    Nvalues = np.arange(0, Nmax)
    S = compute_S_from_N(Nvalues=Nvalues, alpha=alpha, mu=mu)
    S_approx = compute_S_approx(N=Nvalues, alpha=alpha, mu=mu)
    plot_S(S=S, S_approx=S_approx, x=Nvalues)
    save_figure(f"S_vs_N_alpha{alpha}_mu{mu}.pdf")

def generate_figure_S_vs_N_when_alpha_smaller_than_one() -> None:
    Nmax = 100
    alpha = 0.8
    mu = 0.1
    _generate_figure_S_vs_N(Nmax=Nmax, alpha=alpha, mu=mu)

def generate_figure_S_vs_N_when_alpha_larger_than_one() -> None:
    Nmax = 100
    alpha = 1.5
    mu = 0.1
    _generate_figure_S_vs_N(Nmax=Nmax, alpha=alpha, mu=mu)

def _generate_figure_popularity_distribution(*, N: int, alpha: float, mu: float) -> None:
    f = compute_f(N=N, alpha=alpha, mu=mu)
    f_approx = compute_f_approx(N=N, alpha=alpha, mu=mu)
    plot_f_with_approx(f, f_approx)
    save_figure(f"popularity_distribution_N{N}_alpha{alpha}_mu{mu}.pdf")

def generate_figure_popularity_distribution_alpha_larger_than_one() -> None:
    _generate_figure_popularity_distribution(N=100, alpha=2, mu=0.1)

def generate_figure_popularity_distribution_alpha_smaller_than_one() -> None:
    _generate_figure_popularity_distribution(N=100, alpha=0.9, mu=0.1)

def generate_things_vs_alpha() -> None:
    mu = 0.1
    N = 100
    alphavalues = np.linspace(0.01, 2, 200)  # avoid 0 to prevent division by zero

    S = compute_S(N=N, alpha=alphavalues, mu=mu)
    S_approx = compute_S_approx(N=N, alpha=alphavalues, mu=mu)
    
    # masks for splitting domain
    mask_less_than_1 = alphavalues < 1
    mask_greater_than_1 = alphavalues > 1
    
    R_over_S_true = compute_R_over_S(N=N, alpha=alphavalues, mu=mu)
    R_over_S_approx = compute_R_over_S_approx(N=N, alpha=alphavalues, mu=mu)
  
    # create two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    
    # Left plot: α in (0,1)
    axes[0,0].plot(
        alphavalues[mask_less_than_1],
        S[mask_less_than_1],
        label=EXACT_LABEL,
        color=EXACT_COLOR,
        linewidth=2,
    )
    axes[0,0].plot(
        alphavalues[mask_less_than_1],
        S_approx[mask_less_than_1],
        "--",
        label=APPROXIMATION_LABEL,
        color=APPROXIMATION_COLOR,
    )
#    axes[0,0].set_xlabel("social earning efficiency $\\alpha$")
    axes[0,0].set_xticklabels([])
    axes[0,0].set_ylabel("$E(S)$")
    axes[0,0].set_title("Total amount of culture")
    axes[0,0].legend()
    axes[0,0].grid(True)
    axes[0,0].set_yscale('log')
    
    # Right plot: α in (1,2)
    axes[0,1].plot(
        alphavalues[mask_greater_than_1],
        S[mask_greater_than_1],
        label=EXACT_LABEL,
        color=EXACT_COLOR,
        linewidth=2,
    )
    axes[0,1].plot(
        alphavalues[mask_greater_than_1],
        S_approx[mask_greater_than_1],
        "--",
        label=APPROXIMATION_LABEL,
        color=APPROXIMATION_COLOR,
    )
#    axes[0,1].set_xlabel("social earning efficiency $\\alpha$")
    axes[0,1].set_xticklabels([])
    axes[0,1].set_ylabel("$E(S)$")
    axes[0,1].set_title("Total amount of culture")
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].set_yscale('log')

    # Left panel: α in (0,1)
    axes[1,0].plot(
        alphavalues[mask_less_than_1],
        R_over_S_true[mask_less_than_1],
        label=EXACT_LABEL,
        color=EXACT_COLOR,
        linewidth=2,
    )
    axes[1,0].plot(
        alphavalues[mask_less_than_1],
        R_over_S_approx[mask_less_than_1],
        "--",
        label=APPROXIMATION_LABEL,
        color=APPROXIMATION_COLOR,
    )
    axes[1,0].set_xlabel(r"social learning efficiency $\alpha$")
    axes[1,0].set_ylabel(r"$E(R) / E(S)$")
    axes[1,0].set_title(r"Proportion of traits per individual")
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Right panel: α in (1,2)
    axes[1,1].plot(
        alphavalues[mask_greater_than_1],
        R_over_S_true[mask_greater_than_1],
        label=EXACT_LABEL,
        color=EXACT_COLOR,
        linewidth=2,
    )
    axes[1,1].plot(
        alphavalues[mask_greater_than_1],
        R_over_S_approx[mask_greater_than_1],
        "--",
        label=APPROXIMATION_LABEL,
        color=APPROXIMATION_COLOR,
    )
    axes[1,1].set_xlabel(r"social learning efficiency $\alpha$")
    axes[1,1].set_ylabel(r"$E(R) / E(S)$")
    axes[1,1].set_title(r"Proportion of traits per individual")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    save_figure(f"things_vs_alpha_N{N}_mu{mu}.pdf")


def main() -> None:
    generate_figure_popularity_distribution_alpha_smaller_than_one()
    generate_figure_popularity_distribution_alpha_larger_than_one()
    generate_figure_S_vs_N_when_alpha_smaller_than_one()
    generate_figure_S_vs_N_when_alpha_larger_than_one()
    generate_things_vs_alpha()
    plt.show()
    
if __name__ == '__main__':
    main()


