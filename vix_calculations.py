# Packages imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Constant eta (30 days / 252 trading days)
ETA = 30.0 / 252.0

def d_func(tau: float, u: float, lambda_p: float, xi_p: float) -> float:
    """
    This is the formula for d(T-t,u)
    """
    val = 2 * lambda_p + u * xi_p**2
    denominator = val * np.exp(lambda_p * tau) - u * xi_p**2
    return (2 * lambda_p * u) / denominator

def c_func(tau: float, u: float, lambda_p: float, theta_p: float, xi_p: float) -> float:
    """
    This is the formula for c(T-t,u)
    """
    val = 2 * lambda_p + u * xi_p**2
    
    # If val is non-positive, the formula is not well-defined
    if val <= 0:
        return np.inf

    # log_numerator = np.exp(lambda_p * tau) * val
    # log_numerator = np.exp(lambda_p * tau) * (2 * lambda_p)
    # log_denominator = (u * xi_p**2 + val * np.exp(lambda_p * tau))
    log_numerator = val - u * xi_p**2 * np.exp(- lambda_p * tau)
    log_denominator = 2 * lambda_p
    
    # Handle potential division by zero or log of non-positive number
    if log_denominator <= 0:
        return np.inf
    
    log_term = np.log(log_numerator / log_denominator)
    
    return (2 * lambda_p * theta_p / xi_p**2) * log_term


def compute_vix_futures_price(
    tau: float,
    V_t: float,
    lambda_p: float,
    theta_p: float,
    xi_p: float
) -> float:
    """
    Computes the VIX futures price
    """
    b_prime = (1 - np.exp(-lambda_p * ETA)) / lambda_p
    a_prime = theta_p * (ETA - b_prime)

    def integrand(s: float) -> float:
        """
        Defines the integrand for the VIX futures pricing formula, which is integrated over 's'.
        """
        # The argument for the Laplace transform functions is u = s * b'
        u = s * b_prime

        # Calculate the exponent l(s, tau, V_t)
        # l = s*a' + c(tau, u) + d(tau, u)*V_t
        c_val = c_func(tau, u, lambda_p, theta_p, xi_p)
        d_val = d_func(tau, u, lambda_p, xi_p)
        
        l_val = s * a_prime + c_val + d_val * V_t
        
        if np.isinf(l_val):
            return 1 / (s**1.5)
        if np.isnan(l_val):
            return 0

        return (1 - np.exp(-l_val)) / (s**1.5)

    # Perform the numerical integration from 0 to infinity
    integral_val, _ = quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9, limit=200)
    
    price = (50 / np.sqrt(np.pi * ETA)) * integral_val
    return price


def analyze_parameter_sensitivity():
    """
    Analyzes how the VIX futures term structure depends on the model parameters.
    Includes a visual proof of the convergence point for the Xi plot.
    """
    print('--- Running Term Structure Sensitivity Analysis ---')

    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Base Case Parameters for Analysis
    base_V_t = 0.04  # Corresponds to a VIX of 20
    base_params = {'lambda_p': 2.5, 'theta_p': 0.06, 'xi_p': 0.7}
    tau_range = np.linspace(0.005, 1.0, 200) # Time to maturity from ~1 day to 1 year

    # Sensitivity to Lambda
    print('Generating plot for sensitivity to Lambda...')
    plt.figure(figsize=(8, 6))
    lambda_values = np.linspace(0.5, 4.5, 5)
    for lam in lambda_values:
        prices = [compute_vix_futures_price(tau, base_V_t, lam, base_params['theta_p'], base_params['xi_p']) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\lambda={lam:.1f}$')
    plt.title('VIX Futures Term Structure vs. $\\lambda$')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('VIX Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'vix_term_structure_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Sensitivity to Theta
    print('Generating plot for sensitivity to Theta...')
    plt.figure(figsize=(8, 6))
    theta_values = np.linspace(0.02, 0.10, 5)
    for a_theta in theta_values:
        prices = [compute_vix_futures_price(tau, base_V_t, base_params['lambda_p'], a_theta, base_params['xi_p']) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\theta={a_theta:.2f}$')
    plt.title('VIX Futures Term Structure vs. $\\theta$')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('VIX Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'vix_term_structure_vs_theta.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Sensitivity to Xi
    print('Generating plot for sensitivity to Xi...')
    plt.figure(figsize=(8, 6))
    xi_values = np.linspace(0.3, 1.1, 5)
    for a_xi in xi_values:
        prices = [compute_vix_futures_price(tau, base_V_t, base_params['lambda_p'], base_params['theta_p'], a_xi) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\xi={a_xi:.1f}$')

    plt.title('VIX Futures Term Structure vs. $\\xi$')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('VIX Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'vix_term_structure_vs_xi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\nSensitivity analysis complete. Plots saved to \'{output_dir}\'.')

if __name__ == '__main__':
    analyze_parameter_sensitivity()