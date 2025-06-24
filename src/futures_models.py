# Packages imports
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_variance_futures_price(
    tau: float,
    V_t: float,
    accrued_variance_input: float,
    tau_elapsed: float,
    lambda_p: float,
    theta_p: float,
    xi_p: float
) -> float:
    """
    Computes the price of a Variance Futures contract
    """
    # xi_p is not used in the calculation, as the price depends only on the
    # expected value of the variance, not its own volatility.

    if np.isclose(tau, 0): # Avoid division by zero if at expiry
        b_star = 1.0 # The limit of the expression as tau -> 0 -- L'Hôpital's Rule
    else:
        b_star = (1 - np.exp(-lambda_p * tau)) / (lambda_p * tau)
    
    a_star = theta_p * (1 - b_star)

    # Calculate the total expected future variance.
    expected_future_var = a_star * tau + b_star * tau * V_t

    # Process the accrued variance term from the input
    # The project gives the input as integral[(100*sqrt(V_u))^2 du]
    accrued_var = accrued_variance_input / 10000.0

    # Calculate the total term of the contract (T - t_0)
    total_term = tau + tau_elapsed
    if np.isclose(total_term, 0):
        return np.nan # Result is undefined if the contract has no duration

    total_integrated_variance = accrued_var + expected_future_var
    price = 10000 * total_integrated_variance / total_term

    return price

def analyze_variance_futures_sensitivity():
    """
    Analyzes how the Variance futures price depends on the model parameters
    """
    print("--- Running Variance Futures Term Structure Sensitivity Analysis ---")

    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Base Case Parameters for Analysis
    base_V_t = 0.04  # VIX = 20
    base_tau_elapsed = 0.25 # 3 months elapsed
    # For this analysis, we assume the accrued part and time elapsed are fixed.
    base_accrued_var = 10000 * base_V_t * base_tau_elapsed # Simplified assumption
    base_params = {'lambda_p': 2.5, 'theta_p': 0.06, 'xi_p': 0.7}
    tau_range = np.linspace(0.01, 1.0, 100) # Time to maturity from ~4 days to 1 year

    # Sensitivity to Lambda (Mean-Reversion Speed)
    print('Generating plot for sensitivity to Lambda...')
    plt.figure(figsize=(8, 6))
    lambda_values = np.linspace(1.5, 5.5, 5)

    for lam in lambda_values:
        prices = [compute_variance_futures_price(tau, base_V_t, base_accrued_var, base_tau_elapsed, lam, base_params['theta_p'], base_params['xi_p']) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\lambda={lam:.1f}$')

    plt.title('Variance Futures Term Structure vs. $\\lambda$')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('Variance Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'variance_term_structure_vs_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Sensitivity to Theta (Long-Run Variance)
    print('Generating plot for sensitivity to Theta...')
    plt.figure(figsize=(8, 6))
    theta_values = np.linspace(0.02, 0.10, 5)

    for a_theta in theta_values:
        prices = [compute_variance_futures_price(tau, base_V_t, base_accrued_var, base_tau_elapsed, base_params['lambda_p'], a_theta, base_params['xi_p']) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\theta={a_theta:.2f}$')

    plt.title('Variance Futures Term Structure vs. $\\theta$')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('Variance Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'variance_term_structure_vs_theta.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Sensitivity to Xi (Volatility of Volatility)
    print('Generating plot for sensitivity to Xi...')
    plt.figure(figsize=(8, 6))
    xi_values = np.linspace(0.3, 1.1, 5)

    for a_xi in xi_values:
        prices = [compute_variance_futures_price(tau, base_V_t, base_accrued_var, base_tau_elapsed, base_params['lambda_p'], base_params['theta_p'], a_xi) for tau in tau_range]
        plt.plot(tau_range, prices, label=f'$\\xi={a_xi:.1f}$')

    plt.title('Variance Futures Term Structure vs. $\\xi$ (Price is Independent of $\\xi$)')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('Variance Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'variance_term_structure_vs_xi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\nSensitivity analysis complete. Plots saved to \'{output_dir}\'.')


if __name__ == '__main__': 
    analyze_variance_futures_sensitivity() 