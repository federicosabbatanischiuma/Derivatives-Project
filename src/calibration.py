# Packages imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import warnings
from tqdm import tqdm

try:
    from src.vix_calculations import compute_vix_futures_price
    from src.futures_models import compute_variance_futures_price
except ImportError:
    from vix_calculations import compute_vix_futures_price
    from futures_models import compute_variance_futures_price

# Suppress potential integration warnings for cleaner output.
warnings.filterwarnings('ignore', category=UserWarning)

# Market Data -- Synthetic
vix_futures_data = [
    {'symbol': 'VX/K5', 'expiry': '5/21/25', 'price': 22.3484, 'days': 12},
    {'symbol': 'VX/M5', 'expiry': '6/18/25', 'price': 21.8897, 'days': 40},
    {'symbol': 'VX/N5', 'expiry': '7/16/25', 'price': 21.7491, 'days': 68},
    {'symbol': 'VX/Q5', 'expiry': '8/20/25', 'price': 21.7805, 'days': 103},
    {'symbol': 'VX/U5', 'expiry': '9/17/25', 'price': 21.8737, 'days': 131},
    {'symbol': 'VX/V5', 'expiry': '10/22/25', 'price': 22.0178, 'days': 166},
    {'symbol': 'VX/X5', 'expiry': '11/19/25', 'price': 22.1365, 'days': 194},
    {'symbol': 'VX/Z5', 'expiry': '12/17/25', 'price': 22.2502, 'days': 222},
]

variance_futures_data = [
    {'symbol': 'VA/K5', 'price': 579.58, 'accrued': 582.219, 'total_d': 123, 'elapsed_d': 119, 'remain_d': 4},
    {'symbol': 'VA/M5', 'price': 476.95, 'accrued': 470.917, 'total_d': 186, 'elapsed_d': 159, 'remain_d': 27},
    {'symbol': 'VA/N5', 'price': 699.11, 'accrued': 805.347, 'total_d': 123, 'elapsed_d': 77, 'remain_d': 46},
    {'symbol': 'VA/Q5', 'price': 783.27, 'accrued': 1092.85, 'total_d': 120, 'elapsed_d': 54, 'remain_d': 66},
    {'symbol': 'VA/U5', 'price': 796.14, 'accrued': 1474.92, 'total_d': 124, 'elapsed_d': 34, 'remain_d': 90},
    {'symbol': 'VA/V5', 'price': 517.64, 'accrued': 300.293, 'total_d': 125, 'elapsed_d': 15, 'remain_d': 110},
    {'symbol': 'VA/Z5', 'price': 515.90, 'accrued': 470.908, 'total_d': 313, 'elapsed_d': 159, 'remain_d': 154},
    {'symbol': 'VA/M6', 'price': 619.76, 'accrued': 691.246, 'total_d': 372, 'elapsed_d': 95, 'remain_d': 277},
]

def plot_calibration_results(calibrated_params, output_dir):
    """Generates and saves plots comparing market prices to model prices."""
    print("\n--- Generating Calibration Plots ---")
    V_t_cal, lambda_cal, theta_cal, xi_cal = calibrated_params

    # Plot for VIX Futures
    plt.figure(figsize=(10, 6))
    market_maturities = [c['days'] / 252.0 for c in vix_futures_data]
    market_prices = [c['price'] for c in vix_futures_data]
    model_prices = [compute_vix_futures_price(tau, *calibrated_params) for tau in market_maturities]
    
    plt.scatter(market_maturities, market_prices, label='Market Prices', color='red', zorder=5)
    plt.plot(market_maturities, model_prices, label='Model Prices', linestyle='-', marker='o', color='blue')
    plt.title('VIX Futures: Market vs. Calibrated Model')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('VIX Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calibration_vix_futures.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VIX futures comparison plot saved to '{output_dir}'.")

    # Plot for Variance Futures
    plt.figure(figsize=(10, 6))
    market_maturities_var = [c['remain_d'] / 252.0 for c in variance_futures_data]
    market_prices_var = [c['price'] for c in variance_futures_data]
    model_prices_var = [compute_variance_futures_price(c['remain_d']/252.0, V_t_cal, c['accrued'] * (c['elapsed_d']/252.0), c['elapsed_d']/252.0, lambda_cal, theta_cal, xi_cal) for c in variance_futures_data]
    
    # Sort by maturity for a clean line plot
    sorted_data = sorted(zip(market_maturities_var, market_prices_var, model_prices_var))
    market_maturities_var, market_prices_var, model_prices_var = zip(*sorted_data)

    plt.scatter(market_maturities_var, market_prices_var, label='Market Prices', color='red', zorder=5)
    plt.plot(market_maturities_var, model_prices_var, label='Model Prices', linestyle='-', marker='o', color='blue')
    plt.title('Variance Futures: Market vs. Calibrated Model')
    plt.xlabel('Time to Maturity (τ) in Years')
    plt.ylabel('Variance Futures Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calibration_variance_futures.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Variance futures comparison plot saved to '{output_dir}'.")


def run_calibration_pipeline(vix_data, variance_data, initial_guess, bounds):
    """
    Runs the full calibration process: optimization, printing results, and plotting.
    """
    
    def calibration_error(params):
        """Error function to be minimized. Defined inside to access market data."""
        V_t, lambda_p, theta_p, xi_p = params
        total_error = 0.0

        # Calculate VIX futures error (squared difference)
        for contract in vix_data:
            tau = contract['days'] / 252.0
            # Skip the spot VIX for calibration if its tau is 0
            model_price = compute_vix_futures_price(tau, V_t, lambda_p, theta_p, xi_p)
            total_error += (model_price - contract['price'])**2

        # Calculate Variance futures error (scaled squared difference)
        for contract in variance_data:
            tau = contract['remain_d'] / 252.0
            tau_elapsed = contract['elapsed_d'] / 252.0
            accrued_variance_input = contract['accrued'] * tau_elapsed
            
            model_price = compute_variance_futures_price(tau, V_t, accrued_variance_input, tau_elapsed, lambda_p, theta_p, xi_p)
            # Variance prices are large, so we scale the error to be comparable to VIX error (100)
            total_error += ((model_price - contract['price']) / 10)**2

        return total_error

    print("--- Starting Model Calibration ---")
    
    # Initialize and run the optimizer with a progress bar
    with tqdm(desc="Calibrating Parameters", unit=" iter") as progress_bar:
        def callback_func(xk):
            progress_bar.update(1)
        
        result = minimize(
            calibration_error, 
            initial_guess, 
            bounds=bounds, 
            method='L-BFGS-B', 
            callback=callback_func
        )
    
    calibrated_params = result.x
    V_t_cal, lambda_cal, theta_cal, xi_cal = calibrated_params
    
    print("\n--- Calibration Complete ---")
    print(f"Final minimized error: {result.fun:.4f}")
    print("\nCalibrated Parameters:")
    print(f"  Current Squared Volatility (V_t): {V_t_cal:.4f}  (Implies VIX of {100*np.sqrt(V_t_cal):.2f})")
    print(f"  Mean-Reversion Speed (λ):         {lambda_cal:.4f}")
    print(f"  Long-Run Mean Variance (θ):       {theta_cal:.4f}  (Implies long-run VIX of {100*np.sqrt(theta_cal):.2f})")
    print(f"  Volatility of Volatility (ξ):     {xi_cal:.4f}")

    # Print detailed results table
    print("\n--- Model Fit Analysis ---")
    print("\nVIX Futures:")
    print(f"{'Symbol':<8} {'Market':>10} {'Model':>10} {'Difference':>12}")
    print("-" * 42)
    for contract in vix_data:
        tau = contract['days'] / 252.0
        model_price = compute_vix_futures_price(tau, *calibrated_params)
        diff = model_price - contract['price']
        print(f"{contract['symbol']:<8} {contract['price']:>10.4f} {model_price:>10.4f} {diff:>12.4f}")

    print("\nS&P 500 Variance Futures:")
    print(f"{'Symbol':<8} {'Market':>10} {'Model':>10} {'Difference':>12}")
    print("-" * 42)
    for contract in variance_data:
        tau = contract['remain_d'] / 252.0
        tau_elapsed = contract['elapsed_d'] / 252.0
        accrued_variance_input = contract['accrued'] * tau_elapsed
        model_price = compute_variance_futures_price(tau, V_t_cal, accrued_variance_input, tau_elapsed, lambda_cal, theta_cal, xi_cal)
        diff = model_price - contract['price']
        print(f"{contract['symbol']:<8} {contract['price']:>10.2f} {model_price:>10.2f} {diff:>12.2f}")
    
    # Generate and save plots
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    plot_calibration_results(calibrated_params, output_dir)


if __name__ == '__main__':
    # Define initial guess and bounds for the optimization
    initial_guess = [0.04, 2.5, 0.05, 0.6]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)]
    
    # Run the entire pipeline
    run_calibration_pipeline(vix_futures_data, variance_futures_data, initial_guess, bounds)