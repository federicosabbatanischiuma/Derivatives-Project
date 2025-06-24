# main.py
import argparse
import sys
from pathlib import Path

# Ensure src is in path if not using setup.py install
# This allows importing modules from the 'src' directory
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the necessary functions from the source modules
from src.vix_calculations import analyze_parameter_sensitivity as run_vix_sensitivity_analysis
from src.futures_models import analyze_variance_futures_sensitivity as run_variance_sensitivity_analysis
from src.calibration import run_calibration_pipeline, vix_futures_data, variance_futures_data

def run_part3_q6():
    """
    Runs the numerical analysis for Part 3, Question 6: VIX futures price.
    This function calls the sensitivity analysis, which generates and saves plots
    showing how the VIX futures price depends on model parameters (lambda, theta, xi).
    """
    print("Running Part 3, Question 6: VIX Futures Sensitivity Analysis...")
    run_vix_sensitivity_analysis()
    print("\nPart 3, Q6 - Analysis complete.")

def run_part3_q7():
    """
    Runs the graphical analysis for Part 3, Question 7: Variance futures price.
    This function calls the sensitivity analysis, which generates and saves plots
    showing how the variance futures price depends on model parameters (lambda, theta, xi).
    """
    print("Running Part 3, Question 7: Variance Futures Sensitivity Analysis...")
    run_variance_sensitivity_analysis()
    print("\nPart 3, Q7 - Analysis complete.")


def run_part3_q8():
    """
    Runs the model calibration for Part 3, Question 8.
    This function calibrates the Heston model parameters (V_t, lambda, theta, xi)
    using market data for VIX and Variance futures. It then prints the results
    and saves comparison plots.
    """
    print("Running Part 3, Question 8: Model Calibration...")
    
    # Define initial guess and bounds for the optimization, as specified in the calibration script
    initial_guess = [0.04, 2.5, 0.05, 0.6]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)]
    
    # The data is now imported directly from the calibration module.
    run_calibration_pipeline(vix_futures_data, variance_futures_data, initial_guess, bounds)
    print("\nPart 3, Q8 - Calibration complete.")


def main():
    """
    Main entry point for the project script.
    Parses command line arguments to run specific project parts.
    """
    parser = argparse.ArgumentParser(description="Fin404 Derivatives Project: VIX and Related Derivatives")
    
    # Argument to run a specific task by name
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['part3_q6', 'part3_q7', 'part3_q8'],
        help="Specify the task to run (e.g., part3_q6, part3_q7, part3_q8)"
    )
    
    # Argument to run all questions for a specific part
    parser.add_argument(
        '--all-questions',
        action='store_true',
        help="Run all questions for Part 3."
    )

    args = parser.parse_args()

    if args.task:
        if args.task == 'part3_q6':
            run_part3_q6()
        elif args.task == 'part3_q7':
            run_part3_q7()
        elif args.task == 'part3_q8':
            run_part3_q8()
    elif args.all_questions:
        print("--- Running all questions for Part 3 ---")
        run_part3_q6()
        print("-" * 50)
        run_part3_q7()
        print("-" * 50)
        run_part3_q8()
        print("\n--- All Part 3 tasks completed. ---")
    else:
        print("No specific task or part selected. Use --help for options.")
        parser.print_help()

if __name__ == '__main__':
    main()
