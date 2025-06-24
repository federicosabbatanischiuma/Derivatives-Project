# Fin404 Derivatives Project: The VIX and Related Derivatives

This repository contains the work for the Fin404 Derivatives project at EPFL, focusing on the study of the VIX index construction and the analysis of different approaches to modeling volatility.

## Project Structure

```text
fin404-project/             # root directory
├── data/                   # raw and processed datasets (e.g., Fin404-2025-VIXnCo-Data.xlsx)
├── src/                    # source code modules
│   ├── __init__.py
│   ├── vix_calculations.py # Functions for VIX index related calculations and analysis
│   ├── futures_models.py   # Implementation of VIX and Variance futures pricing models
│   ├── calibration.py      # Code for model calibration (Part 3, Q8)
├── results/                # Generated figures, tables, and potentially intermediate results
│   ├── figures/            # Plots and graphs
│   └── tables/             # Numerical results
├── main.py                 # Main script to run different parts of the project analysis
├── environment.yml         # Conda environment specification
├── requirements.txt        # pip-installable dependencies
├── setup.py                # Installation script for the 'src' package (optional)
└── README.md               # This file
```

## Authors

* Gabriele Calandrino [gabriele.calandrino@epfl.ch](mailto:gabriele.calandrino@epfl.ch)
* Alex Martinez de Francisco [alex.martinezdefrancisco@epfl.ch](mailto:alex.martinezdefrancisco@epfl.ch)
* Federico Sabbatani Schiuma [federico.sabbatanischiuma@epfl.ch](mailto:federico.sabbatanischiuma@epfl.ch)
* Letizia Seveso [letizia.seveso@epfl.ch](mailto:letizia.seveso@epfl.ch)

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate fin404-project
```

### 2. Install dependencies

Install your package in editable mode (core dependencies) and then extras:

```bash
pip install -e .
pip install -r requirements.txt  # notebook & test dependencies
```

### 3. Run Analyses and Generate Results

Use `main.py` with appropriate arguments to run specific analyses. The exact arguments will depend on your `main.py` implementation. For example:

```bash
# Example: Run VIX futures analysis (Part 3, Q6)
python main.py --task part3_q6

# Example: Run Variance futures analysis (Part 3, Q7)
python main.py --task part3_q7

# Example: Run model calibration (Part 3, Q8)
python main.py --task part3_q8

# Example: Generate all results for Part 3
python main.py --all-questions
```

### 4. Launch an individual file

```bash
python src/vix_calculations.py
```

## License

This project is licensed under the [MIT License](LICENSE).