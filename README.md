# Learning from the Extremes: VaR Backtesting Toolkit

> A Python toolkit for **Value at Risk (VaR) estimation, backtesting, and simulation**, developed for the paper *“Learning from the Extremes: Machine Learning Approaches for Rare Event Prediction”*. 
> This repository implements standard and novel VaR estimation methods, including **MLE** and a **Forward Validation (FV)** approach, and supports multiple volatility models and distributions.

---

## Table of Contents



- [Project Structure](#project-structure)  

- [Installation](#installation)  

- [Volatility Models (`VolModel`)](#volatility-models-volmodel)  

- [Distributions (`Distribution`)](#distributions-distribution)  

- [Data](#data)  

- [Usage](#usage)  

- [Empirical VaR Backtesting](#empirical-var-backtesting)  

- [Monte Carlo Simulation](#monte-carlo-simulation)  

- [Backtesting Functions](#backtesting-functions)  

- [References](#references)  

---

## Project Structure

├─ data/ # Raw and preprocessed datasets (Excel files)

├─ src/

│ ├─ Distribution/ # Probability distributions

│ │ ├─ NormalDistribution.py

│ │ └─ StudentsDistribution.py

│ │

│ ├─ VolModel/ # Volatility models

│ │ ├─ BASEModel.py # parent class for all models

│ │ ├─ EWMAModel.py # Model classes

│ │ ├─ GARCHModel.py

│ │ └─ GJRGARCHModel.py

│ │

│ └─ utils.py # Preprocessing, backtesting, and support functions

│

├─ main.py # Empirical VaR backtesting script (used in the paper)

├─ main_simulation.py # Monte Carlo simulation of the FV estimator

└─ README.md

---

## Installation

1. Clone the repository:

```bash

git clone https://github.com/LucasAurouet/forward-validation.git

cd forward-validation

2. Install requirements

pip install -r requirements.txt


Dependencies include:

- numpy

- pandas

- scipy

- matplotlib

openpyxl (for Excel files)

---

## Code

Volatility Models (VolModel)

All models inherit from BASEModel, which implements:



- Maximum Likelihood Estimation (MLE) → fit_mle()

- Forward Validation Estimation (FV) → fit_fv()

- Value at Risk computation → get_valueatrisk()


Available Models

>- EWMAModel – Exponentially Weighted Moving Average volatility model.
>
>- GARCHModel – Standard GARCH model.
>
>- GJRGARCHModel – Asymmetric GJR-GARCH model capturing leverage effects.

Each model implements:



- model-specific conditional variance → get_variance()

- optimization support → init_params(), init_bounds(), constraints() → 


Distributions (Distribution)



These classes define the probability distributions for residuals.



NormalDistribution



loglik_resid(returns, variance, params) → log-likelihood of standardized residuals



ppf(q, params) → quantile function for VaR calculation



random_draw(mu=0, std=1) → generates a single random draw



StudentsDistribution



Similar to NormalDistribution but includes degrees-of-freedom for heavy-tailed behavior

---

## Data

Place raw Excel files in the data/ folder.

utils.prepare_data(path) handles:

- Loading Excel data

- Calculating percentage/log returns

- Cleaning missing values

- Constructing portfolios (Index, Commodity, Currency)

- Monte Carlo Simulation (main_simulation.py)

---

## Support Functions (utils.py)

- count_violations(returns, VaR) – number of breaches

- test_kupiec(returns, VaR, VaR_level) – unconditional coverage

- test_independence(returns, VaR, VaR_level) – conditional coverage

- test_duration(returns, VaR) – duration-based test (Christoffersen & Pelletier, 2004)

- test_dq(returns, VaR, v_lag, f_lag, VaR_level) – Engle & Manganelli dynamic quantile test

### Minimal Reproducible Example

This example shows how to run a simple VaR backtest using a small synthetic dataset.  



```python

import numpy as np

from src.VolModel.EWMAModel import EWMAModel

from src.Distribution.NormalDistribution import NormalDistribution

from src import utils



# Generate synthetic returns

np.random.seed(42)

returns = np.random.normal(0, 0.01, 100)  # 100 days of returns



# Split into train and test sets

train, test = utils.train_test_split(returns, train_size=0.8)



# Initialize model and distribution

dist = NormalDistribution()

model = EWMAModel(dist)



# Fit MLE parameters

model.fit_mle(train, show=False)



# Compute 5% VaR on test set

VaR_mle = model.get_valueatrisk(test, model.mle_params, 0.05)



# Run a simple backtest

kupiec_p = utils.test_kupiec(test, VaR_mle, 0.05)

indep_p = utils.test_independence(test, VaR_mle, 0.05)[0]



print(f"5% VaR Kupiec test p-value: {kupiec_p:.4f}")

print(f"5% VaR Independence test p-value: {indep_p:.4f}")

