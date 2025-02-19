# Bayesian Linear Regression

This project implements a Bayesian linear regression model using Pyro with a twist: it leverages machine learning techniques to set informative priors.

## Key Features
- **ML-Optimized Priors:**  
  Uses Ridge Regression (with cross-validation) and Huber Regression to generate robust initial estimates for the regression parameters.
- **Bayesian Inference:**  
  Performs MCMC sampling with the NUTS algorithm to approximate the posterior distribution.
- **Visualization:**  
  Generates posterior predictive checks with interactive Plotly graphs.

## Requirements
- Python 3.x
- PyTorch, Pyro, NumPy, scikit-learn, Plotly

## Usage
Clone the repository and run the main script. Customize simulation and inference parameters via command-line arguments.


This project integrates machine learning methods to optimize the Bayesian priors, enhancing the overall robustness and accuracy of the inference process.

Happy coding!
