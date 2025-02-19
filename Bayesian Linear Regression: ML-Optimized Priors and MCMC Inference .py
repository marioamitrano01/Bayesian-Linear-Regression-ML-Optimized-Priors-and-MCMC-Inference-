import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import plotly.graph_objects as go
import plotly.io as pio

# Import scikit-learn regressors for complex ML initialization
from sklearn.linear_model import RidgeCV, HuberRegressor

# Set the default Plotly renderer (opens the plot in your browser)
pio.renderers.default = "browser"

# Set random seeds for reproducibility
pyro.set_rng_seed(42)
torch.manual_seed(42)


@dataclass
class BayesianLRConfig:
    """
    Configuration for Bayesian Linear Regression MCMC inference.
    
    Attributes:
        num_samples (int): Number of MCMC samples after warmup.
        warmup_steps (int): Number of warmup (burn-in) steps.
        chains (int): Number of MCMC chains.
        debug (bool): Enable verbose debug output.
        save_trace (bool): Whether to save the posterior trace.
        
    """
    num_samples: int = 2000
    warmup_steps: int = 1000
    chains: int = 1
    debug: bool = False
    save_trace: bool = False
    trace_filename: Optional[str] = "posterior_trace.pt"


def complex_ml_initialization(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    



    # Flatten X for scikit-learn
    X_flat = X.reshape(-1, 1)
    
    # Ridge Regression with cross-validation
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge_cv.fit(X_flat, y)
    ridge_intercept = ridge_cv.intercept_
    ridge_slope = ridge_cv.coef_[0]
    
    # Huber Regressor for robust estimation
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber.fit(X_flat, y)
    huber_intercept = huber.intercept_
    huber_slope = huber.coef_[0]
    
    # Combine estimates (simple average)
    est_intercept = (ridge_intercept + huber_intercept) / 2.0
    est_slope = (ridge_slope + huber_slope) / 2.0
    
    return est_intercept, est_slope


class BayesianLinearRegression:
    """
    Bayesian Linear Regression Model using Pyro.
    
    Attributes:
        X (torch.Tensor): Predictor tensor of shape (N, 1).
        y (torch.Tensor): Response tensor of shape (N,).
        config (BayesianLRConfig): MCMC configuration.
        prior_beta_0_mean (float): Prior mean for β₀.
        prior_beta_0_sd (float): Prior standard deviation for β₀.
        prior_beta_1_mean (float): Prior mean for β₁.
        prior_beta_1_sd (float): Prior standard deviation for β₁.
        posterior_samples (Optional[Dict[str, torch.Tensor]]): Posterior samples.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 config: BayesianLRConfig, 
                 prior_beta_0_mean: float = 0.0, 
                 prior_beta_0_sd: float = 10.0,
                 prior_beta_1_mean: float = 0.0,
                 prior_beta_1_sd: float = 10.0) -> None:
        self.config = config
        self.X_np: np.ndarray = X
        self.y_np: np.ndarray = y
        
        # Device selection: use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Convert data to PyTorch tensors on the selected device
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y: torch.Tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.prior_beta_0_mean = prior_beta_0_mean
        self.prior_beta_0_sd = prior_beta_0_sd
        self.prior_beta_1_mean = prior_beta_1_mean
        self.prior_beta_1_sd = prior_beta_1_sd
        self.posterior_samples: Optional[Dict[str, torch.Tensor]] = None
        self.mcmc: Optional[MCMC] = None

    def model(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pyro model for Bayesian linear regression with informative priors.
        
        Args:
            X (torch.Tensor): Predictor tensor (N, 1).
            y (Optional[torch.Tensor]): Observed responses (N,).
        
        Returns:
            torch.Tensor: Linear predictor μ.
        """
        beta_0 = pyro.sample("beta_0", dist.Normal(self.prior_beta_0_mean, self.prior_beta_0_sd))
        beta_1 = pyro.sample("beta_1", dist.Normal(self.prior_beta_1_mean, self.prior_beta_1_sd))
        sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
        mu = beta_0 + beta_1 * X.squeeze(-1)
        with pyro.plate("data", X.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

    def run_mcmc(self) -> Dict[str, torch.Tensor]:
        """
        Runs MCMC sampling using NUTS to approximate the posterior.
        
        Returns:
            Dict[str, torch.Tensor]: Posterior samples.
        """
        try:
            kernel = NUTS(self.model)
            self.mcmc = MCMC(
                kernel,
                num_samples=self.config.num_samples,
                warmup_steps=self.config.warmup_steps,
                num_chains=self.config.chains,
                disable_progbar=not self.config.debug,
            )
            logging.info("Starting MCMC sampling using NUTS...")
            self.mcmc.run(self.X, self.y)
            logging.info("MCMC sampling complete.")
            self.posterior_samples = self.mcmc.get_samples()
            if self.config.save_trace and self.config.trace_filename:
                torch.save(self.posterior_samples, self.config.trace_filename)
                logging.info(f"Posterior samples saved to {self.config.trace_filename}")
        except Exception as e:
            logging.error("An error occurred during MCMC sampling.", exc_info=e)
            raise e
        return self.posterior_samples

    def compute_posterior_predictive(self, X_new: np.ndarray, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Computes the posterior predictive distribution for new data.
        
        Args:
            X_new (np.ndarray): New predictor data (M, 1).
            num_samples (int): Number of predictive samples.
        
        Returns:
            Dict[str, np.ndarray]: Contains keys:
                "y_pred": Predictive samples (num_samples, M)
                "mean": Predictive mean (M,)
                "std": Predictive standard deviation (M,)
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available. Run MCMC inference first.")
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        predictive = Predictive(self.model, posterior_samples=self.posterior_samples, return_sites=["obs"])
        samples = predictive(X_new_tensor)
        # Move samples to CPU if necessary
        y_pred_samples = samples["obs"].detach().cpu().numpy()
        mean_pred = y_pred_samples.mean(axis=0)
        std_pred = y_pred_samples.std(axis=0)
        return {"y_pred": y_pred_samples, "mean": mean_pred, "std": std_pred}

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Computes and logs a summary of the posterior samples, including means, standard deviations,
        and 95% credible intervals for each parameter.
        
        Returns:
            Dict[str, Dict[str, float]]: Summary statistics.
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available for summary.")
        summary_dict = {}
        for param, samples in self.posterior_samples.items():
            samples_np = samples.detach().cpu().numpy()
            mean_val = np.mean(samples_np)
            std_val = np.std(samples_np)
            ci_lower, ci_upper = np.percentile(samples_np, [2.5, 97.5])
            summary_dict[param] = {
                "mean": mean_val,
                "std": std_val,
                "2.5%": ci_lower,
                "97.5%": ci_upper,
            }
            logging.info(f"Parameter {param}: {summary_dict[param]}")
        return summary_dict


def simulate_linear_data(n_samples: int = 10000, noise_std: float = 1.0, hetero: bool = False, random_seed: int = 42
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates synthetic linear data with Gaussian noise.
    
    Model:
        y = β₀ + β₁ x + ε,     with     ε ~ Normal(0, σ²)
    
    The true parameters are randomly drawn:
        β₀ ~ Uniform(-5, 5)
        β₁ ~ Uniform(0, 5)
    
    If hetero is True, noise standard deviation varies with x.
    
    Args:
        n_samples (int): Number of samples.
        noise_std (float): Baseline noise standard deviation.
        hetero (bool): Whether to simulate heteroscedastic noise.
        random_seed (int): Random seed.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: Predictor array of shape (n_samples, 1).
            - y: Response array of shape (n_samples,).
            - true_params: Array [β₀, β₁] of true parameters.
    """
    np.random.seed(random_seed)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    true_beta_0 = np.random.uniform(-5, 5)
    true_beta_1 = np.random.uniform(0, 5)
    signal = true_beta_0 + true_beta_1 * X[:, 0]
    if hetero:
        sigma = noise_std * (1 + 0.5 * X[:, 0])
    else:
        sigma = noise_std * np.ones_like(X[:, 0])
    noise = np.random.normal(0, sigma)
    y = signal + noise
    return X, y, np.array([true_beta_0, true_beta_1])


def plot_results(X: np.ndarray, y: np.ndarray, predictive_results: Dict[str, np.ndarray]) -> None:
   
    mean_pred = predictive_results["mean"]
    std_pred = predictive_results["std"]
    lower_bound = mean_pred - 2 * std_pred
    upper_bound = mean_pred + 2 * std_pred

    fig = go.Figure()
    # Plot observations
    fig.add_trace(go.Scatter(
        x=X.ravel(),
        y=y,
        mode='markers',
        name='Observations',
        marker=dict(color='blue', size=8, opacity=0.7)
    ))
    # Plot posterior predictive mean as a dashed red line
    fig.add_trace(go.Scatter(
        x=X.ravel(),
        y=mean_pred,
        mode='lines',
        name='Posterior Predictive Mean',
        line=dict(color='red', width=2, dash='dash')
    ))
    # Plot 95% credible interval as a filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([X.ravel(), X.ravel()[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='95% Credible Interval'
    ))
    # Annotation with advanced techniques note (kept in red as requested)
    fig.add_annotation(
        x=0.5, y=0.95, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="red")
    )
    fig.update_layout(
        title='Bayesian Linear Regression: Posterior Predictive Check (Simulated Data)',
        xaxis_title='Predictor (x)',
        yaxis_title='Response (y)',
        template='plotly_white'
    )
    fig.show()


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for simulation and MCMC configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Linear Regression with Simulated Data and ML Initialization"
    )
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of simulated data samples (default: 10000)")
    parser.add_argument("--noise_std", type=float, default=1.0, help="Baseline noise standard deviation")
    parser.add_argument("--hetero", action="store_true", help="Simulate heteroscedastic noise")
    parser.add_argument("--num_mcmc_samples", type=int, default=2000, help="Number of MCMC samples to draw")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for MCMC")
    parser.add_argument("--chains", type=int, default=1, help="Number of MCMC chains")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output")
    parser.add_argument("--save_trace", action="store_true", help="Save the posterior trace to a file")
    parser.add_argument("--trace_filename", type=str, default="posterior_trace.pt", help="Filename for saving the trace")
    parser.add_argument("--num_predictive_samples", type=int, default=1000, help="Number of samples for the posterior predictive distribution")
    parser.add_argument("--plotly_renderer", type=str, default="browser", help="Plotly renderer (e.g., 'browser', 'notebook')")
    return parser.parse_args()


def main():
    # Parse arguments and set renderer if provided
    args = parse_arguments()
    pio.renderers.default = args.plotly_renderer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Setup configuration
    config = BayesianLRConfig(
        num_samples=args.num_mcmc_samples,
        warmup_steps=args.warmup_steps,
        chains=args.chains,
        debug=args.debug,
        save_trace=args.save_trace,
        trace_filename=args.trace_filename
    )
    
    logging.info("Simulating linear data...")
    X, y, true_params = simulate_linear_data(n_samples=args.n_samples,
                                             noise_std=args.noise_std,
                                             hetero=args.hetero)
    logging.info(f"Simulated Data: True parameters: β₀ = {true_params[0]:.3f}, β₁ = {true_params[1]:.3f}")
    
    logging.info("Estimating initial parameters using ML techniques...")
    ml_beta_0, ml_beta_1 = complex_ml_initialization(X, y)
    logging.info(f"ML Estimates: β₀ = {ml_beta_0:.3f}, β₁ = {ml_beta_1:.3f}")
    
    # Use the ML estimates as informative priors (with a reduced prior SD)
    ablr = BayesianLinearRegression(X, y, config,
                                              prior_beta_0_mean=ml_beta_0,
                                              prior_beta_0_sd=1.0,
                                              prior_beta_1_mean=ml_beta_1,
                                              prior_beta_1_sd=1.0)
    logging.info("Running MCMC inference using NUTS...")
    ablr.run_mcmc()
    logging.info("Posterior Summary:")
    ablr.summary()
    
    logging.info("Computing posterior predictive distribution...")
    predictive_results = ablr.compute_posterior_predictive(X, num_samples=args.num_predictive_samples)
    
    logging.info("Plotting the results...")
    plot_results(X, y, predictive_results)


if __name__ == "__main__":
    main()
