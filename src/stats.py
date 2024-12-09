import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
from plotting import load
from pref_voting import voting_methods as vr
import pymc as pm
import pandas as pd
import multiprocessing as mp
from functools import partial

from main import (format_key, gen_vr_list, gen_ut_list)
from utility_functions import (
    nash_optimal,
    nietzschean_optimal,
    rawlsian_optimal,
    utilitarian_optimal,
)


def split(key):
    return key.split(", ")

def fit_bayesian_regression(key_data, n_vals, m_vals):
    """
    Fit Bayesian regressions to results for each combination of voting rules and utility functions.

    Parameters:
        results (dict): Dictionary with keys "voting_rule, utility_function" and values as np.array
                        of shape [N, M, num_trials].
        n_vals (list or np.array): Array of N values corresponding to the first dimension.
        m_vals (list or np.array): Array of M values corresponding to the second dimension.

    Returns:
        dict: A dictionary containing the PyMC traces for each combination of voting_rule and utility_function.
    """
    key, data = key_data
    voting_rule, utility_function = split(key)

    # Flatten the data to align with regression inputs
    N, M, num_trials = data.shape
    n_grid, m_grid = np.meshgrid(n_vals, m_vals, indexing="ij")

    n_flat = n_grid.flatten()
    m_flat = m_grid.flatten()
    y_flat = data.mean(axis=2).flatten()  # Average over trials

    with pm.Model() as model:
        # Priors for regression coefficients
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_n = pm.Normal("beta_n", mu=0, sigma=10)
        beta_m = pm.Normal("beta_m", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Linear regression mean
        mu = alpha + beta_n * n_flat + beta_m * m_flat

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_flat)

        # Sampling
        trace = pm.sampling.sample(nuts_sampler="numpyro", progressbar=True)

    return (voting_rule, utility_function), trace


def fit_parallel(results, func, kwargs, n_processes=None):
    if n_processes is None:
        n_processes = max(mp.cpu_count()//2, 1)

    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Map the fitting function to all combinations
        worker_func = partial(func, **kwargs)
        results_list = pool.map(worker_func, results.items())

    # Convert results list to dictionary
    return dict(results_list)


def get_regression_stats(traces):
    """
    Extract regression statistics from PyMC traces.

    Parameters:
        traces (dict): Dictionary of PyMC traces from fit_bayesian_regression

    Returns:
        dict: Dictionary containing statistics for each voting rule and utility function combination
    """
    stats = {}

    for (voting_rule, utility_function), trace in traces.items():
        # Get beta_n statistics
        beta_n_samples = trace.posterior['beta_n'].values.flatten()
        beta_n_mean = np.mean(beta_n_samples)
        beta_n_std = np.std(beta_n_samples)
        beta_n_ci = np.percentile(beta_n_samples, [0.5, 99.5])  # 99% CI

        # Get beta_m statistics
        beta_m_samples = trace.posterior['beta_m'].values.flatten()
        beta_m_mean = np.mean(beta_m_samples)
        beta_m_std = np.std(beta_m_samples)
        beta_m_ci = np.percentile(beta_m_samples, [0.5, 99.5])  # 99% CI

        stats[(voting_rule, utility_function)] = {
            "voting_rule": voting_rule,
            "utility_function": utility_function,
            "beta_n": {
                "mean": beta_n_mean,
                "std": beta_n_std,
                "ci_99": tuple(beta_n_ci)
            },
            "beta_m": {
                "mean": beta_m_mean,
                "std": beta_m_std,
                "ci_99": tuple(beta_m_ci)
            }
        }

    return stats

def mean_estimate(key_data):
    """
    Compute mean, standard deviation, and Bayesian statistics for each result set.

    Parameters:
        results (dict): Dictionary with keys "voting_rule, utility_function" and values as 1D arrays of samples.

    Returns:
        dict: A dictionary with computed statistics for each key.
    """
    key,data = key_data
    stats = {}

    voting_rule, utility_function = split(key)

    # Compute basic statistics
    mean = np.mean(data)
    std = np.std(data)

    # Bayesian factor estimation (example using PyMC, adjust as needed)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=2, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data)

        # Compute log-likelihood for Bayes factor estimation
        trace = pm.sampling.sample(nuts_sampler="numpyro")
        posterior_predictive = pm.sample_posterior_predictive(trace, return_inferencedata=False)

        # Get posterior predictive samples
        y_obs_samples = posterior_predictive["y_obs"]
        lower_bound, upper_bound = np.percentile(y_obs_samples, [0.05, 95])

    return (voting_rule, utility_function), {
        "voting_rule": voting_rule,
        "utility_function": utility_function,
        "mean": mean,
        "std": std,
        "ci 99%": (lower_bound, upper_bound),  # example, refine based on your specific need
    }



def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 10)
    voting_rules = gen_vr_list()
    socialwelfare_rules = gen_ut_list()

    results = load("results/random_sampling.pkl")
    results_data = load("results/sushi_data.pkl")
    models = fit_parallel(results, fit_bayesian_regression,{"n_vals": n_vals, "m_vals": m_vals})
    stats = get_regression_stats(models)
    models_means = fit_parallel(results_data, mean_estimate, {})

    df = pd.DataFrame(stats)
    print(df.head())

    df = pd.DataFrame(models_means)
    print(df.head())



if __name__ == "__main__":
    main()
