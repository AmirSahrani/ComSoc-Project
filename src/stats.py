import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import pymc as pm

from main import save_data
from plotting import load


def split(key):
    return key.split(", ")


def fit_parallel(results, func, kwargs, n_processes=None):
    if n_processes is None:
        n_processes = max(mp.cpu_count() // 2, 1)

    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Map the fitting function to all combinations
        worker_func = partial(func, **kwargs)
        results_list = pool.map(worker_func, results.items())

    # Convert results list to nested dictionary
    merged_results = {}
    for result_dict in results_list:
        for rule, utility_dict in result_dict.items():
            if rule not in merged_results:
                merged_results[rule] = {}
            for utility_function, result in utility_dict.items():
                merged_results[rule][utility_function] = result

    return merged_results


def get_correlations(results):
    """
    Return correlations and credibility intervals on results.
    Parameters:
        results (dict): Dictionary with keys as "voting_rule, utility_function" and values as np.array
                        of shape [N, M, num_trials].
    Returns:
        dict: Dictionary containing correlation statistics for each voting rule and utility function combination.
    """
    stats = {}
    for key, data in results.items():
        voting_rule, utility_function = key.split(", ")
        # Flatten the data to align with correlation inputs
        N, M, num_trials = data.shape
        n_vals, m_vals = np.meshgrid(range(N), range(M), indexing="ij")
        n_flat = n_vals.flatten()
        m_flat = m_vals.flatten()
        y_flat = data.mean(axis=2).flatten()  # Average over trials

        # Calculate correlations
        ny_correlation = np.corrcoef(n_flat, y_flat)[0, 1]
        my_correlation = np.corrcoef(m_flat, y_flat)[0, 1]

        # Bootstrap confidence intervals
        n_bootstrap_correlations = []
        m_bootstrap_correlations = []
        n_samples = len(y_flat)

        for _ in range(1000):  # Number of bootstrap iterations
            indices = np.random.randint(0, n_samples, size=n_samples)
            n_bootstrap = n_flat[indices]
            m_bootstrap = m_flat[indices]
            y_bootstrap = y_flat[indices]

            n_bootstrap_correlations.append(np.corrcoef(n_bootstrap, y_bootstrap)[0, 1])
            m_bootstrap_correlations.append(np.corrcoef(m_bootstrap, y_bootstrap)[0, 1])

        # Calculate means and confidence intervals
        ny_correlation_mean = np.mean(n_bootstrap_correlations)
        my_correlation_mean = np.mean(m_bootstrap_correlations)

        ny_correlation_ci = np.percentile(n_bootstrap_correlations, [0.5, 99.5])
        my_correlation_ci = np.percentile(m_bootstrap_correlations, [0.5, 99.5])

        # Store results
        stats.setdefault(voting_rule, {})[utility_function] = {
            "n_y_correlation_mean": ny_correlation_mean,
            "n_y_correlation_ci99": ny_correlation_mean - ny_correlation_ci[1],
            "m_y_correlation_mean": my_correlation_mean,
            "m_y_correlation_ci99": my_correlation_mean - my_correlation_ci[1],
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
    key, data = key_data
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
        posterior_predictive = pm.sample_posterior_predictive(
            trace, return_inferencedata=False
        )

        # Get posterior predictive samples
        y_obs_samples = posterior_predictive["y_obs"]
        lower_bound, upper_bound = np.percentile(y_obs_samples, [0.5, 99.5])

    return {
        voting_rule: {
            utility_function: {
                "mean": mean,
                "std": std,
                "ci 99%": (
                    upper_bound - mean
                ),  # example, refine based on your specific need
            }
        }
    }


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)

    results = load("results/random_sampling_k_15.pkl")
    results_data = load("results/sushi_data_k_15.pkl")
    stats = get_correlations(results)
    models_means = fit_parallel(results_data, mean_estimate, {})
    save_data(stats, "results/stats_regression_k_15.pkl")
    save_data(models_means, "results/stats_means_k_15.pkl")

    df = pd.DataFrame(stats)
    print(df.head())

    df = pd.DataFrame(models_means)
    print(df.head())


if __name__ == "__main__":
    main()
