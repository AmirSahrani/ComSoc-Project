# %%

import pandas as pd

from plotting import load

# %%
dic = load("../results/stats_regression_k_10.pkl")
df = pd.DataFrame(dic)


# %%
def process_voting_data(df):
    """
    Process voting data and return a reshaped dataframe grouped by voting rule and utility function.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns:
        - voting_rule
        - utility_function
        - beta_n_mean
        - beta_n_std
        - beta_n_ci99
        - beta_m_mean
        - beta_m_std
        - beta_m_ci99

    Returns:
    pd.DataFrame: Processed dataframe with hierarchical index (voting_rule, utility_function)
        and columns for means, std deviations, and confidence intervals for both beta_n and beta_m.
    """
    # Create new dataframe with processed data
    result_data = []

    # Process each row
    for _, row in df.iterrows():
        # Extract confidence intervals
        n_ci_low, n_ci_high = row["beta_n_ci99"]
        m_ci_low, m_ci_high = row["beta_m_ci99"]

        result_data.append(
            {
                "Voting rule": row["voting_rule"].strip(),
                "Utility function": row["utility_function"].strip(),
                "N mean": row["beta_n_mean"],
                "N 99% CI": f"({n_ci_low:.2f}, {n_ci_high:.2f})",
                "M mean": row["beta_m_mean"],
                "M 99% CI": f"({m_ci_low:.2f}, {m_ci_high:.2f})",
            }
        )

    # Create result dataframe
    result_df = pd.DataFrame(result_data)

    # Set multi-index
    result_df = result_df.set_index(["Voting rule", "Utility function"])

    # Sort index
    result_df = result_df.sort_index()

    return result_df


def parse_voting_data(df):
    """
    Parse voting data from text format and return a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns for voting_rule, utility_function, beta_n/m statistics.

    Returns:
    pd.DataFrame: DataFrame with parsed data.
    """
    data = []

    for _, row in df.items():
        data.append(
            {
                "voting_rule": row["voting_rule"].strip(),
                "utility_function": row["utility_function"].strip(),
                "beta_n_mean": float(row["beta_n_mean"]),
                "beta_n_ci99": tuple(map(float, row["beta_n_ci99"])),
                "beta_m_mean": float(row["beta_m_mean"]),
                "beta_m_ci99": tuple(map(float, row["beta_m_ci_99"])),
            }
        )

    return pd.DataFrame(data)


def format_strings(x):
    if isinstance(x, str):
        return x.replace("_", "").capitalize()
    return x  # Leave non-strings unchanged


df2 = process_voting_data(parse_voting_data(df))
print(
    df2.to_latex(
        float_format="%.3f",
        formatters={"Utility function": format_strings, "Voting rule": format_strings},
    )
)
