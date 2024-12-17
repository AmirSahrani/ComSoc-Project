# %%

import pandas as pd

from plotting import load


def reformat_data(df, group_by_cols=["Voting Rule", "Utility Function"]):
    """
    Reformats data into a grouped DataFrame matching the LaTeX table format.

    Parameters:
        df (pd.DataFrame): Input DataFrame with hierarchical data
        group_by_cols (list): Columns to group by, default ['Voting Rule', 'Utility Function']

    Returns:
        pd.DataFrame: Reformatted DataFrame with hierarchical index
    """
    # Create empty lists to store the reformatted data
    rows = []

    # Iterate through the DataFrame
    for idx, row in df.iterrows():
        # Extract all the data columns (assuming they're dictionaries)
        for col, data in row.items():
            if isinstance(data, dict):
                # Create base row with group_by information
                new_row = {
                    group_col: col if group_col == "Utility Function" else idx
                    for group_col in group_by_cols
                }

                # Add all dictionary values as columns
                for key, value in data.items():
                    new_row[key] = value

                rows.append(new_row)

    # Create the DataFrame
    result_df = pd.DataFrame(rows)

    # Set multi-index using group_by_cols
    result_df = result_df.set_index(group_by_cols)

    # Sort the index
    result_df = result_df.sort_index()

    return result_df


def df_to_latex(df, precision=2):
    """
    Converts the reformatted DataFrame to LaTeX format with CIs shown as ± values.

    Parameters:
        df (pd.DataFrame): Input DataFrame with mean and CI columns
        precision (int): Number of decimal places for formatting

    Returns:
        str: LaTeX table string
    """
    # Create a copy to avoid modifying the original
    formatted_df = df.copy()

    # Function to format floats with consistent precision
    def format_float(x, prec=precision):
        return f"{{:.{prec}f}}".format(x)

    # Identify and combine mean/CI pairs
    cols_to_drop = []
    for col in df.columns:
        if "_ci95" in col or "_ci99" in col:  # Skip CI columns
            continue

        # Look for corresponding CI column
        ci_col = None
        if col == "n_y_correlation_mean":
            ci_col = "n_y_correlation_ci99"
        elif col == "m_y_correlation_mean":
            ci_col = "m_y_correlation_ci99"

        if ci_col:
            # Format means
            means = formatted_df[col].apply(format_float)

            # Format CIs - assuming CI is tuple of (lower, upper)
            cis = formatted_df[ci_col].apply(lambda x: format_float(abs(x)))

            # Combine with ± symbol
            formatted_df[col] = means + " $\\pm$ " + cis

            # Mark CI column for dropping
            cols_to_drop.append(ci_col)

    # Drop all CI columns at once
    formatted_df = formatted_df.drop(columns=cols_to_drop)

    return formatted_df.to_latex(
        multirow=True, escape=False, caption="Analysis Results", label="tab:results"
    )


# %%
dic_rand = load("../results/stats_regression_k_15.pkl")
dic_data = load("../results/stats_means_k_15.pkl")
df_rand = reformat_data(pd.DataFrame(dic_rand))
df_data = reformat_data(pd.DataFrame(dic_data))


# %%
print(df_to_latex(df_rand))
print(df_to_latex(df_data))
