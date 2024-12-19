import pickle

import matplotlib.pyplot as plt
import numpy as np

from main import format_key, gen_ut_list, gen_vr_list

plt.rcParams.update(
    {
        "font.size": 23,
        "axes.labelsize": 23,
        "axes.titlesize": 23,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
        "legend.fontsize": 16,
        "axes.linewidth": 1,
        "grid.linewidth": 1,
        "grid.alpha": 0.3,
        "image.cmap": "viridis",
        "text.usetex": True,
        "font.family": "Computer Modern",
    }
)


def violin_plot_rules(
    distortions: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str],
    show: bool = True,
):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, distortions.shape[0]))

    parts = plt.violinplot(
        [distortions[i, :] for i in range(distortions.shape[0])], showmeans=True
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.4)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"figures/{title}.pdf")
    if show:
        plt.show()
    plt.close()


def plot_distortions_multi_fig(
    distortions: list[np.ndarray],
    titles: list[str],
    main_title: str,
    xlabel: str,
    ylabel: str,
    n_vals: list[int] | range,
    m_vals: list[int] | range,
    show: bool = True,
    n_cols: int = 4,
):
    """
    Plot multiple distortion figures with shared x and y labels.

    Args:
        distortions: List of distortion arrays, each with shape (n_samples, n_m_values, n_repetitions)
        titles: List of titles for each figure
        xlabel: Shared x-axis label
        ylabel: Shared y-axis label
        n_vals: Values for x-axis
        m_vals: Values for different lines in each plot
        show: Whether to display the plots
        n_cols: Number of columns in the grid of plots
    """
    n_figs = len(distortions)
    n_rows = (n_figs + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_figs == 1:
        axes = np.array([[axes]])  # Make it 2D for consistency
    elif n_rows == 1:
        axes = axes.reshape(1, -1)  # Make it 2D for consistency

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Create color map
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, distortions[0].shape[1]))

    for idx, (distortion, title, ax) in enumerate(zip(distortions, titles, axes_flat)):
        if idx < n_figs:  # Only plot if we have data
            var_distortion = np.var(distortion, axis=2)
            mean_distortion = np.mean(distortion, axis=2)

            for m in range(mean_distortion.shape[1]):
                ax.plot(
                    n_vals,
                    mean_distortion[:, m],
                    color=colors[m],
                    label=f"m={m_vals[m]}",
                )
                ax.fill_between(
                    n_vals,
                    mean_distortion[:, m] - var_distortion[:, m],
                    mean_distortion[:, m] + var_distortion[:, m],
                    color=colors[m],
                    alpha=0.2,
                )

            ax.axhline(1, alpha=0.3, linestyle="dashed", color="black")
            ax.set_title(title)
            ax.set_ylim((0, 8))
            ax.grid(True)
            if idx == 3:
                ax.legend(loc="upper right")
        else:
            # Hide unused subplots
            ax.set_visible(False)

    # Set shared labels
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    fig.savefig(f"figures/multiplot{main_title}.pdf")

    if show:
        plt.show()
    plt.close()


def plot_distortions(
    distortions: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    n_vals: list[int] | range,
    m_vals: list[int] | range,
    show: bool = True,
):
    var_distortions = np.var(distortions, axis=2)
    mean_distortions = np.mean(distortions, axis=2)

    plt.figure()
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, mean_distortions.shape[1]))

    for m in range(mean_distortions.shape[1]):
        plt.plot(
            n_vals, mean_distortions[:, m], color=colors[m], label=f"m={m_vals[m]}"
        )
        plt.fill_between(
            n_vals,
            mean_distortions[:, m] - var_distortions[:, m],
            mean_distortions[:, m] + var_distortions[:, m],
            color=colors[m],
            alpha=0.2,
        )

    plt.title(title)
    plt.ylim((0, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"figures/{title}.pdf")
    if show:
        plt.show()
    plt.close()


def violin_plot_social_welfare(
    results_data: dict,
    socialwelfare_rules: list[dict],
    voting_rules: list[dict],
    title: str,
    ylabel: str,
    show: bool = True,
):
    """
    Generate violin plots for multiple notions of social welfare.

    Args:
        results_data: Dictionary of results, where keys are formatted using format_key.
        socialwelfare_rules: List of social welfare rule dictionaries.
        voting_rules: List of voting rule dictionaries.
        title: Title of the plot.
        ylabel: Y-axis label.
        show: Whether to display the plot.
    """
    plt.figure(figsize=(16, 8))

    # Generate colors for each social welfare rule
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(socialwelfare_rules)))

    # Prepare data for the violin plot
    data = []
    positions = []
    labels = []
    color_mapping = []

    for vr_idx, voting_rule in enumerate(voting_rules):
        for sw_idx, sw in enumerate(socialwelfare_rules):
            key = format_key(voting_rule["name"], sw["name"])
            if key in results_data:
                data.append(results_data[key])
                positions.append(vr_idx * (len(socialwelfare_rules) + 1) + sw_idx)
                if sw_idx == 0:  # Only add voting rule labels once per group
                    labels.append(voting_rule["name"])
                color_mapping.append(colors[sw_idx])

    # Create the violin plot
    parts = plt.violinplot(data, positions=positions, showmeans=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_mapping[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)

    # Calculate tick positions for voting rules
    tick_positions = [
        vr_idx * (len(socialwelfare_rules) + 1) + (len(socialwelfare_rules) - 1) / 2
        for vr_idx in range(len(voting_rules))
    ]

    # Add labels and title
    plt.xticks(ticks=tick_positions, labels=labels, rotation=45, ha="right")
    plt.axhline(1, alpha=0.6, linestyle="dashed")
    plt.ylabel(ylabel)
    plt.ylim((0, 10))
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add legend for social welfare rules
    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=6, label=sw["name"])
        for i, sw in enumerate(socialwelfare_rules)
    ]
    plt.legend(handles=handles, title="Social Welfare Rules", loc="upper left")

    plt.tight_layout()

    # Save and optionally show the plot
    plt.savefig(f"figures/{title.replace(' ', '_')}.pdf")
    if show:
        plt.show()
    plt.close()


def load(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)

    voting_rules = gen_vr_list()
    socialwelfare_rules = gen_ut_list()

    results = load("results/random_sampling_k_15.pkl")
    results_data = load("results/sushi_data_k_15.pkl")

    for sw  in socialwelfare_rules:
        results_data[format_key("Inverse-Plurality", sw["name"])] = results_data[format_key("Anti-Plurality", sw["name"])]

    to_plot = [
        results[format_key("Anti-Plurality", sw["name"])] for sw in socialwelfare_rules
    ]
    titles_plot = [sw["name"] for sw in socialwelfare_rules]
    plot_distortions_multi_fig(
        distortions=to_plot,
        titles=titles_plot,
        main_title="Instance Distortion under the Anti-Plurality Rule",
        ylabel="Instance Distortion",
        xlabel="Number of voters",
        n_vals=n_vals,
        m_vals=m_vals,
        show=False,
    )

    to_plot = [
        results[format_key("Black's Rule", sw["name"])] for sw in socialwelfare_rules
    ]

    titles_plot = [sw["name"] for sw in socialwelfare_rules]
    plot_distortions_multi_fig(
        distortions=to_plot,
        titles=titles_plot,
        main_title="Instance Distortion under the Black Rule",
        ylabel="Instance Distortion",
        xlabel="Number of voters",
        n_vals=n_vals,
        m_vals=m_vals,
        show=False,
    )
    violin_plot_social_welfare(
        results_data,
        socialwelfare_rules,
        voting_rules,
        "Distribution of Instance Distortion",
        "Instance Distortion",
    )


if __name__ == "__main__":
    main()
