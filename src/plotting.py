import pickle

import matplotlib.pyplot as plt
import numpy as np

from main import format_key, gen_ut_list, gen_vr_list


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
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
    xlabel: str,
    ylabel: str,
    n_vals: list[int] | range,
    m_vals: list[int] | range,
    show: bool = True,
    n_cols: int = 2,
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

            ax.set_title(title)
            ax.set_ylim((0, 8))
            ax.grid(True)
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

    # Save each figure separately
    for title in titles:
        plt.savefig(f"figures/{title}.pdf")

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


def load(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)

    voting_rules = gen_vr_list()
    socialwelfare_rules = gen_ut_list()

    results = load("results/random_sampling_k_10.pkl")
    results_data = load("results/sushi_data_k_10.pkl")

    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            plot_distortions(
                results[format_key(voting_rule["name"], sw["name"])],
                f"Distortion of {voting_rule['name']}, {sw['name']}",
                "Number of voters",
                "Distortion",
                n_vals,
                m_vals,
                show=False,
            )

    for sw in socialwelfare_rules:
        r = []
        names = []
        for voting_rule in voting_rules:
            names.append(voting_rule["name"])
            r.append(results_data[format_key(voting_rule["name"], sw["name"])])
        violin_plot_rules(
            np.array(r),
            f"Distortion Under {sw['name']} Social Utility",
            "Number of voters",
            "Distortion",
            names,
            show=False,
        )


if __name__ == "__main__":
    main()
