import pickle

import matplotlib.pyplot as plt
import numpy as np
from pref_voting import voting_methods as vr

from main import format_key
from voting_rules import (
    nash_optimal,
    nietzschean_optimal,
    rawlsian_optimal,
    utilitarian_optimal,
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
    plt.savefig(f"figures/{title}.svg")
    if show:
        plt.show()


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
    plt.savefig(f"figures/{title}.svg")
    if show:
        plt.show()


def load(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    np.random.seed(1)
    n_vals = range(2, 10, 5)
    m_vals = range(2, 25, 10)
    borda_rule = {"rule": vr.borda, "name": "Borda rule"}
    copeland_rule = {"rule": vr.copeland, "name": "Copeland's Rule"}
    plurality_rule = {"rule": vr.plurality, "name": "Plurality rule"}
    blacks_rule = {"rule": vr.blacks, "name": "Black's Rule"}

    voting_rules = [borda_rule, copeland_rule, plurality_rule, blacks_rule]

    nash_rule = {"rule": nash_optimal, "name": "Nash"}
    utilitarian_rule = {"rule": utilitarian_optimal, "name": "Utiliratian"}
    rawlsian_rule = {"rule": rawlsian_optimal, "name": "Rawlsian"}
    nietz_rule = {"rule": nietzschean_optimal, "name": "Nietzschean"}

    socialwelfare_rules = [utilitarian_rule, nietz_rule, rawlsian_rule, nash_rule]
    results = load("results/random_sampling.pkl")
    results_data = load("results/sushi_data.pkl")

    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            plot_distortions(
                results[format_key(voting_rule["name"], sw["name"])],
                f"test -- Distortion of {voting_rule['name'] }, {sw['name']}",
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
            f"Distortion Under {sw['name']}",
            "Number of voters",
            "Distortion",
            names,
            show=False,
        )


if __name__ == "__main__":
    main()
