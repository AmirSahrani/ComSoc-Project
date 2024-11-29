from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup
from pref_voting import voting_methods as vr

from voting_rules import (
    nash_optimal,
    nietzschean_optimal,
    rawlsian_optimal,
    utilitarian_optimal,
)


def vr_wrapper(vr_rule):
    def rule(x):
        return vr_rule(x)[0] - 1

    return rule


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1,
        "grid.linewidth": 1,
        "grid.alpha": 0.3,
        "image.cmap": "viridis",
        "text.usetex": True,
        "font.family": "Computer Modern",
    }
)


class VotingGame:
    def __init__(self, n, m, rule, optimal_rule, k=1, min_util=0, max_util=1) -> None:
        self.n: int = n
        self.m: int = m
        self.max_util: float = max_util
        self.min_util: float = min_util
        self.rule: Callable = rule
        self.optimal_rule: Callable = optimal_rule
        self.k: int = k * m
        self.utils: np.ndarray = np.ndarray([])
        self.utility_profile = self.generate_random_profile()
        self.linear_profile = gp.Profile(
            [np.argsort(-self.utils[v]) for v in range(self.n)], [1] * self.n
        )

        # Assertions to ensure profiles have been properly generated
        assert (
            self.linear_profile.num_cands == self.m
        ), f"num_candidates: {self.linear_profile.num_cands}, m: {self.m}\n profile: {self.linear_profile}"
        assert (
            self.linear_profile.num_voters == self.n
        ), f"num_voters: {self.linear_profile.num_voters}, n: {self.n}"
        assert (
            self.utility_profile.num_cands == self.m
        ), f"num_candidates: {self.utility_profile.num_cands}, m: {self.m}\n profile: {self.utility_profile.display()}"
        assert (
            self.utility_profile.num_voters == self.n
        ), f"num_voters: {self.utility_profile.num_voters}, n: {self.n}"

    def generate_random_profile(self) -> gup.UtilityProfile:
        utils = np.array(
            [generate_random_sum_k_utilities(self.m, self.k) for _ in range(self.n)]
        ).T
        self.utils = utils.T
        uprofs = gup.UtilityProfile(
            [
                {cand: utils[cand][voter] for cand in range(self.m)}
                for voter in range(self.n)
            ]
        )
        return uprofs

    def get_winner(self) -> int:
        winner = self.rule(self.linear_profile)
        assert (
            winner < self.m
        ), f"winner: {winner}, m: {self.m}\nprofile: {self.linear_profile}"
        return winner

    def get_winner_opt(self) -> int:
        winner = self.optimal_rule(self.utility_profile)
        assert (
            winner < self.m
        ), f"winner: {winner}, m: {self.m}\nprofile: {self.utility_profile.display()}"
        return winner

    def get_utility(self, winners) -> np.ndarray:
        return self.utils[:, winners]

    def distortion(self) -> float:
        rule_winner = self.get_winner()
        opt_winner = self.get_winner_opt()
        return (
            self.get_utility(opt_winner).sum() / self.get_utility(rule_winner).sum()
            if self.get_utility(rule_winner).sum() > 0
            else 1000
        )


def generate_random_sum_k_utilities(m: int, k: int):
    assert k >= m
    first = np.random.randint(0, k, m - 1)
    first = np.sort(first)
    second = np.diff(np.concatenate(([0], first, [k])))
    assert len(second) == m, f"len: {len(second)}, m: {m}, k: {k}"
    assert second.sum() == k, f"sum: {second.sum()}, k: {k}"
    return second


def trails(kwargs: dict, num_trails: int):
    distortions = []
    for _ in range(num_trails):
        distortions.append(VotingGame(**kwargs).distortion())
    return distortions


def plot_distortions(
    distortions: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    n_vals: Iterable[int] | range,
    m_vals: Iterable[int] | range,
    show: bool = True,
):
    var_distortions = np.var(distortions, axis=2)
    mean_distortions = np.mean(distortions, axis=2)
    limit = mean_distortions[-3:, :].mean()

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


def evaluate_rule(
    rule: Callable,
    optimal_rule: Callable,
    n_vals: Iterable[int] | range,
    m_vals: Iterable[int] | range,
    num_trails: int,
):
    """
    Evalute the distortion of a voting rule (`rule`) against a rule that maximizes the social welfare (`optimal_rule`)

    args:
        rule: rule to be evaluated
        optimal_rule: rule that maximizes social welfare
        n_vals: The values for the number of voters to be tested
        m_vals: The values for the number of candidates to be tested
        num_trails: The number of trials ran for each setting

    returns:
        distortions: ND-array of shape (len(n_vals), len(m_vals), num_trails) containign the distortion of each run
    """
    distortions = np.empty((len(n_vals), len(m_vals), num_trails))
    for i, n in enumerate(n_vals):
        for j, m in enumerate(m_vals):
            kwargs = {
                "n": m,
                "m": m,
                "k": 3,
                "rule": rule,
                "optimal_rule": optimal_rule,
                "min_util": 0,
                "max_util": 1,
            }
            distortions[i, j] = trails(kwargs, num_trails)
    return distortions


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(5, 25, 5)
    n_trails = 1
    borda_rule = {"rule": vr.borda, "name": "the Borda rule"}
    copeland_rule = {"rule": vr.copeland, "name": "Copeland's Rule"}
    plurality_rule = {"rule": vr.plurality, "name": "the Plurality rule"}
    blacks_rule = {"rule": vr.blacks, "name": "Black's Rule"}

    voting_rules = [borda_rule, copeland_rule, plurality_rule, blacks_rule]

    nash_rule = {"rule": nash_optimal, "name": "Nash"}
    utilitarian_rule = {"rule": utilitarian_optimal, "name": "Utiliratian"}
    rawlsian_rule = {"rule": rawlsian_optimal, "name": "Rawlsian"}
    nietz_rule = {"rule": nietzschean_optimal, "name": "Nietzschean"}

    socialwelfare_rules = [nash_rule, utilitarian_rule, rawlsian_rule, nietz_rule]

    results = {}
    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            results[voting_rule["name"] + sw["name"]] = evaluate_rule(
                vr_wrapper(voting_rule["rule"]),
                sw["rule"],
                n_vals,
                m_vals,
                n_trails,
            )

    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            plot_distortions(
                results[voting_rule["name"] + sw["name"]],
                f"Distortion of {voting_rule['name'] }, {sw['name']}",
                "Number of voters",
                "Distortion",
                n_vals,
                m_vals,
                show=False,
            )


if __name__ == "__main__":
    main()
