from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    def __init__(
        self,
        n,
        m,
        rule,
        optimal_rule,
        k=1,
        min_util=0,
        max_util=1,
        sample_size=100,
        linear_profile: None | gp.Profile = None,
    ) -> None:
        self.n: int = n
        self.m: int = m
        self.max_util: float = max_util
        self.min_util: float = min_util
        self.rule: Callable = rule
        self.optimal_rule: Callable = optimal_rule
        self.k: int = k * m

        # Allowing for the sampling of the larger profile.
        self.sample_size = sample_size
        self.sample_linear_profiles = None
        self.sample_utils = None
        self.sample_utility_profiles = None

        if linear_profile is not None:
            self.utils: np.ndarray = np.ndarray([])
            self.linear_profile = linear_profile
            self.utility_profile = self.generate_random_profile_from(
                np.array(linear_profile.rankings)
            )
            self.n, self.m = self.utils.shape
        else:
            self.utils: np.ndarray = np.ndarray([])
            self.utility_profile = self.generate_random_profile()
            self.linear_profile = self.linear_from_utility_profile(self.utils)

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
        return self.update_utility_profile(self.utils)

    def update_utility_profile(self, utils):
        uprofs = gup.UtilityProfile(
            [
                {cand: self.utils[cand][voter] for cand in range(self.m)}
                for voter in range(self.n)
            ]
        )
        return uprofs

    def linear_from_utility_profile(self, utils):
        return gp.Profile([np.argsort(-utils[v]) for v in range(self.n)], [1] * self.n)

    def generate_random_profile_from(self, profile) -> gup.UtilityProfile:
        utils = np.array(
            [
                generate_random_sum_k_utilities(self.m, self.k, profile[i, :])
                for i in range(self.n)
            ]
        ).T
        self.utils = utils.T
        uprofs = gup.UtilityProfile(
            [
                {cand: utils[cand][voter] for cand in range(self.m)}
                for voter in range(self.n)
            ]
        )
        return uprofs

    def get_winner(self, profile) -> int:
        winner = self.rule(profile)
        assert winner < self.m, f"winner: {winner}, m: {self.m}\nprofile: {profile}"
        return winner

    def get_winner_opt(self, profile) -> int:
        winner = self.optimal_rule(profile)
        assert (
            winner < self.m
        ), f"winner: {winner}, m: {self.m}\nprofile: {profile.display()}"
        return winner

    def get_utility(self, winners, utility) -> np.ndarray:
        return utility[:, winners]

    def gen_sample(self):
        voters = np.random.randint(0, self.n, self.sample_size)
        self.sample_utils = self.utils[voters, :]
        self.sample_utility_profiles = self.update_utility_profile(self.sample_utils)
        self.sample_linear_profiles = self.linear_from_utility_profile(
            self.sample_utils
        )

    def distortion(self, sample=True) -> float:
        if sample:
            self.gen_sample()
            profile = self.sample_linear_profiles
            util_profile = self.sample_utility_profiles
            utilities = self.sample_utils
        else:
            profile = self.linear_profile
            util_profile = self.utility_profile
            utilities = self.utils
        assert profile is not None
        opt_winner = self.get_winner_opt(util_profile)
        rule_winner = self.get_winner(profile)
        return (
            self.get_utility(opt_winner, utilities).sum()
            / self.get_utility(rule_winner, utilities).sum()
            if self.get_utility(rule_winner, utilities).sum() > 0
            else 1000_000
        )


def generate_random_sum_k_utilities(
    m: int, k: int, linear_pref: np.ndarray | None = None
):
    assert k >= m
    random_values = np.random.randint(0, k, m - 1)
    random_values = np.sort(random_values)
    utilities = np.diff(np.concatenate(([0], random_values, [k])))
    assert len(utilities) == m, f"len: {len(utilities)}, m: {m}, k: {k}"
    assert utilities.sum() == k, f"sum: {utilities.sum()}, k: {k}"
    if linear_pref is not None:
        order = np.argsort(linear_pref)
        return utilities[order]
    return utilities


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


def evaluate_rule(
    rule: Callable,
    optimal_rule: Callable,
    n_vals: list[int] | range,
    m_vals: list[int] | range,
    num_trails: int,
    linear_profile=None,
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
                "sample_size": n,
                "linear_profile": linear_profile,
                # "linear_profile": None,
            }
            distortions[i, j] = trails(kwargs, num_trails)
    return distortions


def save_results(results: dict):
    df = pd.DataFrame(results)
    print(f"Saving Results: {df.head()}")
    df.to_csv("data/results.csv")
    print("Results saved to data/results.csv")


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(5, 25, 5)
    n_trails = 2
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
    np.random.seed(1)
    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            results[voting_rule["name"] + sw["name"]] = evaluate_rule(
                vr_wrapper(voting_rule["rule"]),
                sw["rule"],
                n_vals,
                m_vals,
                n_trails,
            )

    print(results)
    # save_results(results)

    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            plot_distortions(
                results[voting_rule["name"] + sw["name"]],
                f"test --Distortion of {voting_rule['name'] }, {sw['name']}",
                "Number of voters",
                "Distortion",
                n_vals,
                m_vals,
                show=False,
            )


if __name__ == "__main__":
    main()
