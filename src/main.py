from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup
from pref_voting import voting_methods as vr
from tqdm import tqdm

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
        utility_fun,
        k=1,
        sample_size=100,
        linear_profile: None | gp.Profile = None,
    ) -> None:
        self.n: int = n
        self.m: int = m
        self.rule: Callable = rule
        self.utility_fun: Callable = utility_fun
        self.optimal_rule: Callable = lambda x: np.argmax(self.utility_fun(x))
        self.k: int = k * m

        # Allowing for the sampling of the larger profile.
        self.sample_size = sample_size
        self.sample_linear_profiles = None
        self.sample_utils = None
        self.sample_utility_profiles = None
        self.utils: np.ndarray = np.ndarray([])

        if linear_profile is not None:
            self.linear_profile = linear_profile
            rankings = np.array(linear_profile.rankings).T
            self.utility_profile = self.generate_random_profile_from(rankings)
        else:
            self.utility_profile = self.generate_random_profile()
            self.linear_profile = self.linear_from_utility_profile(self.utils.T)

        self.test_valid_init()

    def test_valid_init(self):
        assert (
            self.linear_profile.num_cands == self.m
            and self.linear_profile.num_voters == self.n
        ), f"num_candidates: {self.linear_profile.num_cands}, m: {self.m}, num_voters: {self.linear_profile.num_voters}, n: {self.n}"
        assert (
            self.utility_profile.num_cands == self.m
            and self.utility_profile.num_voters == self.n
        ), f"num_candidates: {self.utility_profile.num_cands}, m: {self.m}, num_voters: {self.utility_profile.num_voters}, n: {self.n}"

    def generate_random_profile(self) -> gup.UtilityProfile:
        self.utils = np.array(
            [generate_random_sum_k_utilities(self.m, self.k) for _ in range(self.n)]
        )
        return self.update_utility_profile(self.utils.T)

    def update_utility_profile(self, utils):
        m, n = utils.shape
        uprofs = gup.UtilityProfile(
            [{cand: utils[cand][voter] for cand in range(m)} for voter in range(n)]
        )
        return uprofs

    def linear_from_utility_profile(self, utils):
        n = utils.shape[1]
        return gp.Profile([np.argsort(-utils[:, v]) for v in range(n)], [1] * n)

    def generate_random_profile_from(self, profile) -> gup.UtilityProfile:
        self.utils = np.array(
            [
                generate_random_sum_k_utilities(self.m, self.k, profile[:, i])
                for i in range(self.n)
            ]
        ).T
        uprofs = gup.UtilityProfile(
            [
                {cand: self.utils[cand][voter] for cand in range(self.m)}
                for voter in range(self.n)
            ]
        )
        return uprofs

    def gen_sample(self):
        voters = np.random.randint(0, self.n, self.sample_size)
        self.sample_utils = self.utils[:, voters]
        self.sample_utility_profiles = self.update_utility_profile(self.sample_utils)
        self.sample_linear_profiles = self.linear_from_utility_profile(
            self.sample_utils
        )

    def get_winner(self, profile) -> int:
        winner = self.rule(profile)
        assert winner < self.m, f"winner: {winner}, m: {self.m}"
        return winner

    def get_winner_opt(self, profile) -> int:
        winner = self.optimal_rule(profile)
        assert winner < self.m, f"winner: {winner}, m: {self.m}\n"
        return winner

    def get_utility(self, winners, utility) -> np.ndarray:
        return self.utility_fun(utility)[winners]

    def distortion(self, sample=False) -> float:
        if sample:
            self.gen_sample()
            profile = self.sample_linear_profiles
            util_profile = self.sample_utility_profiles
        else:
            profile = self.linear_profile
            util_profile = self.utility_profile
        assert profile is not None
        opt_winner = self.get_winner_opt(util_profile)
        rule_winner = self.get_winner(profile)
        assert (
            self.get_utility(opt_winner, util_profile).sum()
            >= self.get_utility(rule_winner, util_profile).sum()
        )

        return (
            self.get_utility(opt_winner, util_profile).sum()
            / self.get_utility(rule_winner, util_profile).sum()
            if self.get_utility(rule_winner, util_profile).sum() > 0
            else 1000_000
        )


def generate_random_sum_k_utilities(
    m: int, k: int, linear_pref: np.ndarray | None = None
):
    assert k >= m
    random_values = np.random.choice(range(1, k), m - 1, replace=False)
    random_values = np.sort(random_values)
    utilities = np.diff(np.concatenate(([0], random_values, [k])))
    assert len(utilities) == m, f"len: {len(utilities)}, m: {m}, k: {k}"
    assert utilities.sum() == k, f"sum: {utilities.sum()}, k: {k}"
    if linear_pref is not None:
        utilities = np.sort(utilities)[::-1]
        order = np.argsort(linear_pref)
        return utilities[order]
    return utilities


def trials(kwargs: dict, num_trials: int, sample=False):
    distortions = []
    for _ in range(num_trials):
        distortions.append(VotingGame(**kwargs).distortion(sample))
    return distortions


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


def evaluate_rule_on_data(
    rule: Callable,
    optimal_rule: Callable,
    num_trials: int,
    linear_profile=None,
):
    """
    Evalute the distortion of a voting rule (`rule`) against a rule that maximizes the social welfare (`optimal_rule`)

    args:
        rule: rule to be evaluated
        optimal_rule: rule that maximizes social welfare
        num_trials: The number of trials ran for each setting

    returns:
        distortions: ND-array of shape (len(n_vals), len(m_vals), num_trials) containign the distortion of each run
    """
    if linear_profile is not None:
        n, m = linear_profile.num_voters, linear_profile.num_cands
    else:
        n, m = 1, 1
    kwargs = {
        "n": n,
        "m": m,
        "k": 50,
        "rule": rule,
        "utility_fun": optimal_rule,
        "sample_size": 10,
        "linear_profile": linear_profile,
    }
    return trials(kwargs, num_trials, sample=True)


def evaluate_rule(
    rule: Callable,
    optimal_rule: Callable,
    n_vals: list[int] | range,
    m_vals: list[int] | range,
    num_trials: int,
    linear_profile=None,
):
    """
    Evalute the distortion of a voting rule (`rule`) against a rule that maximizes the social welfare (`optimal_rule`)

    args:
        rule: rule to be evaluated
        optimal_rule: rule that maximizes social welfare
        n_vals: The values for the number of voters to be tested
        m_vals: The values for the number of candidates to be tested
        num_trials: The number of trials ran for each setting

    returns:
        distortions: ND-array of shape (len(n_vals), len(m_vals), num_trials) containign the distortion of each run
    """
    distortions = np.empty((len(n_vals), len(m_vals), num_trials))
    for i, n in enumerate(n_vals):
        for j, m in enumerate(m_vals):
            kwargs = {
                "n": n,
                "m": m,
                "k": 30,
                "rule": rule,
                "utility_fun": optimal_rule,
                "sample_size": n,
                "linear_profile": None,
            }
            distortions[i, j] = trials(kwargs, num_trials)
    return distortions


def save_results(results: dict):
    df = pd.DataFrame(results)
    print(f"Saving Results: {df.head()}")
    df.to_csv("data/results.csv")
    print("Results saved to data/results.csv")


def sampling_experiment(voting_rules, socialwelfare_rules, n_vals, m_vals, n_trials):
    results = {}
    # np.random.seed(1)
    for voting_rule in tqdm(voting_rules):
        for sw in socialwelfare_rules:
            results[voting_rule["name"] + sw["name"]] = evaluate_rule(
                vr_wrapper(voting_rule["rule"]),
                sw["rule"],
                n_vals,
                m_vals,
                n_trials,
            )
    return results


def full_data_set_experiment(
    voting_rules, socialwelfare_rules, n_trials, linear_profile
):
    results = {}
    for voting_rule in voting_rules:
        for sw in socialwelfare_rules:
            results[voting_rule["name"] + sw["name"]] = evaluate_rule_on_data(
                vr_wrapper(voting_rule["rule"]), sw["rule"], n_trials, linear_profile
            )
    return results


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)
    n_trials = 100
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

    profile = gp.Profile.from_preflib("data/00014-00000001.soc")
    results = sampling_experiment(
        voting_rules, socialwelfare_rules, n_vals, m_vals, n_trials
    )
    results_data = full_data_set_experiment(
        voting_rules, socialwelfare_rules, n_trials, profile
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

    for sw in socialwelfare_rules:
        r = []
        names = []
        for voting_rule in voting_rules:
            names.append(voting_rule["name"])
            r.append(results_data[voting_rule["name"] + sw["name"]])
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
