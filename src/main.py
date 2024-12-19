import pickle
from typing import Callable

import numpy as np
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup
from pref_voting import voting_methods as vr
from pref_voting.iterative_methods import instant_runoff
from tqdm import tqdm

from utility_functions import (
    anti_plurality,
    nash_optimal,
    nietzschean_optimal,
    rawlsian_optimal,
    utilitarian_optimal,
)


def vr_wrapper(vr_rule):
    def rule(x):
        return vr_rule(x)[0]

    return rule


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

    def get_winner(self, profile: gp.Profile) -> int:
        assert isinstance(profile, gp.Profile)
        winner = self.rule(profile)
        assert winner < self.m, f"winner: {winner}, m: {self.m}"
        return winner

    def get_winner_opt(self, profile: gup.UtilityProfile) -> int:
        winner = self.optimal_rule(profile)
        assert winner < self.m, f"winner: {winner}, m: {self.m}\n"
        return winner

    def get_utility(self, winner: int, utility: gup.UtilityProfile) -> np.ndarray:
        return self.utility_fun(utility)[winner]

    def distortion(self, sample=False) -> float:
        if sample:
            self.gen_sample()
            profile = self.sample_linear_profiles
            util_profile = self.sample_utility_profiles
        else:
            profile = self.linear_profile
            util_profile = self.utility_profile
        assert profile is not None
        assert util_profile is not None
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


def evaluate_rule_on_data(
    rule: Callable,
    optimal_rule: Callable,
    num_trials: int,
    sample_size: int,
    k: int,
    linear_profile: gp.Profile,
    sample,
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
    n, m = linear_profile.num_voters, linear_profile.num_cands
    kwargs = {
        "n": n,
        "m": m,
        "k": k,
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
    k: int,
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
                "k": k,
                "rule": rule,
                "utility_fun": optimal_rule,
                "sample_size": n,
                "linear_profile": None,
            }
            distortions[i, j] = trials(kwargs, num_trials)
    return distortions


def format_key(vr, sw):
    return vr + ", " + sw


def sampling_experiment(
    voting_rules, socialwelfare_rules, n_vals, m_vals, n_trials, k, profile
):
    results = {}
    # np.random.seed(1)
    for voting_rule in tqdm(voting_rules):
        for sw in socialwelfare_rules:
            results[format_key(voting_rule["name"], sw["name"])] = evaluate_rule(
                vr_wrapper(voting_rule["rule"]),
                sw["rule"],
                n_vals,
                m_vals,
                n_trials,
                k,
                profile,
            )
    return results


def full_data_set_experiment(
    voting_rules,
    socialwelfare_rules,
    n_trials,
    sample_size,
    k,
    linear_profile,
    sample_from_data_set,
):
    results = {}
    for voting_rule in tqdm(voting_rules):
        for sw in socialwelfare_rules:
            results[format_key(voting_rule["name"], sw["name"])] = (
                evaluate_rule_on_data(
                    vr_wrapper(voting_rule["rule"]),
                    sw["rule"],
                    n_trials,
                    sample_size,
                    k,
                    linear_profile,
                    sample_from_data_set,
                )
            )
    return results


def save_data(results: dict, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"resutls saved as {filename}")


def gen_vr_list():
    """function so that all other files can simply import this function"""

    borda_rule = {"rule": vr.borda, "name": "Borda rule"}
    veto_rule = {"rule": vr.plurality_veto, "name": "Veto - Plurality rule"}
    # copeland_rule = {"rule": vr.copeland, "name": "Copeland's Rule"}
    plurality_rule = {"rule": vr.plurality, "name": "Plurality rule"}
    blacks_rule = {"rule": vr.blacks, "name": "Black's Rule"}
    ir_rule = {"rule": instant_runoff, "name": "Instand-runoff voting"}
    anti_rule = {"rule": anti_plurality, "name": "Inverse-Plurality"}

    return [borda_rule, veto_rule, plurality_rule, blacks_rule, ir_rule, anti_rule]
    # return [anti_rule]


def gen_ut_list():
    # Utility function to test on each voting rule
    nash_rule = {"rule": nash_optimal, "name": "Nash"}
    utilitarian_rule = {"rule": utilitarian_optimal, "name": "Utilitarian"}
    rawlsian_rule = {"rule": rawlsian_optimal, "name": "Rawlsian"}
    nietz_rule = {"rule": nietzschean_optimal, "name": "Nietzschean"}

    return [utilitarian_rule, nietz_rule, rawlsian_rule, nash_rule]


def main():
    # Experimental parameters
    np.random.seed(2)
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)
    n_trials = 10
    sample_size = 5000
    k = 15
    voting_rules = gen_vr_list()
    socialwelfare_rules = gen_ut_list()

    profile = gp.Profile.from_preflib("data/00014-00000001.soc")
    # results = sampling_experiment(
    #     voting_rules, socialwelfare_rules, n_vals, m_vals, n_trials, k, profile
    # )
    # save_data(results, f"results/random_sampling_k_{k}.pkl")

    results_data = full_data_set_experiment(
        voting_rules,
        socialwelfare_rules,
        n_trials,
        sample_size,
        k,
        profile,
        sample_from_data_set=False,
    )
    save_data(results_data, f"results/sushi_data_k_{k}.pkl")


if __name__ == "__main__":
    main()
