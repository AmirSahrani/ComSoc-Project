import matplotlib.pyplot as plt
import numpy as np
# from scipy import optimize
from voting_rules import utilitarian_optimal
from pref_voting import voting_methods as vr
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup


class VotingGame():
    def __init__(self, n, m, k=None, min_util=0, max_util=1) -> None:
        self.n = n
        self.m = m
        self.max_util = max_util
        self.min_util = min_util
        self.k = k
        self.utility_profile = self.generate_random_profile()
        self.linear_profile = self.utility_profile.to_ranking_profile()

    def generate_random_profile(self) -> gup.UtilityProfile:

        utils = np.random.uniform(size=(self.n, self.m))
        if self.k:
            for voter in range(self.n):
                utils[voter, :] *= self.k/np.sum(utils[voter, :])
        uprofs = gup.UtilityProfile([{c: utils[v][c]
                                for c in range(self.n)}
                                for v in range(self.m)])
        return uprofs

    def linear_order(self) -> np.ndarray:
        return np.argsort(self.utility_profile, axis=1)


    def get_winner(self, rule) -> int: return rule(self.linear_profile)
    def get_winner_opt(self, rule) -> int: return rule(self.utility_profile)

    def get_utility(self, rule) -> np.ndarray:
        return self.utility_profile[:, self.get_winner(rule)]

    def get_utility_opt(self, rule) -> np.ndarray:
        return self.utility_profile[:, self.get_winner_opt(rule)]

    def distortion(self, rule, optimal_rule) -> float:
        return self.get_utility_opt(optimal_rule).sum() /self.get_utility(rule).sum() if self.get_utility(rule).sum() > 0 else np.inf


def main():
    votingGame = VotingGame(10, 100, 0, 100)
    print(votingGame.utility_profile.write())
    print(votingGame.linear_profile.display())
    # distortion = votingGame.distortion(vr.borda, utilitarian_optimal)
    # print(distortion)


if __name__ == "__main__":
    main()

    # range_n = range(10, 30, 5)
    # range_m = np.logspace(3, 5, 10)
    # trials = 30
    # data = np.zeros((len(range_n), len(range_m)))
    # for i,n in enumerate(range_n):
    #     for j,m in enumerate(range_m):
    #         distortion = 0
    #         for _ in range(trials):
    #             votingGame = VotingGame(int(n), int(m), 0, 100)
    #             distortion += votingGame.distortion(vr.borda, vr.utilitarian_optimal)
    #         data[i,j] = distortion / trials

    # plt.imshow(data, cmap='hot', interpolation='nearest')
    # plt.yticks(range(len(range_n)), [f'{x}' for x  in range_n])
    # plt.xticks(range(len(range_m)), [f'{x:.2e}' for x  in range_m])
    # plt.colorbar()
    # plt.show()
