import matplotlib.pyplot as plt
import numpy as np
from voting_rules import utilitarian_optimal, utilities_to_np
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
        self.utils = np.ndarray([])
        self.utility_profile = self.generate_random_profile()
        self.linear_profile = self.utility_profile.to_ranking_profile().to_linear_profile()

    def generate_random_profile(self) -> gup.UtilityProfile:

        utils = np.random.uniform(size=(self.m, self.n))
        if self.k:
            for voter in range(self.n):
                utils[voter, :] *= self.k/np.sum(utils[voter, :])

        self.utils = utils.T
        uprofs = gup.UtilityProfile([{c: utils[c][v]
                                for c in range(self.m)}
                                for v in range(self.n)])
        return uprofs


    def get_winner(self, rule) -> int: return rule(self.linear_profile)
    def get_winner_opt(self, rule) -> int: return rule(self.utility_profile)

    def get_utility(self, winners) -> np.ndarray:
        return self.utils[:, winners]

    def distortion(self, rule, optimal_rule) -> float:
        rule_winner = self.get_winner(rule)
        opt_winner = self.get_winner_opt(optimal_rule)
        return self.get_utility(opt_winner).sum() /self.get_utility(rule_winner).sum() if self.get_utility(rule_winner).sum() > 0 else np.inf


def main():
    range_n = range(10, 30, 5)
    range_m = range(10,300,5)
    trials = 30
    data = np.zeros((len(range_n), len(range_m)))
    for i,n in enumerate(range_n):
        for j,m in enumerate(range_m):
            distortion = 0
            for _ in range(trials):
                votingGame = VotingGame(int(n), int(m), 0, 100)
                distortion += votingGame.distortion(lambda x: vr.plurality_veto(x)[0], utilitarian_optimal)
            data[i,j] = distortion / trials

    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.yticks(range(len(range_n)), [f'{x}' for x  in range_n])
    plt.xticks(range(len(range_m)), [f'{x:.2e}' for x  in range_m])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
