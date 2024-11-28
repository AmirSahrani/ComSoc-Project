import matplotlib.pyplot as plt
import numpy as np
from voting_rules import utilitarian_optimal, utilities_to_np
from pref_voting import voting_methods as vr
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup
import multiprocessing as mp
import decorators as dec

## TODO make decotaror for the voting rules so that you don't have to use lambda functions



plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1,
    'grid.linewidth': 1,
    'grid.alpha': 0.3,
    'image.cmap': 'viridis',
    'text.usetex': True
})



class VotingGame():
    def __init__(self, n, m, k=None, min_util=0, max_util=1) -> None:
        self.n = n
        self.m = m
        self.max_util = max_util
        self.min_util = min_util
        self.k = k
        self.utils = np.ndarray([])
        self.utility_profile = self.generate_random_profile()
        self.linear_profile = gp.Profile([np.argsort(-self.utils[v]) for v in range(self.n)], [1] * self.n)

        assert self.linear_profile.num_cands == self.m, f'num_candidates: {self.linear_profile.num_cands}, m: {self.m}\n profile: {self.linear_profile}'
        assert self.linear_profile.num_voters == self.n, f'num_voters: {self.linear_profile.num_voters}, n: {self.n}'
        assert self.utility_profile.num_cands == self.m, f'num_candidates: {self.utility_profile.num_cands}, m: {self.m}\n profile: {self.utility_profile.display()}'
        assert self.utility_profile.num_voters == self.n, f'num_voters: {self.utility_profile.num_voters}, n: {self.n}'

    def generate_random_profile(self) -> gup.UtilityProfile:
        utils = np.random.uniform(size=(self.m, self.n))
        if self.k:
            for voter in range(self.n):
                utils[:, voter] *= self.k/np.sum(utils[:, voter])
        self.utils = utils.T
        uprofs = gup.UtilityProfile([{cand: utils[cand][voter]
                                for cand in range(self.m)}
                                for voter in range(self.n)])
        return uprofs

    def get_winner(self, rule) -> int:
        winner = rule(self.linear_profile)
        assert winner < self.m, f'winner: {winner}, m: {self.m}\nprofile: {self.linear_profile}'
        return winner

    def get_winner_opt(self, rule) -> int:
        winner = rule(self.utility_profile)
        assert winner < self.m, f'winner: {winner}, m: {self.m}\nprofile: {self.utility_profile.display()}'
        return winner

    def get_utility(self, winners) -> np.ndarray:
        return self.utils[:, winners]

    def distortion(self, rule, optimal_rule) -> float:
        rule_winner = self.get_winner(rule)
        opt_winner = self.get_winner_opt(optimal_rule)
        return self.get_utility(opt_winner).sum() /self.get_utility(rule_winner).sum() if self.get_utility(rule_winner).sum() > 0 else np.inf

def trails(n, m, k, min_util, max_util, rule, optimal_rule, num_trails):
    distortions = []
    with mp.Pool(mp.cpu_count()) as pool:
        games = [VotingGame(n, m, k, min_util, max_util) for _ in range(num_trails)]
        distortions = pool.starmap(VotingGame.distortion, [(game, rule, optimal_rule) for game in games])
    return distortions


def plot_distortions(distortions, title, xlabel, ylabel, n_vals, m_vals):
    var_distortions = np.var(distortions, axis=2)
    mean_distortions = np.mean(distortions, axis=2)
    # find distortion in the limit
    limit = mean_distortions[-3:, :].mean()


    plt.figure()
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, mean_distortions.shape[1]))

    for m in range(mean_distortions.shape[1]):
        plt.plot(n_vals, mean_distortions[:, m], color=colors[m], label=f'm={m_vals[m]}')
        plt.fill_between(n_vals,
                        mean_distortions[:, m] - var_distortions[:, m],
                        mean_distortions[:, m] + var_distortions[:, m],
                        color=colors[m], alpha=0.2)
    plt.axhline(limit, color='gray', linestyle='--', alpha=0.5)
    plt.text(n_vals[-5], limit + 0.5, f'Distortion converging to {limit:.2f}', color='Black', fontsize=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../figures/{title}.png')
    plt.show()


def evaluate_rule(rule, optimal_rule, n_vals, m_vals, num_trails):
    distortions = np.empty((len(n_vals), len(m_vals), num_trails))
    for i,n in enumerate(n_vals):
        for j,m in enumerate(m_vals):
            k = 10
            min_util = 0
            max_util = 1
            distortions[i,j] = trails(n, m, k, min_util, max_util, rule, optimal_rule, num_trails)
    return distortions

def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)
    n_trails = 10
    borda_distortion = evaluate_rule(lambda x: vr.borda(x)[0] -1, utilitarian_optimal, n_vals, m_vals, n_trails)
    plot_distortions(borda_distortion, 'Distortion of the Borda rule', 'Number of voters', 'Distortion', n_vals, m_vals)
    # plurarity_distortion = evaluate_rule(lambda x: vr.plurality(x)[0] -1, utilitarian_optimal, n_vals, m_vals, n_trails)
    # plot_distortions(plurarity_distortion, 'Distortion of the Plurality rule', 'Number of voters', 'Distortion', n_vals, m_vals)





if __name__ == "__main__":
    main()
