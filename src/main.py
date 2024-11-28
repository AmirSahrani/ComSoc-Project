import matplotlib.pyplot as plt
import numpy as np
from pref_voting.c1_methods import copeland
from prefsampling.point.ball import Callable
from voting_rules import utilitarian_optimal, nietzschean_optimal, rawlsian_optimal
from pref_voting import voting_methods as vr
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup

def vr_wrapper(vr_rule):
    def rule(x):
        return vr_rule(x)[0] -1
    return rule

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1,
    'grid.linewidth': 1,
    'grid.alpha': 0.3,
    'image.cmap': 'viridis',
    'text.usetex': True,
    'font.family': 'Computer Modern'
})


class VotingGame():
    def __init__(self, n, m, rule, optimal_rule, k=None, min_util=0, max_util=1) -> None:
        self.n: int = n
        self.m: int = m
        self.max_util: float = max_util
        self.min_util: float = min_util
        self.rule: Callable = rule
        self.optimal_rule: Callable = optimal_rule
        self.k: float | None = k
        self.utils: np.ndarray = np.ndarray([])
        self.utility_profile  = self.generate_random_profile()
        self.linear_profile = gp.Profile([np.argsort(-self.utils[v]) for v in range(self.n)], [1] * self.n)

        # Assertions to ensure profiles have been properly generated
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

    def get_winner(self) -> int:
        winner = self.rule(self.linear_profile)
        assert winner < self.m, f'winner: {winner}, m: {self.m}\nprofile: {self.linear_profile}'
        return winner

    def get_winner_opt(self) -> int:
        winner = self.optimal_rule(self.utility_profile)
        assert winner < self.m, f'winner: {winner}, m: {self.m}\nprofile: {self.utility_profile.display()}'
        return winner

    def get_utility(self, winners) -> np.ndarray:
        return self.utils[:, winners]

    def distortion(self) -> float:
        rule_winner = self.get_winner()
        opt_winner = self.get_winner_opt()
        return self.get_utility(opt_winner).sum() /self.get_utility(rule_winner).sum() if self.get_utility(rule_winner).sum() > 0 else np.inf


def trails(kwargs, num_trails):
    distortions = []
    for _ in range(num_trails):
        distortions.append(VotingGame(**kwargs).distortion())
    return distortions


def plot_distortions(distortions, title, xlabel, ylabel, n_vals, m_vals):
    var_distortions = np.var(distortions, axis=2)
    mean_distortions = np.mean(distortions, axis=2)
    limit = mean_distortions[-3:, :].mean()


    plt.figure()
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, mean_distortions.shape[1]))

    for m in range(mean_distortions.shape[1]):
        plt.plot(n_vals, mean_distortions[:, m], color=colors[m], label=f'm={m_vals[m]}')
        plt.fill_between(n_vals,
                        mean_distortions[:, m] - var_distortions[:, m],
                        mean_distortions[:, m] + var_distortions[:, m],
                        color=colors[m], alpha=0.2)
    plt.axhline(limit, color='gray', linestyle='--', alpha=0.5)
    # plt.text(n_vals[-5], limit + 0.5, f'Distortion converging to {limit:.2f}', color='Black', fontsize=12)
    plt.title(title)
    plt.ylim((0, 20))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{title}.png')
    plt.show()


def evaluate_rule(rule, optimal_rule, n_vals, m_vals, num_trails):
    '''
    Evalute the distortion of a voting rule (`rule`) against a rule that maximizes the social welfare (`optimal_rule`)

    args:
        rule: rule to be evaluated
        optimal_rule: rule that maximizes social welfare
        n_vals: The values for the number of voters to be tested
        m_vals: The values for the number of candidates to be tested
        num_trails: The number of trials ran for each setting

    returns:
        distortions: ND-array of shape (len(n_vals), len(m_vals), num_trails) containign the distortion of each run
    '''
    distortions = np.empty((len(n_vals), len(m_vals), num_trails))
    for i,n in enumerate(n_vals):
        for j,m in enumerate(m_vals):
            kwargs = {
                "n": m,
                "m": m,
                'k': 1,
                'rule': rule,
                'optimal_rule': optimal_rule,
                'min_util': 0,
                'max_util': 1,
            }
            distortions[i,j] = trails(kwargs, num_trails)
    return distortions


def main():
    n_vals = range(2, 100, 5)
    m_vals = range(2, 25, 5)
    n_trails = 300
    borda_distortion_utilitarian = evaluate_rule(vr_wrapper(vr.borda), utilitarian_optimal, n_vals, m_vals, n_trails)
    plurarity_distortion_utilitarian = evaluate_rule(vr_wrapper(vr.plurality), utilitarian_optimal, n_vals, m_vals, n_trails)
    copeland_distortion_utilitarian = evaluate_rule(vr_wrapper(vr.copeland), utilitarian_optimal, n_vals, m_vals, n_trails)
    black_distortion_utilitarian = evaluate_rule(vr_wrapper(vr.blacks), utilitarian_optimal, n_vals, m_vals, n_trails)

    borda_distortion_nietz = evaluate_rule(vr_wrapper(vr.borda), nietzschean_optimal , n_vals, m_vals, n_trails)
    plurarity_distortion_nietz = evaluate_rule(vr_wrapper(vr.plurality), nietzschean_optimal, n_vals, m_vals, n_trails)
    copeland_distortion_nietz = evaluate_rule(vr_wrapper(vr.copeland), nietzschean_optimal, n_vals, m_vals, n_trails)
    black_distortion_nietz = evaluate_rule(vr_wrapper(vr.blacks), nietzschean_optimal, n_vals, m_vals, n_trails)

    borda_distortion_rawlsian= evaluate_rule(vr_wrapper(vr.borda), rawlsian_optimal, n_vals, m_vals, n_trails)
    plurarity_distortion_rawlsian = evaluate_rule(vr_wrapper(vr.plurality), rawlsian_optimal, n_vals, m_vals, n_trails)
    copeland_distortion_rawlsian = evaluate_rule(vr_wrapper(vr.copeland), rawlsian_optimal, n_vals, m_vals, n_trails)
    black_distortion_rawlsian = evaluate_rule(vr_wrapper(vr.blacks), rawlsian_optimal, n_vals, m_vals, n_trails)

    plot_distortions(borda_distortion_rawlsian, 'Distortion of the Borda rule, rawlsian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(plurarity_distortion_rawlsian, 'Distortion of the Plurality rule, rawlsian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(copeland_distortion_rawlsian, 'Distortion of Copelands\' rule, rawlsian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(black_distortion_rawlsian, 'Distortion of Blacks\' rule, rawlsian', 'Number of voters', 'Distortion', n_vals, m_vals)

    plot_distortions(borda_distortion_nietz, 'Distortion of the Borda rule, Nietzschean', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(plurarity_distortion_nietz, 'Distortion of the Plurality rule, Nietzschean', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(copeland_distortion_nietz, 'Distortion of Copelands\' rule, Nietzschean', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(black_distortion_nietz, 'Distortion of Blacks\' rule, Nietzschean', 'Number of voters', 'Distortion', n_vals, m_vals)

    plot_distortions(borda_distortion_utilitarian, 'Distortion of the Borda rule, Utilitarian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(plurarity_distortion_utilitarian, 'Distortion of the Plurality rule, Utilitarian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(copeland_distortion_utilitarian, 'Distortion of Copeland\'s rule, Utilitarian', 'Number of voters', 'Distortion', n_vals, m_vals)
    plot_distortions(black_distortion_utilitarian, 'Distortion of Black\'s rule, Utilitarian', 'Number of voters', 'Distortion', n_vals, m_vals)

if __name__ == "__main__":
    main()
