import numpy as np


def borda(profile):
    return np.sum(profile, axis=0).argmax()


def plurality(profile):
    pass


def veto(profile):
    pass


def single_transverable_vote(profile): pass


def utilitarian_optimal(profile):
    """
    Returns the optimal winner, such that the sum of the utilities of the voters is maximized.
    """
    return np.argmax(np.sum(profile, axis=0))


def sen_optimal(profile):
    """
    Returns the optimal winner, such that the sum of utilities and the gini-index are maximized
    """
    pass


def nietzschean_optimal(profile):
    """
    Returns the optimal winner, such that the utility of any individual voter is maximized
    """
    return np.argmax(np.max(profile, axis=0))
