import numpy as np
from pref_voting import generate_utility_profiles as gup
from pref_voting import generate_profiles as gp
from pref_voting import voting_methods as vr


def anti_plurality(profile):
    rank = profile._rankings
    rank = rank[..., ::-1]
    return vr.plurality(gp.Profile(rank))



def utilities_to_np(utility_profile: gup.UtilityProfile) -> np.ndarray:
    profile_as_dict = utility_profile.as_dict()["utilities"]
    utilities = []
    for utl in profile_as_dict:
        utilities.append(list(utl.values()))
    return np.array(utilities)


def utilitarian_optimal(profile):
    """
    Returns the optimal winner, such that the sum of the utilities of the voters is maximized.
    """
    profile = utilities_to_np(profile)
    sw = profile.sum(axis=0)
    return sw


def nash_optimal(profile):
    """
    Returns the optimal winner, such that the sum of the utilities of the voters is maximized.
    """
    profile = utilities_to_np(profile) + np.exp(-6)
    # profile = utilities_to_np(profile)
    m = profile.shape[0]
    sw = np.power(profile, 1 / m).prod(axis=0)
    return sw


def nietzschean_optimal(profile):
    """
    Returns the optimal winner, such that the utility of least happy voter is maximized
    """
    profile = utilities_to_np(profile)
    return np.max(profile, axis=0)


def rawlsian_optimal(profile):
    """
    Returns the optimal winner, such that the utility of any individual voter is maximized
    """
    sw = np.min(utilities_to_np(profile), axis=0)
    return sw
