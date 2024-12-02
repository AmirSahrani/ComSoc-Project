import numpy as np
from pref_voting import generate_utility_profiles as gup


def utilities_to_np(utility_profile: gup.UtilityProfile) -> np.ndarray:
    profile_as_dict = utility_profile.as_dict()['utilities']
    utilities = []
    for utl in profile_as_dict:
        utilities.append(list(utl.values()))
    return np.array(utilities)

def utilitarian_optimal(profile):
    """
    Returns the optimal winner, such that the sum of the utilities of the voters is maximized.
    """
    return utilities_to_np(profile).sum(axis=0).argmax()

def nash_optimal(profile):
    """
    Returns the optimal winner, such that the sum of the utilities of the voters is maximized.
    """
    profile = utilities_to_np(profile)
    m = profile.shape[1]
    return np.pow(profile, 1/m).prod(axis=0).argmax()

def nietzschean_optimal(profile):
    """
    Returns the optimal winner, such that the utility of least happy voter is maximized
    """
    return np.argmax(np.max(utilities_to_np(profile), axis=0))

def rawlsian_optimal(profile):
    """
    Returns the optimal winner, such that the utility of any individual voter is maximized
    """
    return np.argmax(np.min(utilities_to_np(profile), axis=0))
