import numpy as np
from main import VotingGame, generate_random_sum_k_utilities, vr_wrapper
from utility_functions import *
import pytest
from pref_voting import generate_profiles as gp
from pref_voting import generate_utility_profiles as gup
from pref_voting import voting_methods as vr
import matplotlib.pyplot as plt

def test_distortion_calculation():
    linear_profile = gp.Profile(
        [
            [0,1],
            [1,0],
            [0,1],
        ]
    )
    kwargs = {
        "n": 3,
        "m": 2,
        "k": 50,
        "rule": vr_wrapper(vr.borda),
        "utility_fun": utilitarian_optimal,
        "sample_size": 10,
        "linear_profile": linear_profile,
    }
    vg = VotingGame(**kwargs)
    assert vg.get_winner(linear_profile) == 0
    assert vg.get_winner_opt(vg.utility_profile) == 0
    assert vg.distortion(sample=False) == 1

def test_vr_wrapper():
    for _ in range(100):
        linear_profile = gp.Profile(np.random.random((4,4)).argsort(axis=-1) )
        rule = vr_wrapper(vr.borda)
        winner = rule(linear_profile)
        assert 0 <= winner < 4
