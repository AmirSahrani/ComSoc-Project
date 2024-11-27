import numpy as np
import pytest
import voting_rules as vr

def test_borda():
    profile = np.array([[1, 2, 3],
                        [3, 2, 1]])
    assert vr.borda(profile) == 1
