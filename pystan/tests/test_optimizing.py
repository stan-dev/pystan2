import numpy as np

from pystan import StanModel

def test_optimizing_basic():
    sm = StanModel(model_code='parameters {real y;} model {y ~ normal(0,1);}')
    op = sm.optimizing()
    assert op['par']['y'].shape == ()
    assert abs(op['par']['y']) < 1
