import os
import pickle
import bz2

import pystan
from pystan._compat import PY2


def get_bz2_open():
    if PY2:
        bz2_open = bz2.BZ2File
    else:
        bz2_open = bz2.open
    return bz2_open

def load_model(path):
    _, ext = os.path.splitext(path)
    if ext != '.bz2':
        path = path + ".bz2"
    bz2_open = get_bz2_open()
    with bz2_open(path, "rb") as f:
        pickled_model = f.read()
    stan_model = pickle.loads(pickled_model)
    return stan_model

def save_model(stan_model, path):
    pickled_model = pickle.dumps(stan_model, protocol=pickle.HIGHEST_PROTOCOL)
    _, ext = os.path.splitext(path)
    if ext != '.bz2':
        path = path + ".bz2"
    bz2_open = get_bz2_open()
    with bz2_open(path, "wb") as f:
        f.write(pickled_model)

def get_model(filename, model_code, **kwargs):
    root = os.path.join(os.path.dirname(__file__), 'cached_models')
    # py27 does not have 'exist_ok'
    try:
        os.makedirs(root)
    except OSError:
        pass
    path = os.path.join(root, filename)
    # use .bz2 compression
    _, ext = os.path.splitext(path)
    if ext != '.bz2':
        path = path + ".bz2"
    if os.path.exists(path):
        stan_model = load_model(path)
    else:
        stan_model = pystan.StanModel(model_code=model_code, **kwargs)
        save_model(stan_model, path)
    return stan_model
