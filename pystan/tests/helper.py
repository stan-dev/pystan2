import os
import pickle
import gzip

import pystan

def load_model(path):
    _, ext = os.path.splitext(path)
    if ext != '.gz':
        path = ext + ".gz"
    with gzip.open(path, "rb") as f:
        pickled_model = f.read()
    stan_model = pickle.loads(pickled_model)
    return stan_model

def save_model(stan_model, path):
    pickled_model = pickle.dumps(stan_model, protocol=pickle.HIGHEST_PROTOCOL)
    _, ext = os.path.splitext(path)
    if ext != '.gz':
        path = ext + ".gz"
    with gzip.open(path, "wb") as f:
        f.write(pickled_model)

def get_model(filename, model_code, **kwargs):
    root = os.path.join(os.path.dirname(__file__), 'cached_models')
    # py27 does not have 'exist_ok'
    try:
        os.makedirs(root)
    except OSError:
        pass
    path = os.path.join(root, filename)
    if os.path.exists(path):
        stan_model = load_model(path)
    else:
        stan_model = pystan.StanModel(model_code=model_code, **kwargs)
        save_model(stan_model, path)
    return stan_model
