import numpy as np
import pandas as pd

def to_aeon_3d(X):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return np.expand_dims(X, axis=1)
