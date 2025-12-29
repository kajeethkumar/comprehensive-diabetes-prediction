from aeon.regression.convolution_based import (
    RocketRegressor,
    MiniRocketRegressor,
    MultiRocketRegressor
)
from aeon.regression.sklearn import RotationForestRegressor

def get_regressors():
    return {
        "Rocket": RocketRegressor(n_kernels=500),
        "MiniRocket": MiniRocketRegressor(n_kernels=500),
        "MultiRocket": MultiRocketRegressor(n_kernels=500),
        "RotationForest": RotationForestRegressor(n_estimators=10)
    }
