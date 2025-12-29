from sklearn.dummy import DummyClassifier
from aeon.classification.convolution_based import (
    RocketClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    Arsenal
)

def get_classifiers():
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Rocket": RocketClassifier(n_kernels=500),
        "MiniRocket": MiniRocketClassifier(n_kernels=500),
        "MultiRocket": MultiRocketClassifier(n_kernels=500),
        "Arsenal": Arsenal(n_kernels=100, n_estimators=5)
    }
