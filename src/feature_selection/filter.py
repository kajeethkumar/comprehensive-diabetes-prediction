import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from .base import FeatureSelector

class FilterSelector(FeatureSelector):
    def select(self, X, y, k):
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        selected = X.columns[selector.get_support()]
        scores = selector.scores_[selector.get_support()]

        fs_df = pd.DataFrame({
            "Feature": selected,
            "F_score": scores
        }).sort_values(by="F_score", ascending=False)

        return X_new, fs_df
