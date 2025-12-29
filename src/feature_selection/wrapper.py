import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import FeatureSelector

class WrapperSelector(FeatureSelector):
    def __init__(self, task: str):
        self.task = task

    def select(self, X, y, k):
        estimator = (
            RandomForestRegressor(n_estimators=100, random_state=42)
            if self.task == "regression"
            else RandomForestClassifier(n_estimators=100, random_state=42)
        )

        rfe = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
        X_new = rfe.fit_transform(X, y)

        fs_df = pd.DataFrame({
            "Feature": X.columns[rfe.support_]
        })

        return X_new, fs_df
