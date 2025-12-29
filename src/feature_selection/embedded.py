import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import FeatureSelector

class EmbeddedSelector(FeatureSelector):
    def __init__(self, task: str):
        self.task = task

    def select(self, X, y, k):
        model = (
            RandomForestRegressor(n_estimators=100, random_state=42)
            if self.task == "regression"
            else RandomForestClassifier(n_estimators=100, random_state=42)
        )

        model.fit(X, y)
        importances = model.feature_importances_
        idx = np.argsort(importances)[-k:]

        fs_df = pd.DataFrame({
            "Feature": X.columns[idx],
            "Importance": importances[idx]
        }).sort_values(by="Importance", ascending=False)

        return X.iloc[:, idx], fs_df
