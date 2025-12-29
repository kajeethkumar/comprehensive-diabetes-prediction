import pandas as pd
from src.utils.aeon_utils import to_aeon_3d
from src.utils.metrics import (
    classification_metrics,
    regression_metrics
)

def evaluate(models, X_train, X_test, y_train, y_test, task):
    results = []

    for name, model in models.items():
        try:
            # --- Fit & Predict ---
            if task == "regression" or name == "Dummy":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(to_aeon_3d(X_train), y_train)
                y_pred = model.predict(to_aeon_3d(X_test))

            # --- Metrics ---
            if task == "regression":
                metrics = regression_metrics(
                    y_test,
                    y_pred,
                    p=X_train.shape[1]
                )
            else:
                metrics = classification_metrics(
                    y_test,
                    y_pred
                )

            results.append({
                "Model": name,
                **metrics
            })

        except Exception as e:
            results.append({
                "Model": name,
                "Error": str(e)
            })

    return pd.DataFrame(results)
