import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numerics
    scaler = MinMaxScaler()
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X
