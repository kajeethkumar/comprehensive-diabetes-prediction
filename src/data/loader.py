import pandas as pd

DROP_COLS = [
    "diagnosed_diabetes",
    "diabetes_stage",
    "diabetes_risk_score"
] # Columns to drop from the dataset when loading

# include your dataset fetures here

def load_dataset(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=DROP_COLS)
    y = df[target]
    return X, y
