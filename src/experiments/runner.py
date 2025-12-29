from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.data.preprocessing import preprocess_features
from src.feature_selection.filter import FilterSelector
from src.feature_selection.wrapper import WrapperSelector
from src.feature_selection.embedded import EmbeddedSelector
from src.models.classifiers import get_classifiers
from src.models.regressors import get_regressors
from src.utils.evaluator import evaluate
from src.utils.io import ensure_dir
from src.utils.config import load_config


def run_from_yaml(config_path: str):
    cfg = load_config(config_path)

    task = cfg["task"]
    target = cfg["target"]
    csv_path = cfg["csv_path"]
    output_dir = cfg["output_dir"]

    selectors = cfg["feature_selection"]["methods"]
    k_values = cfg["feature_selection"]["k_values"]

    split_cfg = cfg["split"]
    test_size = split_cfg["test_size"]
    random_state = split_cfg["random_state"]
    stratify = split_cfg["stratify"]

    ensure_dir(output_dir)

    X, y = load_dataset(csv_path, target)
    X = preprocess_features(X)

    for k in k_values:
        for sel in selectors:

            selector = (
                FilterSelector() if sel == "filter"
                else WrapperSelector(task) if sel == "wrapper"
                else EmbeddedSelector(task)
            )

            X_sel, fs_df = selector.select(X, y, k)
            fs_df.to_csv(
                f"{output_dir}/{sel}_features_k{k}.csv",
                index=False
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X_sel,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if stratify and task != "regression" else None
            )

            models = (
                get_regressors()
                if task == "regression"
                else get_classifiers()
            )

            # ✅ CORRECT CALL
            results = evaluate(
                models,
                X_train,
                X_test,
                y_train,
                y_test,
                task
            )

            results["k"] = k
            results["selector"] = sel

            results.to_csv(
                f"{output_dir}/{task}_{sel}_results_k{k}.csv",
                index=False
            )
