from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

@dataclass
class Split:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list

def load_data(seed: int = 0) -> Split:
    ds = load_breast_cancer()
    X = ds.data
    y = ds.target
    feature_names = list(ds.feature_names)

    # Train+Temp split, then temp -> val+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return Split(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)