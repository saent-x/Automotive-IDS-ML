import pandas as pd
from enum import Enum


class SplitDataset:

    def __init__(self, x_features: pd.DataFrame, y_features: pd.DataFrame):
        self.x_features = x_features
        self.y_features = y_features


class AlgoToPredict(str, Enum):
    random_forest = "Random Forests"
    xgboost = "Extreme Gradient Boosting"
    k_means = "K-Means"
    ensemble = "Ensemble"


class TestDataType(str, Enum):
    kv_ka = "./datasets/clean-data/test_data/kv-ka/"
    kv_ua = "./datasets/clean-data/test_data/kv-ua/"
    uv_ka = "./datasets/clean-data/test_data/uv-ka/"
    uv_ua = "./datasets/clean-data/test_data/uv-ua/"


class AttackType(str, Enum):
    dos = "DoS.csv"
    rpm = "rpm.csv"
    force_neutral = "force_neutral.csv"
    standstill = "standstill.csv"

    double = "double.csv"
    fuzzing = "fuzzing.csv"
    interval = "interval.csv"
    speed = "speed.csv"
    systematic = "systematic.csv"
    triple = "triple.csv"
