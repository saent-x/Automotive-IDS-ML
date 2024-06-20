import numpy
import pandas as pd
import matplotlib.pyplot as plt
from py_pkg.entities.entities import SplitDataset, TestDataType, AttackType

# import datasets
dataset_dir = './datasets/clean-data/updated_dataset.csv'


# separate features from target variables
def read_dataset_and_split(filename: str) -> SplitDataset:
    df = pd.read_csv(filename)
    df_copy = df

    x_features_df = df_copy.drop(columns=["attack_0", "attack_1", "attack_2", "attack_3", "attack_4", "attack_5"])
    y_target_df = df_copy.drop(columns=["timestamp", "arbitration_id", "data_field"])

    # correct column header datatypes
    x_features_df["data_field"] = x_features_df["data_field"].astype(str)
    x_features_df["arbitration_id"] = x_features_df["arbitration_id"].astype(str)

    return SplitDataset(x_features_df, y_target_df)


# import test dataset and return as tuple
def get_test_dataset(test_type: TestDataType, attack_type: AttackType) -> pd.DataFrame:
    df = pd.read_csv(test_type.value + attack_type.value)

    return df
