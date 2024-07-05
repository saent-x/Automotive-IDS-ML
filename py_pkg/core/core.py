import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report

from py_pkg.entities.entities import AttackType, SplitDataset, TestDataType

# import datasets
dataset_dir = './datasets/clean-data/updated_dataset.csv'


# separate features from target variables
def read_dataset_and_split(filename: str) -> SplitDataset:
    df = pd.read_csv(filename)
    df_copy = df

    x_features_df = df_copy.drop(columns=["attack"])
    y_target_df = df_copy.drop(columns=["timestamp", "arbitration_id", "data_field"])

    # correct column header datatypes
    x_features_df["data_field"] = x_features_df["data_field"].astype(str)
    x_features_df["arbitration_id"] = x_features_df["arbitration_id"].astype(str)

    return SplitDataset(x_features_df, y_target_df)


# import test dataset and return as tuple
def get_test_dataset(test_type: TestDataType, attack_type: AttackType) -> pd.DataFrame:
    df = pd.read_csv(test_type.value + attack_type.value)

    return df

def generate_metrics_report(y_true, y_pred, pos_label):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted',  zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label) # predicted label
    roc_auc = auc(fpr, tpr)

    target_names = ['dos', 'force neutral', 'rpm', 'standstill']

    return (classification_report(y_true, y_pred, zero_division=1, target_names=target_names), pd.DataFrame(
        {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1],
            'ROC AUC': [roc_auc],
            'FPR': [fpr],
            'TPR': [tpr]
        }
    ), cm)


def get_human_time(close_time: float, start_time: float) -> str:
    total_time = close_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    return f"{minutes:02}:{seconds:02}"

