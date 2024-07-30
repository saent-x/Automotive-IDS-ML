# Data Processing
import time
from typing import Any
import matplotlib
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from py_pkg.core.core import get_human_time
from py_pkg.entities.entities import AlgoToPredict, SplitDataset


class Algo:
    __rf: RandomForestClassifier
    __xgb: GradientBoostingClassifier
    __kmeans: KMeans
    __ensemble_model: VotingClassifier

    def __init__(self, split_dataset: SplitDataset):
        self.split_dataset = split_dataset

    def impl_random_forests(self):
        # Initiate and train model
        start_time = time.time()

        self.__rf = RandomForestClassifier(random_state=42, n_jobs=-1) # TODO: change random state
        self.__rf.fit(self.split_dataset.x_features, self.split_dataset.y_features.values.ravel())

        close_time = time.time()
        print(f"==> ml-algo [Random Forests Implemented in {get_human_time(close_time, start_time)}]")


    def impl_xgboost(self):
        # Initiate and train model
        start_time = time.time()

        self.__xgb = GradientBoostingClassifier()
        y_features = np.ravel(self.split_dataset.y_features)
        self.__xgb.fit(self.split_dataset.x_features, y_features)

        close_time = time.time()
        print(f"==> ml-algo [Gradient Boosting Implemented in {get_human_time(close_time, start_time)}]")

    def impl_kmeans(self):
        # create scaled DataFrame where each variable has mean of 0 and standard dev of 1
        df = pd.concat([self.split_dataset.x_features, self.split_dataset.y_features], axis=1)

        # scaled_df = StandardScaler().fit_transform(df)
        start_time = time.time()

        self.__kmeans = KMeans(n_clusters=2, init="random", n_init='auto', random_state=1)
        self.__kmeans.fit(df)

        close_time = time.time()
        print(f"==> ml-algo [KMeans Implemented in {get_human_time(close_time, start_time)}]")

    def predict(self, test_data: pd.DataFrame, algo: AlgoToPredict, attack: str):
        match algo:
            case algo.random_forest:
                if self.__rf != None:
                    print(f"==> ml-algo [Predicting for Random Forests - {attack}]")
                    return self.__rf.predict(test_data)
            case algo.xgboost:
                if self.__xgb != None:
                    print(f"==> ml-algo [Predicting for Extreme Gradient Boosting - {attack}]")
                    return self.__xgb.predict(test_data)
            # case algo.ensemble:
            #     if self.__ensemble_model != None:
            #         print("==> ml-algo [Predicting for Ensemble Model]")
            #         return self.__ensemble_model.predict(test_data)
            case algo.k_means:
                if self.__kmeans != None:
                    print(f"==> ml-algo [Predicting for KMeans - {attack}]")
                    return self.__kmeans.predict(test_data)
            case _:
                return np.empty((0,0))

        return np.empty((0,0))

    def algo_voting_classifier(self):
        if self.__rf or self.__xgb or self.__kmeans != None:
            self.__ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', self.__rf),
                    ('xgb', self.__xgb),
                    ('kmeans', self.__kmeans)
                ], voting='hard'
            )

            self.__ensemble_model.fit(self.split_dataset.x_features, self.split_dataset.y_features)

            print("==> ml-algo [Ensemble Model Implemented]")