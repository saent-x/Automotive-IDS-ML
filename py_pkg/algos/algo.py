# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Local Modules
from py_pkg.entities.entities import SplitDataset, AlgoToPredict


class Algo:
    __rf: RandomForestClassifier
    __xgb: GradientBoostingClassifier
    __kmeans: KMeans
    __ensemble_model: VotingClassifier

    def __init__(self, split_dataset: SplitDataset):
        self.split_dataset = split_dataset

    def impl_random_forests(self):
        # Initiate and train model
        self.__rf = RandomForestClassifier()
        self.__rf.fit(self.split_dataset.x_features, self.split_dataset.y_features)

        print("==> ml-algo [Random Forests Implemented]")


    def impl_xgboost(self):
        # Initiate and train model
        self.__xgb = GradientBoostingClassifier()
        self.__xgb.fit(self.split_dataset.x_features, self.split_dataset.y_features)

        print("==> ml-algo [Gradient Boosting Implemented]")

    def impl_kmeans(self):
        # create scaled DataFrame where each variable has mean of 0 and standard dev of 1
        df = pd.concat([self.split_dataset.x_features, self.split_dataset.y_features])
        scaled_df = StandardScaler().fit_transform(df)

        print(f"==> ml-algo \n ==> {scaled_df[:10]}")

        self.__kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)
        self.__kmeans.fit(scaled_df)

        print("==> ml-algo [Extreme Gradient Boosting Implemented]")

    def predict(self, test_data: pd.DataFrame, algo: AlgoToPredict):
        match algo:
            case algo.random_forest:
                if self.__rf != None:
                    print("==> ml-algo [Predicting for Random Forests]")
                    return self.__rf.predict(test_data)
            case algo.xgboost:
                if self.__xgb != None:
                    print("==> ml-algo [Predicting for Extreme Gradient Boosting]")
                    return self.__xgb.predict(test_data)
            case algo.ensemble:
                if self.__ensemble_model != None:
                    print("==> ml-algo [Predicting for Ensemble Model]")
                    return self.__ensemble_model.predict(test_data)
            case algo.k_means:
                if self.__kmeans != None:
                    print("==> ml-algo [Predicting for KMeans]")
                    return self.__kmeans.predict(test_data)
            case _:
                return

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
