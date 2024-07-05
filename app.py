from cgi import test
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from py_pkg.core.core import get_human_time, read_dataset_and_split, get_test_dataset
from py_pkg.algos.algo import Algo
from py_pkg.entities.entities import TestDataType, AttackType, AlgoToPredict
from py_pkg.testing.testing import Testing

# import datasets
dataset_dir = './datasets/clean-data/updated_dataset.csv'


def main():
    start_time = time.time()
    
    split_dataset = read_dataset_and_split(dataset_dir)

    # initialize algo
    algo = Algo(split_dataset)
    print("==> ml-algo [Initializing]")

    # implement individual algo
    algo.impl_random_forests()
    algo.impl_xgboost()
    algo.impl_kmeans()       

    # test all algos and generate test metrics to be studied

    tests = Testing()

    # Random Forests
    true_values_pred_values_rf_kv_ka = tests.test_for(TestDataType.kv_ka, algo, AlgoToPredict.random_forest)
    test_metrics_rf_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_rf_kv_ka)

    true_values_pred_values_rf_uv_ka = tests.test_for(TestDataType.uv_ka, algo, AlgoToPredict.random_forest)
    test_metrics_rf_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_rf_uv_ka)

    # save results
    tests.save_testing_results(test_metrics_rf_kv_ka, "Random Forests - KV-KA")
    tests.save_testing_results(test_metrics_rf_uv_ka, "Random Forests - UV-KA")


    # Extreme Gradient Boosting
    true_values_pred_values_xg_kv_ka = tests.test_for(TestDataType.kv_ka, algo, AlgoToPredict.xgboost)
    test_metrics_xg_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_xg_kv_ka)

    true_values_pred_values_xg_uv_ka = tests.test_for(TestDataType.uv_ka, algo, AlgoToPredict.xgboost)
    test_metrics_xg_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_xg_uv_ka)

    # save results
    tests.save_testing_results(test_metrics_xg_kv_ka, "Extreme Gradient Boosting - KV-KA")
    tests.save_testing_results(test_metrics_xg_uv_ka, "Extreme Gradient Boosting - UV-KA")

    # K-Means Clustering
    true_values_pred_values_kmeans_kv_ka = tests.test_for(TestDataType.kv_ka, algo, AlgoToPredict.k_means)
    test_metrics_kmeans_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_kmeans_kv_ka)

    true_values_pred_values_kmeans_uv_ka = tests.test_for(TestDataType.uv_ka, algo, AlgoToPredict.k_means)
    test_metrics_kmeans_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_kmeans_uv_ka)

    # save results
    tests.save_testing_results(test_metrics_kmeans_kv_ka, "K-Means - KV-KA")
    tests.save_testing_results(test_metrics_kmeans_uv_ka, "K-Means - UV-KA")

    close_time = time.time()

    print(f"\n==> ml-algo [Model training and Testing Completed in {get_human_time(close_time, start_time)}]")




main()
