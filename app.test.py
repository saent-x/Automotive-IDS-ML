import time
from py_pkg.testing.testing import Testing
from py_pkg.core.core import get_human_time, read_dataset_and_split, get_test_dataset
from py_pkg.algos.algo import Algo
from py_pkg.entities.entities import TestDataType, AttackType, AlgoToPredict
from py_pkg.testing.testing import Testing

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

tests = Testing()


def main():
    dos_dir = "./datasets/clean-data-2/set-1/training-data/dos/dos.csv"
    fn_dir = "./datasets/clean-data-2/set-1/training-data/force_neutral/force_neutral.csv"

    dos_dataset = read_dataset_and_split(dos_dir)
    fn_dataset = read_dataset_and_split(fn_dir)

    # initialize algo
    algo_1 = Algo(dos_dataset)
    algo_2 = Algo(fn_dataset)

    print("==> ml-algo [Initializing]")

    # implement individual algo
    algo_1.impl_random_forests()
    # algo_1.impl_xgboost()
    # algo_1.impl_kmeans()

    algo_2.impl_random_forests()
    # algo_2.impl_xgboost()
    # algo_2.impl_kmeans()

    algos = [(algo_1, 'DoS'), (algo_2, 'Force Neutral')]

    for algo, title in algos:
        run_all_tests(algo, title)



def run_all_tests(algo, title):
    start_time = time.time()
    # Random Forests
    true_values_pred_values_rf_kv_ka = tests.test_for_2(TestDataType.kv_ka, algo, AlgoToPredict.random_forest)
    test_metrics_rf_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_rf_kv_ka)
    
    true_values_pred_values_rf_uv_ka = tests.test_for_2(TestDataType.uv_ka, algo, AlgoToPredict.random_forest)
    test_metrics_rf_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_rf_uv_ka)
    
    # save results
    tests.save_testing_results_2(test_metrics_rf_kv_ka, f"[{title}] Random Forests - KV-KA", ['DOS', 'Force Neutral'])
    tests.save_testing_results_2(test_metrics_rf_uv_ka, f"[{title}] Random Forests - UV-KA", ['DOS', 'Force Neutral'])
    
    
    # # Extreme Gradient Boosting
    # true_values_pred_values_xg_kv_ka = tests.test_for_2(TestDataType.kv_ka, algo, AlgoToPredict.xgboost)
    # test_metrics_xg_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_xg_kv_ka)
    
    # true_values_pred_values_xg_uv_ka = tests.test_for_2(TestDataType.uv_ka, algo, AlgoToPredict.xgboost)
    # test_metrics_xg_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_xg_uv_ka)
    
    # # save results
    # tests.save_testing_results_2(test_metrics_xg_kv_ka, f"[{title}] Extreme Gradient Boosting - KV-KA")
    # tests.save_testing_results_2(test_metrics_xg_uv_ka, f"[{title}] Extreme Gradient Boosting - UV-KA")
    
    # K-Means Clustering
    # true_values_pred_values_kmeans_kv_ka = tests.test_for_2(TestDataType.kv_ka, algo, AlgoToPredict.k_means)
    # test_metrics_kmeans_kv_ka = tests.generate_all_test_metrics(true_values_pred_values_kmeans_kv_ka)
    
    # true_values_pred_values_kmeans_uv_ka = tests.test_for_2(TestDataType.uv_ka, algo, AlgoToPredict.k_means)
    # test_metrics_kmeans_uv_ka = tests.generate_all_test_metrics(true_values_pred_values_kmeans_uv_ka)
    
    # # save results
    # tests.save_testing_results_2(test_metrics_kmeans_kv_ka, f"[{title}] K-Means - KV-KA", ['DOS', 'Force Neutral'])
    # tests.save_testing_results_2(test_metrics_kmeans_uv_ka, f"[{title}] K-Means - UV-KA", ['DOS', 'Force Neutral'])
    
    close_time = time.time()
    
    print(f"\n==> ml-algo [Model training for {title} and Testing Completed in {get_human_time(close_time, start_time)}]")



main()