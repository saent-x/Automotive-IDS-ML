import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_pkg.core.core import read_dataset_and_split, get_test_dataset
from py_pkg.algos.algo import Algo
from py_pkg.entities.entities import TestDataType, AttackType, AlgoToPredict

# import datasets
dataset_dir = './datasets/clean-data/updated_dataset.csv'


def main():
    split_dataset = read_dataset_and_split(dataset_dir)

    # initialize algo
    algo = Algo(split_dataset)
    print("==> ml-algo [Initializing]")

    # implement individual al  okayokay sure11gos
    algo.impl_random_forests()
    algo.impl_xgboost()
    algo.impl_kmeans()

    # test individual algos
    test_data_kv_ka = get_test_dataset(test_type=TestDataType.kv_ka, attack_type=AttackType.dos)
    test_result_1 = algo.predict(test_data_kv_ka, AlgoToPredict.random_forest)

    print(f"==> ml-algo [Test Results]")
    print(test_result_1)
    # show test results and analysis on web server

    # create ensemble model

    # test ensemble model


main()
