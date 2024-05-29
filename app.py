import numpy
import pandas as pd
import matplotlib.pyplot as plt

## import datasets
dataset_dir = './datasets/cantrainandtest/can-train-and-test/set_01/train_01/'
attack_free_1 = pd.read_csv(dataset_dir + "attack-free-1.csv")
attack_free_2 = pd.read_csv(dataset_dir + "attack-free-2.csv")
DoS_1 = pd.read_csv(dataset_dir + "DoS-1.csv")
DoS_2 = pd.read_csv(dataset_dir + "DoS-2.csv")
accessory_1 = pd.read_csv(dataset_dir + "accessory-1.csv")
accessory_2 = pd.read_csv(dataset_dir + "accessory-2.csv")
force_neutral_1 = pd.read_csv(dataset_dir + "force-neutral-1.csv")
force_neutral_2 = pd.read_csv(dataset_dir + "force-neutral-2.csv")
rpm_1 = pd.read_csv(dataset_dir + "rpm-1.csv")
rpm_2 = pd.read_csv(dataset_dir + "rpm-2.csv")
standstill_1 = pd.read_csv(dataset_dir + "standstill-1.csv")
standstill_2 = pd.read_csv(dataset_dir + "standstill-2.csv")

## concatenate related datasets
attack_free = pd.concat([attack_free_1, attack_free_2])
DoS = pd.concat([DoS_1, DoS_2])
accessory = pd.concat([accessory_1, accessory_2])
force_neutral = pd.concat([force_neutral_1, force_neutral_2])
rpm = pd.concat([rpm_1, rpm_2])
standstill = pd.concat([standstill_1, standstill_2])

## concatenate all data-samples into one
can_train_and_test_ds = pd.concat([attack_free, DoS, accessory, force_neutral, rpm, standstill])

## implement One Hot Vector on attack column
