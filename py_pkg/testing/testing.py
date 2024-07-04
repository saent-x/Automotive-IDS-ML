
import time
from turtle import pos
from typing import Any, List, Tuple
from py_pkg.algos.algo import Algo
from py_pkg.core.core import generate_metrics_report, get_test_dataset
from py_pkg.entities.entities import AlgoToPredict, AttackType, TestDataType


class Testing:

    def test_for(self, test_type: TestDataType, algo: Algo, algo_type: AlgoToPredict) -> Any:
        
        test_data_raw_dos = get_test_dataset(test_type, attack_type=AttackType.dos)
        test_data_raw_fn = get_test_dataset(test_type, attack_type=AttackType.force_neutral)
        test_data_raw_rpm = get_test_dataset(test_type, attack_type=AttackType.rpm)
        test_data_raw_ss = get_test_dataset(test_type, attack_type=AttackType.standstill)

        test_data_dos = test_data_raw_dos.drop(columns=["attack"])
        test_data_fn = test_data_raw_fn.drop(columns=["attack"])
        test_data_rpm = test_data_raw_rpm.drop(columns=["attack"])
        test_data_ss = test_data_raw_ss.drop(columns=["attack"])

        y_true_dos = test_data_raw_dos["attack"].replace(1,2) # 2 signifies DoS attacks
        y_true_fn = test_data_raw_fn["attack"].replace(1,3) # 3 signifies force_neutral attacks
        y_true_rpm = test_data_raw_rpm["attack"].replace(1,4) # 4 signifies rpm attacks
        y_true_ss = test_data_raw_ss["attack"].replace(1,5) # 5 signifies standstill attacks

        start_time = time.time()

        y_pred_dos = algo.predict(test_data_dos, algo_type)
        y_pred_fn = algo.predict(test_data_fn, algo_type)
        y_pred_rpm = algo.predict(test_data_rpm, algo_type)
        y_pred_ss = algo.predict(test_data_ss, algo_type)

        close_time = time.time()

        print(f"\n==> ml-algo [{algo_type.name} Predicted in {close_time - start_time} for {test_type} test set \n")

        true_values = [y_true_dos, y_true_fn, y_true_rpm, y_true_ss]
        predicted_values = [y_pred_dos, y_pred_fn, y_pred_rpm, y_pred_ss]
        pos_labels = [2,3,4,5]

        return (true_values, predicted_values, pos_labels)
    
    def generate_all_test_metrics(self, true_values_and_predicted_values) -> List[Tuple[str, Any, Any, Any]]:
        resulting_metrics = []

        for y_true, y_pred, pos_label in true_values_and_predicted_values:
            if y_true & y_pred & pos_label:
                class_report, df, cm = generate_metrics_report(y_true, y_pred, pos_label)
                resulting_metrics.append((class_report, df, cm, pos_label))
        
        return resulting_metrics
    
    def save_testing_results(self, resulting_metrics: List[Tuple[str, Any, Any, Any]], title: str):
        filename = 'test_results.txt'

        with open(filename, 'w') as file:
            # Iterate through the list of resulting metrics and write each classification report to the file
            for idx, (class_report, df, cm, pos_label) in enumerate(resulting_metrics):
                file.write(f"Report for {title}:\n")
                file.write(class_report)
                file.write("\n\n")

        print(f"All test results has been written to {filename}")


