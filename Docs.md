# Automotive In-vehicle IDS System Utilizing Machine Learning

In the evolving landscape of automotive technology, securing in-vehicle networks is crucial. The proposition involves a ***Machine Learning-based Intrusion Detection System*** (IDS) with a multi-tier hybrid architecture that integrates both ***signature-based detection*** (Supervised learning) and ***anomaly-based detection*** (Unsupervised learning). This approach combines the accuracy of ***signature-based detection*** for known threats with the adaptability of ***anomaly-based detection*** methods for new threats, offering a robust and comprehensive security solution for vehicular networks.

## Breakdown of ML Algorithms Utilized

| Name                   | Algorithm Type        | Strong Point                                                                    | Utilized | Reason                                                                                 |
| ---------------------- | --------------------- | ------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| Decision Tree          | Supervised Learning   | Interpretability, minimal data preparation, non-parametric, feature importance. | -        | Random forests already use decision trees and are more accurate. Good for speed only.  |
| Random Forests         | Supervised Learning   | Accuracy, robustness, versatility, feature importance.                          | YES      | Best for a robust, accurate model that mitigates overfitting.                          |
| Extra Gradient Boost   | Supervised Learning   | Performance, regularization, handles missing data, parallel processing.         | YES      | Best for high performance and efficiency on large datasets.                            |
| Support Vector Machine | Supervised Learning   | Effective in high dimensions, memory efficient, and versatile.                  | -        | Less efficient and slower than Random forests for large datasets.                      |
| K-Means Clustering     | Unsupervised Learning | Simplicity, scalability, and speed                                              | -        | Not applicable (supervised learning algorithms preferred for this task)                |                                         | YES      | Best for fast and simple clustering, especially with large datasets, and it scales well to large datasets.               |

## Breakdown of CAN Traffic Datasets

| Dataset                                             | Description                                                                                                                                                                                                                                                                                      | Link                                                                      | Reference                                                |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------|
| Heavy duty truck CAN-bus dataset                    | This dataset features over 180 hours of CAN bus traffic from a Renault Euro VI heavy-duty truck across various driving conditions.                                                                                                                                                               | [Dataset Link](https://etsin.fairdata.fi/dataset/7586f24f-c91b-41df-92af-283524de8b3e) | University of Turku, May 31, 2021                        |
| can-train-and-test                                  | Controller Area Network (CAN) traffic for the 2017 Subaru Forester, the 2016 Chevrolet Silverado, the 2011 Chevrolet Traverse, and the 2011 Chevrolet Impala.                                                                                                                                    | [Dataset Link](https://data.dtu.dk/articles/dataset/can-train-and-test/24805533/1)    | Brooke Lampe, Weizhi Meng, January 17, 2024              |
| Car-Hacking Dataset for intrusion detection         | Datasets which include DoS attack, fuzzy attack, spoofing the drive gear, and spoofing the RPM gauge. Constructed by logging CAN traffic via the OBD-II port from a real vehicle while message injection attacks were performed. Datasets contain each 300 intrusions of message injection.          | [Dataset Link](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset)           | Eunbi Seo, Hyun Min Song, Huy Kang Kim, July 18, 2019    |

## Test Dataset Categories and Attacks

For each category of test dataset that exists, several attack subsets exist within it to test over the training dataset.

| **Category**                   | **Attacks**                                          | **Test Purpose**                                           |
|--------------------------------|------------------------------------------------------|------------------------------------------------------------|
| Known Vehicle Known Attacks    | DoS, force_neutral, rpm, standstill                  | To test trained model of known vehicle with known attacks. |
| Known Vehicle Unknown Attacks  | Double, fuzzing, interval, speed, systematic, triple | To test trained model of known vehicle with unknown attacks. |
| Unknown Vehicle Known Attack   | DoS, force_neutral, rpm, standstill                  | To test trained model of unknown vehicle with known attacks. |
| Unknown Vehicle Unknown Attack | Double, fuzzing, interval, speed, systematic, triple | To test trained model of unknown vehicle with unknown attacks. |

---

## Data Processing Pipeline for CAN Train Dataset

1. **Import necessary data processing libraries.**

    ```python
    import pandas as pd
    from sklearn import preprocessing
    ```

2. **Read in all CSV training dataset and merge separated attack subsets into one.**

    ```python
    # for example:
    attack_free_1 = pd.read_csv(dataset_dir + "attack-free-1.csv") 
    attack_free_2 = pd.read_csv(dataset_dir + "attack-free-2.csv")
    
    DoS_1 = pd.read_csv(dataset_dir + "DoS-1.csv")
    DoS_2 = pd.read_csv(dataset_dir + "DoS-2.csv")
    
    attack_free = pd.concat([attack_free_1, attack_free_2])
    DoS = pd.concat([DoS_1, DoS_2])
    accessory = pd.concat([accessory_1, accessory_2])
    ```

3. **Merge all attack subsets into a single unique data subset based on attack-type.**

    ```python
    accessory['attack'] = accessory['attack'].replace(0, 1)
    DoS['attack'] = DoS['attack'].replace(1, 2)  
    force_neutral['attack'] = force_neutral['attack'].replace(1, 3)  
    rpm['attack'] = rpm['attack'].replace(1, 4)  
    standstill['attack'] = standstill['attack'].replace(1, 5)
    ```

4. **Concatenate all attack subsets into one as the training dataset.**

    ```python
    merged_datasets = pd.concat([attack_free, accessory, DoS, force_neutral, rpm, standstill])
    ```

5. **Encode columns with categorical data and normalize the data.**

    ```python
    label_encoder = preprocessing.LabelEncoder()
    
    merged_datasets["arbitration_id"] = label_encoder.fit_transform(merged_datasets["arbitration_id"])
    merged_datasets["data_field"] = label_encoder.fit_transform(merged_datasets["data_field"])
    
    merged_datasets.to_csv("updated_dataset.csv", sep=',', index=False, encoding='utf-8')
    ```

---

## Data Processing Pipeline for CAN Test Dataset

1. **Import and read test dataset CSV files into a DataFrame for each test data subset.**

    ```python
    # for example:
    DoS_3 = pd.read_csv(dataset_dir + "DoS-3.csv")
    DoS_4 = pd.read_csv(dataset_dir + "DoS-4.csv")
    
    force_neutral_3 = pd.read_csv(dataset_dir + "force-neutral-3.csv")
    force_neutral_4 = pd.read_csv(dataset_dir + "force-neutral-4.csv")
    ```

2. **Merge test subsets into one, resulting in a unique subset for each attack type.**

    ```python
    # merge related datasets
    double = pd.concat([double_3, double_4])
    fuzzing = pd.concat([fuzzing_3, fuzzing_4])
    interval = pd.concat([interval_3, interval_4])
    speed = pd.concat([speed_3, speed_4])
    systematic = pd.concat([systematic_3, systematic_4])
    triple = pd.concat([triple_3, triple_4])
    ```

3. **Encode columns with categorical values.**

    ```python
    double["arbitration_id"] = label_encoder.fit_transform(double["arbitration_id"])
    double["data_field"] = label_encoder.fit_transform(double["data_field"])

    fuzzing["arbitration_id"] = label_encoder.fit_transform(fuzzing["arbitration_id"])
    fuzzing["data_field"] = label_encoder.fit_transform(fuzzing["data_field"])
    
    interval["arbitration_id"] = label_encoder.fit_transform(interval["arbitration_id"])
    interval["data_field"] = label_encoder.fit_transform(interval["data_field"])
    ```

4. **Save the processed testing dataset to CSV format.**

---

# Algorithm Implementation

The *"Algo"* class contains the implementation of the 3 selected algorithms for testing as seen in the table above, namely: **Random Forests, Extreme Gradient Boosting**, and **KMeans Clustering Algorithm**.

## Random Forest Algorithm

**Random Forests** is an ensemble learning method for classification and regression. It operates by constructing multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. This approach helps improve accuracy and control overfitting.

#### Implementation

The **Random Forests Algorithm** utilized is implemented at its core using sci-kit learn machine learning libraries, particularly the ***RandomForestClassifier*** as seen below:

```python
def impl_random_forests(self):
    # Initiate and train model
    start_time = time.time()
    self.__rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    self.__rf.fit(self.split_dataset.x_features, self.split_dataset.y_features.values.ravel())
    
    close_time = time.time()
    print(f"==> ml-algo [Random Forests Implemented in {close_time - start_time}]")
```

The ***RandomForestClassifier*** *fits* the training datasets which is first split using **`read_dataset_and_split`** function into "X_features" which represent the Independent variables or features and the "y_features" a 1D array which represents the dependent variable or target variable i.e. the column to be predicted.


## Extreme Gradient Boosting Algorithm

**Extreme Gradient Boosting (XGBoost)** is an optimized implementation of gradient boosting designed to be highly efficient and scalable. It builds an ensemble of trees sequentially, where each new tree attempts to correct the errors made by the previous ones. XGBoost includes advanced features like regularization to prevent overfitting and parallel processing to speed up training.

#### Implementation

The **Extreme Gradient Boosting Algorithm** utilized is implemented using scikit-learn machine learning libraries, particularly the ***GradientBoostingClassifier*** as seen below:

```python
def impl_xgboost(self):         
    # Initiate and train model         
    start_time = time.time()          
    
    self.__xgb = GradientBoostingClassifier()         
    y_features = np.ravel(self.split_dataset.y_features)
    self.__xgb.fit(self.split_dataset.x_features, y_features)          
    
    close_time = time.time()
    
    print(f"==> ml-algo [Gradient Boosting Implemented in {close_time - start_time}]")
```

The ***GradientBoostingClassifier*** *fits* the training datasets which is first split using **`read_dataset_and_split`** function into "X_features" which represent the Independent variables or features and the "y_features" a 1D array which represents the dependent variable or target variable i.e. the column to be predicted.



## KMeans Clustering Algorithm

**K-Means Algorithm** is a popular clustering method used to partition a dataset into K distinct, non-overlapping subsets (clusters). Each data point is assigned to the cluster with the nearest mean, and the process is repeated iteratively to minimize the variance within clusters. It is widely used for exploratory data analysis and pattern recognition.

#### Implementation

The **K-Means Algorithm** utilized is implemented using sci-kit learn machine learning libraries particularly the ***KMeans*** as seen below;

```python
def impl_kmeans(self):
	# create scaled DataFrame where each variable has mean of 0 and standard dev of 1
	df = pd.concat([self.split_dataset.x_features, self.split_dataset.y_features], axis=1)
	start_time = time.time()
	
	self.__kmeans = KMeans(init="random", n_init='auto', random_state=1)
	self.__kmeans.fit(df)
	
	close_time = time.time()
	print(f"==> ml-algo [KMeans Implemented in {close_time - start_time}]")
```

The ***KMeans Algorithm*** *fits* the training dataset but does not require splitting the dataset into *feature* and *target* or *independent* and *dependent* variables respectively since it is an unsupervised learning algorithm, we pass the pre-processed training dataset as is for fitting.



---


# Testing

Testing was performed on two categories of datasets, the CAN Training dataset obtained after data processing and the Testing dataset also obtained during the data processing stage. The trained model was tested against two categories of test datasets, namely:

- Known Vehicle Known Attack Dataset
- Unknown Vehicle Known Attack Dataset

###### # Read in test data for kv-ka test category and for all attack types

```python
test_data_kv_ka_raw_dos = get_test_dataset(test_type=TestDataType.kv_ka, attack_type=AttackType.dos) 
test_data_kv_ka_raw_fn = get_test_dataset(test_type=TestDataType.kv_ka, attack_type=AttackType.force_neutral) 
test_data_kv_ka_raw_rpm = get_test_dataset(test_type=TestDataType.kv_ka, attack_type=AttackType.rpm) 
test_data_kv_ka_raw_ss = get_test_dataset(test_type=TestDataType.kv_ka, attack_type=AttackType.standstill)
```


###### # drop the attack (target column) column from each test dataset to be used to generate predictions

```python
test_data_kv_ka_dos = test_data_kv_ka_raw_dos.drop(columns=["attack"]) test_data_kv_ka_fn = test_data_kv_ka_raw_fn.drop(columns=["attack"]) test_data_kv_ka_rpm = test_data_kv_ka_raw_rpm.drop(columns=["attack"]) test_data_kv_ka_ss = test_data_kv_ka_raw_ss.drop(columns=["attack"])
```

###### # replace the attack column categorical values in all test dataset to match the attack column values in the training dataset

```python
y_true_dos = test_data_kv_ka_raw_dos["attack"].replace(1,2) # 2 signifies DoS attacks
y_true_fn = test_data_kv_ka_raw_fn["attack"].replace(1,3) # 3 signifies force_neutral attacks
y_true_rpm = test_data_kv_ka_raw_rpm["attack"].replace(1,4) # 4 signifies rpm attacks
y_true_ss = test_data_kv_ka_raw_ss["attack"].replace(1,5) # 5 signifies standstill attacks

start_time = time.time()
```

###### # we then create predictions using the *"algo.predict"* function defined in the *"Algo"* class

```python
y_pred_dos = algo.predict(test_data_kv_ka_dos, AlgoToPredict.random_forest)
y_pred_fn = algo.predict(test_data_kv_ka_fn, AlgoToPredict.random_forest)
y_pred_rpm = algo.predict(test_data_kv_ka_rpm, AlgoToPredict.random_forest)
y_pred_ss = algo.predict(test_data_kv_ka_ss, AlgoToPredict.random_forest)
```


After the predictions are generated using the two categorical test datasets **"known vehicle known attack"** and **"unknown vehicle known attack"**, we can use the **"y_true_*"** values—a data-frame which contains the accurate predictions for the *"attack"* column—and **"y_pred_*"**—a data-frame which contains the predicted values calculated using the trained model—to generate a metrics report using the **`generate_metrics_report`** function. This function calculates the ***accuracy, precision, recall, f1, roc_curve, roc_auc***, and a ***confusion matrix***, then returns it.

More information on the generated metrics can be found in the table below:

| Scoring Metrics    | Explanation                                                                                                                    |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Confusion Matrix   | Summarizes the performance of a classification model by showing the counts of correct and incorrect predictions for each class |
| Detection Accuracy | Ability to classify normal and intrusive traffic                                                                               |
| Precision          | Proportion of true positive predictions among all positive predictions                                                         |
| Recall             | Proportion of true positive predictions among all actual positive instances                                                    |
| ROC curve          | Plot of true positive rate vs false positive rate at various classification thresholds                                         |
| ROC AUC            | Area under the ROC curve, measuring overall classification performance    
