"""
File: titanic_level1.py
Name: Fanny
----------------------------------
This file builds a machine learning algorithm from scratch
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from collections import defaultdict


TRAIN_FILE = "titanic_data/train.csv"
TEST_FILE = "titanic_data/test.csv"


def data_preprocess(filename: str, data: dict, mode="Train", training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: an empty Python dictionary
    :param mode: str, indicating if it is training mode or testing mode
    :param training_data: dict[str: list], key is the column name, value is its data
                                              (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """

    data = defaultdict(list)
    # training mode
    if mode == "Train":
        clean_na = []
        # Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
        with open(filename, "r") as f:
            is_first = True
            for line in f:
                if is_first:
                    is_first = False
                else:
                    line = line.strip()  # Remove the '\n' at the end of each line
                    data_list = line.split(",")
                    #  check missing data < Age, Embarked >
                    age = data_list[6]
                    embarked = data_list[12]
                    if age != "" and embarked != "":
                        clean_na.append(data_list)
        for line in clean_na:
            data["Survived"].append(int(line[1]))
            data["Pclass"].append(int(line[2]))
            data["Sex"].append(1 if line[5] == "male" else 0)  # male → 1, female → 0
            data["Age"].append(float(line[6]))  # age in float
            data["SibSp"].append(int(line[7]))
            data["Parch"].append(int(line[8]))
            data["Fare"].append(float(line[10]))  # float
            data["Embarked"].append({"S": 0, "C": 1, "Q": 2}[line[12]])  # mapping
        return data

    else:  # test mode
        avg_age = round(sum(training_data["Age"]) / len(training_data["Age"]), 3)
        avg_fare = round(sum(training_data["Fare"]) / len(training_data["Fare"]), 3)
        # Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
        with open(filename, "r") as f:
            is_first = True
            for line in f:
                if is_first:
                    is_first = False
                else:
                    line = line.strip()  # Remove the '\n' at the end of each line
                    data_list = line.split(",")
                    data["Pclass"].append(int(data_list[1]))
                    data["Sex"].append(1 if data_list[4] == "male" else 0)
                    data["Age"].append(
                        float(data_list[5]) if data_list[5] != "" else avg_age
                    )
                    data["SibSp"].append(int(data_list[6]))
                    data["Parch"].append(int(data_list[7]))
                    data["Fare"].append(
                        float(data_list[9]) if data_list[9] != "" else avg_fare
                    )
                    data["Embarked"].append({"S": 0, "C": 1, "Q": 2}[data_list[11]])
        return data


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    before_ohe = data[feature]
    feature_n = set(data[feature])
    for n in feature_n:
        if feature == "Sex":
            new_key = "Sex_1" if n == "male" else "Sex_0"
        elif feature == "Embarked":
            new_key = f"Embarked_{n}"
        elif feature == "Pclass":
            key = {1: "0", 2: "1", 3: "2"}
            new_key = f"Pclass_{key[n]}"
        for value in before_ohe:
            data[new_key].append(1 if value == n else 0)
    del data[feature]
    return data


def normalize(data: dict):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :return data: dict[str, list], key is the column name, value is its normalized data
    """
    for key in data:
        max_data = max(data[key])
        min_data = min(data[key])
        for i in range(len(data[key])):
            data[key][i] = (data[key][i] - min_data) / (max_data - min_data)
    return data


def learnPredictor(
    inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float
):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, known as step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    # Step 1 : Initialize weights
    weights = {}  # feature => weight
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0

    def sigmoid(k):
        return 1 / (1 + math.exp(-k))

    # Step 2 : Start training
    for epoch in range(num_epochs):
        # Step 3 : Feature Extract
        for i in range(len(labels)):
            # degree 1
            phi_x = defaultdict(int)
            for key in keys:
                phi_x[key] = inputs[key][i]
            # degree 2
            if degree == 2:
                for j in range(len(keys)):
                    for k in range(j, len(keys)):
                        f1 = keys[j]
                        f2 = keys[k]
                        phi_x[f1 + f2] = inputs[f1][i] * inputs[f2][i]

            # Step 4 : Update weights
            theta = dotProduct(phi_x, weights)
            h = sigmoid(theta)
            increment(weights, -alpha * (h - labels[i]), phi_x)
    return weights


############################################################


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param scale: float, scale value of d2 to add onto the corresponding value of d1
    @param dict d2: a feature vector.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key, value in d2.items():
        d1[key] = d1.get(key, 0) + scale * value
    # END_YOUR_CODE


############################################################


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector. Key is a feature (string); value is its weight (float).
    @param dict d2: a feature vector. Key is a feature (string); value is its weight (float)
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return sum(d1.get(key, 0) * value for key, value in d2.items())
        # END_YOUR_CODE
