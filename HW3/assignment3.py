#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
from typeguard import typechecked
from sklearn.impute import SimpleImputer

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    return pd.read_csv(filename)
    pass


@typechecked
def data_preprocess(feature: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: 
    Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submitted for problem 2 of HW2
    """
    df.dropna(inplace=True)
    label = df["NewLeague"]
    feature = df.drop(columns=['NewLeague','Player'], axis=1)
    label = label.replace({'A': 0, 'N': 1})
    
    non_numerical_cols = feature.select_dtypes(exclude=['int64', 'float64'])
    numerical_cols = feature.select_dtypes(include=['int64', 'float64'])
    categorical_features = pd.get_dummies(non_numerical_cols)
    preprocessed_features = pd.concat([categorical_features, numerical_cols], axis = 1)
    
    return preprocessed_features, label

@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size)
    return X_train, X_test, y_train, y_test
    pass


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Ridge Regression, train the model object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"ridge": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n):
        ridge_scores = []
        for alpha in lambda_vals:
            model_R = Ridge(alpha=alpha)
            model_R.fit(x_train, y_train)
            y_prob = model_R.predict(x_test)
            ridge_scores.append(roc_auc_score(y_test, y_prob))
        aucs["ridge"].append(ridge_scores)

    # Compute the mean AUC for each lambda value
    ridge_aucs = pd.DataFrame(aucs["ridge"])
    ridge_mean_auc = {}
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc

    print("ridge mean AUCs:")
    ridge_aucs = pd.DataFrame(aucs["ridge"])
    ridge_mean_auc = {}
    ridge_aucs = pd.DataFrame(aucs["ridge"])
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % ridge_auc)
    return ridge_mean_auc
    pass


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"lasso": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
         
    for i in range(n):
        lasso_scores = []
        for lambda_val in lambda_vals:
            model_L = Lasso(max_iter=max_iter)
            model_L.set_params(alpha=lambda_val)
            model_L.fit(x_train, y_train)
            y_prob = model_L.predict(x_test)
            lasso_scores.append(roc_auc_score(y_test, y_prob))
        aucs["lasso"].append(lasso_scores)
    
    print("lasso mean AUCs:")
    lasso_mean_auc = {} 
    lasso_aucs = pd.DataFrame(aucs["lasso"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    return lasso_mean_auc
    pass


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """
    model_R = Ridge(alpha=optimal_alpha, max_iter=max_iter)
    model_R.fit(x_train, y_train)
    coefficients = model_R.coef_
    return (model_R, coefficients)
    pass


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    model_L = Lasso(alpha=optimal_alpha, max_iter=max_iter)
    model_L.fit(x_train, y_train)
    coefficients = model_L.coef_
    return (model_L, coefficients)
    pass


@typechecked
def ridge_area_under_curve(
    model_R, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    y_score = model_R.predict(x_test)
    return roc_auc_score(y_test, y_score)
    pass


@typechecked
def lasso_area_under_curve(
    model_L, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_alpha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    y_score = model_L.predict(x_test)
    return roc_auc_score(y_test, y_score)
    pass


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val  # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data  # data can be anything! we recommend dictionary with all variables you need


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int, min_samples_split=2) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        self.min_samples_split = min_samples_split
        
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        # Check if the maximum depth has been reached.
        if len(np.unique(self.data[:, -1])) == 1 or self.max_depth == 0:
            return Node(split_val=np.mean(self.data[:, -1]), data=self.data[:, :-1])

        best_split = self.get_best_split(self.data[:, :-1])

        if best_split is None:
            return Node(split_val=np.mean(self.data[:, -1]), data=self.data[:, :-1])

        # Split the data and recursively build the left and right subtrees.
        left_data, right_data = self.data[best_split.left], self.data[best_split.right]
        left_subtree = self.__class__(left_data, max_depth=self.max_depth - 1).build_tree()
        right_subtree = self.__class__(right_data, max_depth=self.max_depth - 1).build_tree()

        return Node(
            split_val=best_split.split_val,
            data=self.data[:, :-1],
            left=left_subtree,
            right=right_subtree,
        )

    @typechecked
    def mean_squared_error(
        self, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split
        """
        left_labels = left_split if len(left_split.shape) == 1 else left_split[:, -1]
        right_labels = right_split if len(right_split.shape) == 1 else right_split[:, -1]
        left_mean = np.mean(left_labels)
        right_mean = np.mean(right_labels)
        mse_left = np.mean((left_labels - left_mean) ** 2)
        mse_right = np.mean((right_labels - right_mean) ** 2)
        return mse_left + mse_right

    
    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively
        """
        left_data = node.data
        right_data = np.empty((0, self.data.shape[1]))

        if depth == 0 or len(set(left_data[:, -1])) == 1:
            node.left = Node(split_val=np.mean(left_data[:, -1]), data=left_data)
            return

        best_split = self.get_best_split(left_data)
        if best_split is None:
            node.left = Node(split_val=np.mean(left_data[:, -1]), data=left_data)
            return

        left_indices = best_split.left
        right_indices = best_split.right
        node.left = Node(split_val=best_split.split_val, data=left_data[left_indices])
        node.right = Node(split_val=best_split.split_val, data=left_data[right_indices])

        self.split(node.left, depth - 1)
        self.split(node.right, depth - 1)


    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        if data.shape[1] == 0:
            return Node(split_val=0)
        
        best_params = None
        for i in range(data.shape[1] - 1):
            for val in np.unique(data[:, i]):
                left_side, right_side = self.one_step_split(i, val, data)
                if len(left_side) < 2 or len(right_side) < 2:
                    continue
                mse = self.mean_squared_error(left_side[:, -1], right_side[:, -1])
                if best_params is None or mse < best_params[2]:
                    best_params = (i, val, mse, left_side[:, -1].mean(), right_side[:, -1].mean())
        
        if best_params is None:
            return Node(split_val=np.mean(data[:, -1]), data=data)
        
        i, val, mse, left_mean, right_mean = best_params
        left_side, right_side = self.one_step_split(i, val, data)
        
        return Node(
            split_val=val,
            data=data,
            left=Node(split_val=left_mean),
            right=Node(split_val=right_mean),
        )

    @typechecked
    def one_step_split(self, index: int, value: float, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        if len(data.shape) == 3:
            left_side = data[data[:, index, 0] < value]
            right_side = data[data[:, index, 0] >= value]
        else: 
            left_side = data[data[:, index] < value, :]
            right_side = data[data[:, index] >= value, :]
        return left_side, right_side

@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
    if node.split_val <= row[-1]:
      return False
    else:
      return True


@typechecked
def predict(node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]) -> float:
    """
    Predict a target value for a row of data
    """
    while node.left is not None or node.right is not None:
        if comparator(node, row):
            node = node.left
        else:
            node = node.right
    return node.split_val


class TreeClassifier(TreeRegressor):
    def build_tree(self, depth=1) -> Node:
        """
        Build the decision tree recursively
        """
        data_classes, data_counts = np.unique(self.data[:, -1].astype(int), return_counts=True)
        if depth == self.max_depth or len(self.data) < self.min_samples_split or len(data_classes) == 1:
            return Node(split_val=int(data_classes[np.argmax(data_counts)]), data=self.data)

        return self.get_best_split(self.data)
    
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead
        """
        n_samples = left_split.shape[0] + right_split.shape[0]
        left_prob = left_split.shape[0] / n_samples
        right_prob = right_split.shape[0] / n_samples
        prob_left_class = []
        prob_right_class = []
        for cls in classes:
            prob_left_class.append((left_split == cls).sum() / left_split.shape[0])
            prob_right_class.append((right_split == cls).sum() / right_split.shape[0])

        lhs_class_prob = 1 - ((np.array(prob_left_class) * np.array(prob_left_class)).sum())
        rhs_class_prob = 1 - ((np.array(prob_right_class) * np.array(prob_right_class)).sum())

        return left_prob * lhs_class_prob + right_prob * rhs_class_prob

    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        if data.shape[1] == 0:
            return Node(split_val=0)

        best_params = None
        classes = np.unique(data[:, -1].astype(int))
        for i in range(data.shape[1] - 1):
            for val in np.unique(data[:, i]):
                left_side, right_side = self.one_step_split(i, val, data)
                if len(left_side) < 2 or len(right_side) < 2:
                    continue
                left_y, right_y = left_side[:, -1].astype(int), right_side[:, -1].astype(int)
                gini = self.gini_index(left_y, right_y, classes)
                if best_params is None or gini < best_params[2]:
                    best_params = (i, val, gini, left_y, right_y)

        if best_params is None:
            classes, counts = np.unique(data[:, -1].astype(int), return_counts=True)
            return Node(split_val=int(classes[np.argmax(counts)]), data=data)

        i, val, gini, left_y, right_y = best_params
        left_side, right_side = self.one_step_split(i, val, data)

        return Node(
            split_val=val,
            data=data,
            left=self.get_best_split(left_side),
            right=self.get_best_split(right_side)
        )

if __name__ == "__main__":
    # Question 1
    filename = "Hitters.csv"  # Provide the path of the dataset
    df = read_data(filename)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(
        final_features, final_label, 0.2
    )
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)

    # Plot the ROC curve of the Ridge Model. Include axes labels,
    # legend and title in the Plot. Any of the missing
    # items in plot will result in loss of points.
    
    y_score = model_R.predict(x_test)
    ridge_fpr, ridge_tpr, _ = roc_curve(y_test, y_score)
    plt.plot(ridge_fpr, ridge_tpr, label='ridge_regression')
    plt.title("Ridge ROC Curve")
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.legend()
    plt.show()
    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    
    # Plot the ROC curve of the Lasso Model.
    # Include axes labels, legend and title in the Plot.
    # Any of the missing items in plot will result in loss of points.
    
    y_score = model_L.predict(x_test)
    lasso_fpr, lasso_tpr, _ = roc_curve(y_test, y_score)
    plt.plot(ridge_fpr, ridge_tpr, label='lasso_regression')
    plt.title("Lasso ROC Curve")
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.legend()
    plt.show()

    # SUB Q1
    csvname = "noisy_sin_subsample_2.csv"
    data_regress = np.loadtxt(csvname, delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()

    mse_depths = []
    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        tree = regressor.build_tree()
        mse = 0.0
        for data_point in data_regress:
            mse += (
                data_point[1]
                - predict(tree, data_point, compare_node_with_threshold)
            ) ** 2
        mse_depths.append(mse / len(data_regress))
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()

    # SUB Q2
    csvname = "new_circle_data.csv"  # Place the CSV file in the same directory as this notebook
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    plt.figure()
    plt.scatter(
        data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    )
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(
                data_point[2]
                == predict(tree, data_point, compare_node_with_threshold)
            )
        accuracy_depths.append(correct / len(data_class))
    # Plot the MSE
    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()
