"""
Original Author: Gabriel Lojo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X_column, y):
    parent_entropy = calculate_entropy(y)

    values, counts = np.unique(X_column, return_counts=True)
    weighted_child_entropy = 0

    for value, count in zip(values, counts):
        subset_y = y[X_column == value]
        weight = count / len(y)
        weighted_child_entropy += weight * calculate_entropy(subset_y)

    return parent_entropy - weighted_child_entropy


class Node:
    def __init__(self, feature_index=None, branches=None, prediction=None, fallback_class=None):
        self.feature_index = feature_index
        self.branches = branches
        self.prediction = prediction
        self.fallback_class = fallback_class

def build_tree(X, y, feature_indices):
    unique_labels, counts = np.unique(y, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]

    if len(unique_labels) == 1:
        return Node(prediction=unique_labels[0])

    if len(feature_indices) == 0:
        return Node(prediction=majority_class)

    gains = [information_gain(X[:, i], y) for i in feature_indices]

    best_feature_idx = feature_indices[np.argmax(gains)]

    branches = {}
    unique_values = np.unique(X[:, best_feature_idx])

    remaining_features = [f for f in feature_indices if f != best_feature_idx]

    for value in unique_values:
        subset_indices = np.where(X[:, best_feature_idx] == value)[0]
        subset_X = X[subset_indices]
        subset_y = y[subset_indices]

        if len(subset_y) == 0:
            branches[value] = Node(prediction=majority_class)
        else:
            branches[value] = build_tree(subset_X, subset_y, remaining_features)

    return Node(feature_index=best_feature_idx, branches=branches, fallback_class=majority_class)

def predict_instance(node, x):
    if node.prediction is not None:
        return node.prediction

    feature_val = x[node.feature_index]

    if feature_val in node.branches:
        return predict_instance(node.branches[feature_val], x)
    else:
        return node.fallback_class

def predict_tree(tree, X_test):
    return np.array([predict_instance(tree, x) for x in X_test])

def main():
    try:
        df = pd.read_csv('car.csv')
    except FileNotFoundError:
        print("Error: Car dataset CSV not found. Please check the file name/location.")
        return

    X = df.iloc[:, :6].values
    y = df.iloc[:, 6].values

    initial_features = list(range(X.shape[1]))

    num_runs = 100
    train_accuracies = []
    test_accuracies = []

    print(f"Running Decision Tree evaluations {num_runs} times. This will take a moment...")

    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        tree = build_tree(X_train, y_train, initial_features)

        y_train_pred = predict_tree(tree, X_train)
        train_acc = np.mean(y_train_pred == y_train)
        train_accuracies.append(train_acc)

        y_test_pred = predict_tree(tree, X_test)
        test_acc = np.mean(y_test_pred == y_test)
        test_accuracies.append(test_acc)

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/100 runs...")

    train_mean, train_std = np.mean(train_accuracies), np.std(train_accuracies)
    test_mean, test_std = np.mean(test_accuracies), np.std(test_accuracies)

    print("\n--- FINAL RESULTS ---")
    print(f"Training - Mean: {train_mean:.4f}, Std Dev: {train_std:.4f}")
    print(f"Testing  - Mean: {test_mean:.4f}, Std Dev: {test_std:.4f}")


    # Training Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(train_accuracies, bins=15, color='blue', edgecolor='white')
    train_text = f"Mean: {train_mean:.4f}\nStd Dev: {train_std:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, train_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top', bbox=props)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency over Training Data')
    plt.title('Decision Tree Training Accuracy Distribution (100 Runs)')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('dt_training_histogram.png')
    plt.show()

    # Testing Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(test_accuracies, bins=15, color='red', edgecolor='white')
    test_text = f"Mean: {test_mean:.4f}\nStd Dev: {test_std:.4f}"
    plt.gca().text(0.05, 0.95, test_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top', bbox=props)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency over Testing Data')
    plt.title('Decision Tree Testing Accuracy Distribution (100 Runs)')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('dt_testing_histogram.png')
    plt.show()

if __name__ == "__main__":
    main()