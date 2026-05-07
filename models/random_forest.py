"""
Original Author: Shreyas Donti
"""
# Imports
import sklearn
import pandas as pd
import numpy as np
import math
from models.decision_tree import DecisionTree

# Bootstrap sample function
def bootstrap_sample(x, y):
    n = len(x)
    indices = np.random.choice(n, n, replace=True)
    return x[indices], y[indices]

# RandomForest class
class RandomForest():
    def __init__(self, ntree=10):
        self.x = None
        self.y = None
        self.ntree = ntree
        self.trees = []

    def _predict(self, x):
        # Predict a point in each tree
        preds = [tree._predict(x) for tree in self.trees]
        # Return the most common prediction
        return max(set(preds), key=preds.count)

    def _accuracy(self, x, y):
        # Correct guesses and total points
        correct = 0
        for i in range(len(x)):
            correct += 1 if self._predict(x[i]) == y[i] else 0
        return correct/len(x)

    def train(self, x, y):
        self.trees = []
        # Fit the training dataset to the model
        self.x = np.array(x)
        self.y = np.array(y)

        for _ in range(self.ntree):
            # Create a DecisionTree on a bootstrapped sample of the training data
            x_sample, y_sample = bootstrap_sample(self.x, self.y)
            tree = DecisionTree()
            tree.train(x_sample, y_sample)

            # Add it to the random forest
            self.trees.append(tree)

        # Predict each point
        return self._accuracy(self.x, self.y)

    def test(self, x, y):
        # Convert test data into Numpy array
        x_test = np.array(x)
        y_test = np.array(y)

        # Predict each point
        return self._accuracy(x_test, y_test)