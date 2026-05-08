"""
Original Author: Shreyas Donti

The model has been slightly adjusted to fit with the testing pipeline, as well as allowed for more
hyperparam tuning of RandomForest. I've also made it use the DecisionTree model from Gabriel instead
of my own for consistency.
"""
# Imports
import sklearn
import pandas as pd
import numpy as np
import math
from models.decision_tree import DecisionTreeClassifier

# Split criteria
DEFAULT_MIN_SAMPLES_SPLIT=5
DEFAULT_MAX_DEPTH=10
DEFAULT_MIN_GAIN=0

# Bootstrap sample function
def bootstrap_sample(x, y):
    n = x.shape[1]
    indices = np.random.choice(n, n, replace=True)
    return x[:, indices], y[indices]

# RandomForest class
class RandomForest:
    def __init__(self, ntree=10, **kwargs):
        self.x = None
        self.y = None
        self.ntree = ntree
        self.trees = []

    def fit(self, x, y, **kwargs):
        self.trees = []

        # Fit the training dataset to the model
        self.x = np.array(x)
        self.y = np.array(y).flatten()

        for _ in range(self.ntree):
            # Create a DecisionTree on a bootstrapped sample of the training data
            x_sample, y_sample = bootstrap_sample(self.x, self.y)

            tree = DecisionTreeClassifier()

            tree.fit(x_sample, y_sample)

            # Add it to the random forest
            self.trees.append(tree)

        return self


    def predict(self, x):
        # Convert test data into Numpy array
        x_test = np.array(x)

        # Predict each point
        tree_preds = np.array([tree.predict(x_test) for tree in self.trees])
        predictions = []

        # Get average prediction
        for i in range(tree_preds.shape[1]):
            vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            predictions.append(vals[np.argmax(counts)])

        return np.array(predictions)