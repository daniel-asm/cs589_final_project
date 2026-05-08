"""
Original Author: Shreyas Donti
"""
# Imports
import sklearn
import pandas as pd
import numpy as np
import math

# Split criteria
DEFAULT_MIN_SAMPLES_SPLIT=5
DEFAULT_MAX_DEPTH=10
DEFAULT_MIN_GAIN=1e-3

# DecisionTree class
class DecisionTree():
    def __init__(self, min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT, max_depth=DEFAULT_MAX_DEPTH, min_gain=DEFAULT_MIN_GAIN):
        self.x = None
        self.y = None
        self.tree = None

        # Stopping criteria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain

    # Return a dict counting each unique element
    def _unique_count(self, arr):
        dct = {}
        for item in arr:
            if item in dct:
                dct[item] += 1
            else:
                dct[item] = 1
        return dct

    # Return the element with the majority count
    def _majority_count(self, arr):
        dct = self._unique_count(arr)
        return max(dct, key=dct.get)

    def _is_numeric(self, x_col):
        # Might miss a few number category features, but most categorical features are strings
        # Didn't see much gain from marking categorical features directly (via a pre-defined array)
        return np.issubdtype(x_col.dtype, np.number)

    # Calculate entropy of label
    def _entropy(self, feature):
        # Get count of each feature
        feat_cnt = self._unique_count(feature)
        entropy = 0.0

        # Use the formula for entropy calculation
        for cnt in feat_cnt.values():
            p = cnt / len(feature)
            entropy -= p * math.log2(p)

        return entropy

    def _info_gain_categorical(self, x_col, y):
        # Calculate original entropy of feature
        org_entropy = self._entropy(y)

        # Calculate entropy of splitting
        split_entropy = 0.0
        for val in np.unique(x_col):
            y_subset = y[x_col == val]
            wght = len(y_subset) / len(y)
            split_entropy += wght * self._entropy(y_subset)

        # Find entropy difference
        return org_entropy - split_entropy

    def _info_gain_numeric(self, x_col, y):
        # Get all unique values
        thresholds = np.unique(x_col)

        # Derive from each a possible threshold (find midpoint between each)
        if len(thresholds) > 1:
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

        best_gain = -1
        best_thresh = None

        parent_entropy = self._entropy(y)

        # For each threshold
        for t in thresholds:
            left_side = x_col <= t
            right_side = x_col > t

            if np.sum(left_side) == 0 or np.sum(right_side) == 0:
                continue

            # Split values into left and right side
            y_left = y[left_side]
            y_right = y[right_side]

            w_left = len(y_left) / len(y)
            w_right = len(y_right) / len(y)

            # Find entropy of each subtracted from the parent
            gain = parent_entropy - (w_left * self._entropy(y_left) + w_right * self._entropy(y_right))

            # Whichever partition reduces entropy the most, gives us the most informtion
            if gain > best_gain:
                best_gain = gain
                best_thresh = t

        return best_gain, best_thresh

    def _split_crit(self, x_col, y):
        # Call defined splitting criterion (currently using ID3)
        # Can now use both numerical and categorical
        if self._is_numeric(x_col):
            gain, thresh = self._info_gain_numeric(x_col, y)
            return gain, thresh, "numeric"
        else:
            gain = self._info_gain_categorical(x_col, y)
            return gain, None, "categorical"

    def _build_tree(self, x, y, features, depth=0):
        # If y is made up of the same class, turn it into a node
        if len(set(y)) == 1:
            return y[0]

        # If less samples than minimum for splitting, stop
        if len(y) < self.min_samples_split:
           return self._majority_count(y)

        # If all features used up, return the majority class in y
        if len(features) == 0:
            return self._majority_count(y)

        # If maximum tree depth has been reached
        if depth >= self.max_depth:
            return self._majority_count(y)

        # HW3: Consider only random subset of features
        m = int(np.sqrt(len(features)))
        feat_subset = np.random.choice(features, m, replace=False)

        # Get the best feature from subset of features
        best_feature = None
        best_thresh = None
        best_gain = -1
        best_type = None

        # Get gain value, threshold, and feature type (for splitting) of best feature
        for feat in feat_subset:
            gain, thresh, ftype = self._split_crit(x[:, feat], y)

            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_thresh = thresh
                best_type = ftype

        # If best gain < min_gain, return
        if best_gain < self.min_gain:
            return self._majority_count(y)

        # Get the best feature and add it to the tree
        # NUMERICAL SPLIT
        if best_type == "numeric":
            # Create new tree node
            tree = {(best_feature, best_thresh): {}}

            # Split into two for values below and above best threshold
            left_side = x[:, best_feature] <= best_thresh
            right_side = x[:, best_feature] > best_thresh

            # If one side has no values, we have only one numerical category
            if np.sum(left_side) == 0 or np.sum(right_side) == 0:
                return self._majority_count(y)

            # Continue building next level of tree (both sides)
            tree[(best_feature, best_thresh)]['left'] = self._build_tree(
                x[left_side], y[left_side], features, depth + 1
            )

            tree[(best_feature, best_thresh)]['right'] = self._build_tree(
                x[right_side], y[right_side], features, depth + 1
            )

        # CATEGORICAL SPLIT (similar to before)
        else:
            # Create node with best feature
            tree = {best_feature: {}}

            for val in np.unique(x[:, best_feature]):
                # For each unique category, split tree off
                mask = x[:, best_feature] == val

                # Build next layer of tree, removing our feature
                tree[best_feature][val] = self._build_tree(
                    x[mask],
                    y[mask],
                    [f for f in features if f != best_feature],
                    depth + 1
                )

        return tree

    def _predict(self, x):
        # Create a copy of the tree
        tree = self.tree.copy()

        while isinstance(tree, dict):
            key = list(tree.keys())[0]

            # Numerical feature
            if isinstance(key, tuple):
                feature, thresh = key

                if x[feature] <= thresh:
                    tree = tree[key]['left']
                else:
                    tree = tree[key]['right']

            # Categorical feature (similar as before)
            else:
                feature = key
                val = x[feature]

                if val in tree[feature]:
                    tree = tree[feature][val]
                else:
                    return self._majority_count(self.y)

        return tree

    def _accuracy(self, x, y):
        # Correct guesses and total points
        correct = 0
        for i in range(len(x)):
            correct += 1 if self._predict(x[i]) == y[i] else 0
        return correct/len(x)


    def train(self, x, y):
        # Fit the training dataset to the model
        self.x = np.array(x)
        self.y = np.array(y)

        # Build the tree
        features = list(range(self.x.shape[1]))
        self.tree = self._build_tree(self.x, self.y, features)

        # Predict each point (no need to exclude self like kNN)
        return self._accuracy(self.x, self.y)

    def test(self, x_test, y_test):
        # Convert test data into Numpy array
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Predict each point
        return self._accuracy(x_test, y_test)

# Bootstrap sample function
def bootstrap_sample(x, y):
    n = len(x)
    indices = np.random.choice(n, n, replace=True)
    return x[indices], y[indices]

# RandomForest class
class RandomForest():
    def __init__(self, ntree=10, min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT, max_depth=DEFAULT_MAX_DEPTH, min_gain=DEFAULT_MIN_GAIN):
        self.x = None
        self.y = None
        self.ntree = ntree
        self.trees = []

        # Stopping criteria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain

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
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, min_gain=self.min_gain)
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