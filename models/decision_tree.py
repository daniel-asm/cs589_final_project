"""
Original Author: Gabriel Lojo
"""
import numpy as np

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_gini(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def information_gain_continuous(X_column, y, threshold, criterion='entropy'):
    if criterion == 'gini':
        parent_score = calculate_gini(y)
    else:
        parent_score = calculate_entropy(y)
        
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
        
    weight_left = np.sum(left_mask) / len(y)
    weight_right = np.sum(right_mask) / len(y)
    
    if criterion == 'gini':
        child_score = (weight_left * calculate_gini(y[left_mask]) + 
                       weight_right * calculate_gini(y[right_mask]))
    else:
        child_score = (weight_left * calculate_entropy(y[left_mask]) + 
                       weight_right * calculate_entropy(y[right_mask]))
                     
    return parent_score - child_score

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

class DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, **kwargs):
        """ **kwargs absorbs unexpected pipeline parameters """
        self.root = None
        self.criterion = criterion 
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split

    def fit(self, X_train, y_train, **kwargs):
        X = X_train.T
        y = y_train.flatten()
        initial_features = list(range(X.shape[1]))
        self.root = self._build_tree(X, y, initial_features, depth=0)

    def _build_tree(self, X, y, feature_indices, depth):
        unique_labels, counts = np.unique(y, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]

        if len(unique_labels) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(prediction=majority_class)

        if len(feature_indices) == 0:
            return Node(prediction=majority_class)

        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        for i in feature_indices:
            X_col = X[:, i]
            thresholds = np.unique(X_col)
            
            if len(thresholds) > 15:
                thresholds = np.percentile(X_col, np.linspace(10, 90, 9))
            
            for t in thresholds:
                gain = information_gain_continuous(X_col, y, t, self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = i
                    best_threshold = t

        if best_gain <= 0:
            return Node(prediction=majority_class)

        left_indices = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_idx] > best_threshold)[0]

        left_child = self._build_tree(X[left_indices], y[left_indices], feature_indices, depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], feature_indices, depth + 1)

        return Node(feature_index=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child)

    def predict(self, X_test):
        X_test_T = X_test.T
        return np.array([self._predict_instance(self.root, x) for x in X_test_T])

    def _predict_instance(self, node, x):
        if node.prediction is not None:
            return node.prediction
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_instance(node.left, x)
        else:
            return self._predict_instance(node.right, x)