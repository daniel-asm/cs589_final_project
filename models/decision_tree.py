import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, children=None, left=None, right=None, value=None):
        self.feature_index = feature_index 
        self.threshold = threshold
        self.children = children if children is not None else {}
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, min_gain=0.0, max_features=None, **kwargs):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_features = max_features

    def _entropy(self, y):
        label_counts = Counter(y)
        total_instances = len(y)
        entropy = 0.0
        for count in label_counts.values():
            probability = count / total_instances
            if probability > 0:
                entropy -= probability * np.log2(probability)
        return entropy

    def _gini(self, y):
        label_counts = Counter(y)
        total_instances = len(y)
        gini = 1.0
        for count in label_counts.values():
            probability = count / total_instances
            gini -= probability ** 2
        return gini
            
    def _calculate_numerical_gain(self, X_column, y):
        X_column = X_column.astype(float)
        sort_indices = np.argsort(X_column)
        X_sorted = X_column[sort_indices]
        y_sorted = y[sort_indices]
        
        unique_vals = np.unique(X_sorted)
        if len(unique_vals) <= 1:
            return -1, None
        
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        
        if len(thresholds) > 15:
            thresholds = np.percentile(X_column, np.linspace(10, 90, 9))
            
        parent_score = self._gini(y) if self.criterion == 'gini' else self._entropy(y)
        best_gain = -1
        best_threshold = None
        total_instances = len(y)
        
        for thresh in thresholds:
            left_mask = X_column <= thresh
            right_mask = X_column > thresh
            y_left, y_right = y[left_mask], y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            weight_l, weight_r = len(y_left) / total_instances, len(y_right) / total_instances
            score_l = self._gini(y_left) if self.criterion == 'gini' else self._entropy(y_left)
            score_r = self._gini(y_right) if self.criterion == 'gini' else self._entropy(y_right)
            
            gain = parent_score - (weight_l * score_l + weight_r * score_r)
            if gain > best_gain:
                best_gain, best_threshold = gain, thresh
                
        return best_gain, best_threshold

    def fit(self, X_train, y_train, **kwargs):
        X_train = np.array(X_train).T
        y_train = np.array(y_train).flatten()
                
        initial_features = list(range(X_train.shape[1]))
        self.root = self._build_tree(X_train, y_train, initial_features, depth=0)

    def _build_tree(self, X, y, available_features, depth):
        label_counts = Counter(y)
        majority_class = label_counts.most_common(1)[0][0]

        if len(label_counts) == 1 or len(available_features) == 0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=majority_class)

        features_to_evaluate = available_features
        if self.max_features == 'sqrt':
            num_features = max(1, int(np.sqrt(len(available_features))))
            features_to_evaluate = np.random.choice(available_features, num_features, replace=False)

        best_gain, best_feature, best_threshold = -1, None, None

        for feature_idx in features_to_evaluate:
            if self.is_numeric_feature[feature_idx]:
                gain, threshold = self._calculate_numerical_gain(X[:, feature_idx], y)
            else:
                gain, threshold = self._calculate_categorical_gain(X[:, feature_idx], y)
                
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature_idx, threshold

        if best_gain <= self.min_gain:
             return Node(value=majority_class)

        node = Node(feature_index=best_feature, value=majority_class)
        
        if self.is_numeric_feature[best_feature]:
            node.is_numeric = True
            node.threshold = best_threshold
            left_mask = X[:, best_feature].astype(float) <= best_threshold
            right_mask = X[:, best_feature].astype(float) > best_threshold
            
            node.left = self._build_tree(X[left_mask], y[left_mask], available_features, depth + 1)
            node.right = self._build_tree(X[right_mask], y[right_mask], available_features, depth + 1)
        else:
            node.is_numeric = False
            remaining_features = [f for f in available_features if f != best_feature]
            unique_values = np.unique(X[:, best_feature])
            
            for value in unique_values:
                indices = np.where(X[:, best_feature] == value)[0]
                node.children[value] = self._build_tree(X[indices], y[indices], remaining_features, depth + 1)

        return node

    def predict(self, X_test):
        X_test = np.array(X_test).T
        return np.array([self._traverse_tree(x, self.root) for x in X_test])

    def _traverse_tree(self, x, node):
        if node.left is None and node.right is None and not node.children:
            return node.value
            
        feature_val = x[node.feature_index]
        
        if node.is_numeric:
            if float(feature_val) <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if feature_val in node.children:
                return self._traverse_tree(x, node.children[feature_val])
            else:
                return node.value