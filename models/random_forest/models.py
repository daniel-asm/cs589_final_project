import numpy as np

def calculate_entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X_column, y, is_numeric):
    parent_entropy = calculate_entropy(y)
    if len(y) == 0:
        return 0, None

    if is_numeric:
        X_numeric = X_column.astype(float)
        threshold = np.mean(X_numeric)
        left_mask = X_numeric <= threshold
        right_mask = X_numeric > threshold

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0, threshold

        n = len(y)
        left_entropy = calculate_entropy(y[left_mask])
        right_entropy = calculate_entropy(y[right_mask])
        
        weighted_child_entropy = (np.sum(left_mask) / n) * left_entropy + \
                                 (np.sum(right_mask) / n) * right_entropy
                                 
        return parent_entropy - weighted_child_entropy, threshold
    else:
        values, counts = np.unique(X_column, return_counts=True)
        weighted_child_entropy = 0
        for value, count in zip(values, counts):
            subset_y = y[X_column == value]
            weighted_child_entropy += (count / len(y)) * calculate_entropy(subset_y)
            
        return parent_entropy - weighted_child_entropy, None


class Node:
    def __init__(self, feature_index=None, is_numeric=False, threshold=None, 
                 branches=None, left=None, right=None, prediction=None, fallback_class=None):
        self.feature_index = feature_index
        self.is_numeric = is_numeric
        self.threshold = threshold
        self.branches = branches  
        self.left = left          
        self.right = right       
        self.prediction = prediction
        self.fallback_class = fallback_class

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.numeric_features = []

    def fit(self, X, y, numeric_features):
        self.numeric_features = numeric_features
        feature_indices = list(range(X.shape[1]))
        self.root = self._build_tree(X, y, feature_indices, depth=0)

    def _build_tree(self, X, y, feature_indices, depth):
        unique_labels, counts = np.unique(y, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]

        if len(unique_labels) == 1 or len(feature_indices) == 0 or \
           depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(prediction=majority_class)

        m = int(np.round(np.sqrt(X.shape[1])))
        m = max(1, min(m, len(feature_indices))) 
        subset_features = np.random.choice(feature_indices, m, replace=False)

        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        for feature_idx in subset_features:
            is_numeric = feature_idx in self.numeric_features
            gain, threshold = information_gain(X[:, feature_idx], y, is_numeric)
            
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold

        if best_gain <= 0:
            return Node(prediction=majority_class)

        is_best_numeric = best_feature_idx in self.numeric_features
        
        if is_best_numeric:
            feature_col = X[:, best_feature_idx].astype(float)
            
            left_mask = feature_col <= best_threshold
            right_mask = feature_col > best_threshold
            
            left_node = self._build_tree(X[left_mask], y[left_mask], feature_indices, depth + 1)
            right_node = self._build_tree(X[right_mask], y[right_mask], feature_indices, depth + 1)
            
            return Node(feature_index=best_feature_idx, is_numeric=True, threshold=best_threshold, 
                        left=left_node, right=right_node, fallback_class=majority_class)
        else:
            branches = {}
            unique_values = np.unique(X[:, best_feature_idx])
            remaining_features = [f for f in feature_indices if f != best_feature_idx]
            
            for value in unique_values:
                subset_mask = X[:, best_feature_idx] == value
                branches[value] = self._build_tree(X[subset_mask], y[subset_mask], remaining_features, depth + 1)
                
            return Node(feature_index=best_feature_idx, is_numeric=False, branches=branches, fallback_class=majority_class)

    def _predict_instance(self, node, x):
        if node.prediction is not None:
            return node.prediction
            
        feature_val = x[node.feature_index]
        result = None
        
        try:
            if node.is_numeric:
                if float(feature_val) <= node.threshold:
                    result = self._predict_instance(node.left, x)
                else:
                    result = self._predict_instance(node.right, x)
            else:
                if feature_val in node.branches:
                    result = self._predict_instance(node.branches[feature_val], x)
        except Exception:
            pass
            
        if result is None:
            result = node.fallback_class
            
        return result

    def predict(self, X):
        return np.array([self._predict_instance(self.root, x) for x in X])


class RandomForest:
    def __init__(self, ntree=10, max_depth=10, min_samples_split=2):
        self.ntree = ntree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.numeric_features = []

    def fit(self, X, y, numeric_features):
        self.numeric_features = numeric_features
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.ntree):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap, numeric_features)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        final_predictions = []
        for i in range(X.shape[0]):
            unique_labels, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_predictions.append(unique_labels[np.argmax(counts)])
            
        return np.array(final_predictions)