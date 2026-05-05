import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, ntree=10, criterion='entropy', max_depth=None, min_samples_split=2, min_gain=0.0, **kwargs):
        self.ntree = ntree
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = []

    def fit(self, X_train, y_train, **kwargs):
        self.trees = []

        X_train_rows = np.array(X_train).T
        y_train_flat = np.array(y_train).flatten()
        n_samples = len(X_train_rows)

        for _ in range(self.ntree):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_train_rows[bootstrap_indices]
            y_bootstrap = y_train_flat[bootstrap_indices]

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                max_features='sqrt'
            )
            
            tree.fit(X_bootstrap.T, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X_test):
        tree_predictions = np.array([tree.predict(X_test) for tree in self.trees])
        
        final_predictions = []
        for i in range(tree_predictions.shape[1]):
            sample_preds = tree_predictions[:, i]
            most_common = Counter(sample_preds).most_common(1)[0][0]
            final_predictions.append(most_common)
            
        return np.array(final_predictions)