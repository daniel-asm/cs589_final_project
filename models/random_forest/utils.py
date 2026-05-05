import numpy as np

def calculate_metrics(y_true, y_pred, pos_label):
    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))
    TN = np.sum((y_true != pos_label) & (y_pred != pos_label))

    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def stratified_k_fold_split(X, y, k=5):
    classes, y_indices = np.unique(y, return_inverse=True)
    
    class_indices = [np.where(y == c)[0] for c in classes]
    
    for indices in class_indices:
        np.random.shuffle(indices)
        
    folds = [[] for _ in range(k)]
    
    for indices in class_indices:
        fold_splits = np.array_split(indices, k)
        for i in range(k):
            folds[i].extend(fold_splits[i])
            
    for i in range(k):
        val_idx = folds[i]
        train_idx = []
        for j in range(k):
            if i != j:
                train_idx.extend(folds[j])
                
        yield np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)