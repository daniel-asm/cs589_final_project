import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_tabular_dataset
from utils.evaluation import grid_search_cv
from models.knn import KNN
from models.decision_tree import DecisionTreeClassifier

def run_rice_experiments():
    print("Loading Rice Dataset...")
    X, y = load_tabular_dataset('datasets/rice.csv', target_column='label')

    os.makedirs('latex_source', exist_ok=True)

    print("\nRunning K-Nearest Neighbors")
    knn_param_grid = {
        'k_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41]
    }
    
    knn_best_params, knn_results = grid_search_cv(KNN, X, y, knn_param_grid, k=10)

    print("\nKNN Full Results")
    for res in knn_results:
        print(f"k: {res['params']['k_neighbors']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    k_vals = [res['params']['k_neighbors'] for res in knn_results]
    knn_f1_vals = [res['f1'] for res in knn_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, knn_f1_vals, marker='o', color='red', linestyle='dashed', linewidth=2, markersize=8)
    plt.title(f"KNN F1-Score vs. k (Rice)\nBest k: {knn_best_params['k_neighbors']}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(k_vals)
    plt.savefig('latex_source/knn_rice_f1_curve.png')
    plt.show()

    print("\nRunning Decision Tree")
    
    dt_param_grid = {
        'max_depth': [7, 9, 11, 15, None],
        'criterion': ['entropy', 'gini']
    }
    
    dt_best_params, dt_results = grid_search_cv(DecisionTreeClassifier, X, y, dt_param_grid, k=10)

    print("\nDecision Tree Full Results")
    print(f"{'Max Depth':<12} | {'Split Criterion':<15} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 55)
    for res in dt_results:
        depth_str = str(res['params']['max_depth'])
        crit_str = res['params']['criterion'].capitalize()
        print(f"{depth_str:<12} | {crit_str:<15} | {res['accuracy']:.4f}     | {res['f1']:.4f}")
        
    depth_labels = [str(d) for d in dt_param_grid['max_depth']]
    
    entropy_f1 = [res['f1'] for res in dt_results if res['params']['criterion'] == 'entropy']
    gini_f1 = [res['f1'] for res in dt_results if res['params']['criterion'] == 'gini']
    
    plt.figure(figsize=(8, 6))
    plt.plot(depth_labels, entropy_f1, marker='s', color='green', linestyle='solid', linewidth=2, markersize=8, label='Entropy')
    plt.plot(depth_labels, gini_f1, marker='^', color='orange', linestyle='dashed', linewidth=2, markersize=8, label='Gini')
    
    plt.title(f"Decision Tree F1-Score vs. Max Depth & Criterion (Rice)\nBest Setup: {dt_best_params}")
    plt.xlabel("Maximum Tree Depth (None = Unrestricted)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.legend()
    plt.savefig('latex_source/dt_rice_f1_curve.png')
    plt.show()

if __name__ == "__main__":
    run_rice_experiments()