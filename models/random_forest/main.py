import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import RandomForest
from utils import stratified_k_fold_split, calculate_metrics

def evaluate_forest(X, y, numeric_features, pos_label, dataset_name):
    ntree_values = [1, 5, 10, 20, 30, 40, 50]

    results = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    print(f"\n--- Evaluating {dataset_name} ---")
    
    for ntree in ntree_values:
        print(f"Training Random Forest with ntree={ntree}...")
        metrics_per_fold = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        
        # 5-Fold Stratified Cross Validation
        for train_idx, val_idx in stratified_k_fold_split(X, y, k=5):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            rf = RandomForest(ntree=ntree, max_depth=10, min_samples_split=4)
            rf.fit(X_train, y_train, numeric_features)
            
            y_pred = rf.predict(X_val)
            
            acc, prec, rec, f1 = calculate_metrics(y_val, y_pred, pos_label)
            metrics_per_fold['acc'].append(acc)
            metrics_per_fold['prec'].append(prec)
            metrics_per_fold['rec'].append(rec)
            metrics_per_fold['f1'].append(f1)
            
        results['Accuracy'].append(np.mean(metrics_per_fold['acc']))
        results['Precision'].append(np.mean(metrics_per_fold['prec']))
        results['Recall'].append(np.mean(metrics_per_fold['rec']))
        results['F1 Score'].append(np.mean(metrics_per_fold['f1']))

    # Plotting the 4 graphs for this dataset
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Performance Metrics vs. ntree for {dataset_name}', fontsize=16)

    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    axes = axs.flatten()

    for idx, metric in enumerate(metrics_list):
        axes[idx].plot(ntree_values, results[metric], marker='o', linestyle='-', color='b')
        axes[idx].set_xlabel('Number of Trees (ntree)')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} vs ntree')
        axes[idx].grid(True)

        axes[idx].text(0.5, -0.15, f"Figure shows {metric} on {dataset_name} as ntree scales.", 
                       ha='center', va='center', transform=axes[idx].transAxes, fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f'{dataset_name.replace(" ", "_")}_results.png'
    plt.savefig(filename)
    print(f"Saved graphs to {filename}")

def main():
    # Dataset 1: WDBC (Breast Cancer) Processing
    try:

        wdbc = pd.read_csv('wdbc.csv') 

        wdbc.columns = wdbc.columns.str.strip()

        y_wdbc = wdbc['label'].values
        X_wdbc_df = wdbc.drop('label', axis=1)
        X_wdbc = X_wdbc_df.values
        
        numeric_wdbc = list(range(X_wdbc.shape[1]))

        evaluate_forest(X_wdbc, y_wdbc, numeric_wdbc, pos_label=1, dataset_name="WDBC Dataset")
        
    except FileNotFoundError:
        print("Error: wdbc.csv not found.")

    # Dataset 2: Loan Eligibility Processing
    try:
        loan = pd.read_csv('loan.csv')
        
        loan = loan.dropna() 
        if 'Loan_ID' in loan.columns:
            loan = loan.drop('Loan_ID', axis=1)
            
        y_loan = loan['label'].values
        X_loan_df = loan.drop('label', axis=1)
        
        X_loan = X_loan_df.values
        numeric_loan = []
        for i, col in enumerate(X_loan_df.columns):
            if pd.api.types.is_numeric_dtype(X_loan_df[col]):
                numeric_loan.append(i)

        evaluate_forest(X_loan, y_loan, numeric_loan, pos_label=1, dataset_name="Loan Dataset")
        
    except FileNotFoundError:
        print("Error: loan.csv not found.")

    # Extra Credit 1: Raisin Dataset Processing

    try:
        raisin = pd.read_csv('raisin.csv')
        raisin.columns = raisin.columns.str.strip()
        
        y_raisin = raisin['label'].values
        X_raisin_df = raisin.drop('label', axis=1)
        X_raisin = X_raisin_df.values
        
        numeric_raisin = list(range(X_raisin.shape[1]))
        
        evaluate_forest(X_raisin, y_raisin, numeric_raisin, pos_label=1, dataset_name="Raisin Dataset")
        
    except FileNotFoundError:
        print("Error: raisin.csv not found.")

    # Extra Credit 2: Titanic Dataset Processing

    try:
        titanic = pd.read_csv('titanic.csv')
        titanic.columns = titanic.columns.str.strip()
        
        y_titanic = titanic['label'].values
        X_titanic_df = titanic.drop('label', axis=1)
        X_titanic = X_titanic_df.values
        
        numeric_titanic = [2, 3, 4, 5]
        
        evaluate_forest(X_titanic, y_titanic, numeric_titanic, pos_label=1, dataset_name="Titanic Dataset")
        
    except FileNotFoundError:
        print("Error: titanic.csv not found.")

if __name__ == "__main__":
    main()