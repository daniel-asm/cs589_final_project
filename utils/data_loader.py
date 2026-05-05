import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import datasets

def load_tabular_dataset(filepath, target_column='label', explicit_cat_cols=None):
    """
    Universal parser for CSVs. Dynamically finds numerical and categorical columns.
    """
    df = pd.read_csv(filepath)

    if explicit_cat_cols:
        for col in explicit_cat_cols:
            df[col] = df[col].astype('category')
    
    y_raw = df[target_column].values
    X_raw = df.drop(target_column, axis=1)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).reshape(1, -1)
    
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_raw.select_dtypes(exclude=['object', 'category']).columns.tolist()

    for col in cat_cols:
        X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0])
    for col in num_cols:
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
        
    scaler = StandardScaler()
    num_data = scaler.fit_transform(X_raw[num_cols]) if num_cols else np.empty((X_raw.shape[0], 0))
    
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first') 
        cat_data = encoder.fit_transform(X_raw[cat_cols])
        X = np.hstack((num_data, cat_data)).T 
    else:
        X = num_data.T
        
    return X, y

def load_digits_dataset():
    """Loads the 8x8 digits dataset from sklearn"""
    X_raw, y_raw = datasets.load_digits(return_X_y=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).T 
    y = y_raw.reshape(1, -1)
    
    return X, y