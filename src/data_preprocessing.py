import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset caricato: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato in: {filepath}")

def clean_and_engineer_features(df):
    """Gestisce imputazione e creazione nuove feature"""
    # 1. Imputazione
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Feature Engineering
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: -x / 365.25 if x < 0 else 0).astype(int)
    df['IS_UNEMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop colonne originali e ID
    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ID'], inplace=True)
    return df

def preprocess_for_model(df):
    """Encoding e Scaling"""
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    target = 'TARGET'
    features = [c for c in df_encoded.columns if c != target]
    
    # Scaling
    scaler = StandardScaler()
    df_encoded[features] = scaler.fit_transform(df_encoded[features])
    
    return df_encoded, features