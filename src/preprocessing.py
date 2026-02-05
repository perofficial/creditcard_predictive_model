import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_engineer(df):
    """Include tutta la logica di pulizia e feature engineering originale."""
    # Gestione missing values (Mediana per numerici, Moda per categorici)
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in df.columns[df.isnull().any()]:
        if col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Feature Engineering originale
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: -x / 365.25 if x < 0 else 0).astype(int)
    df['IS_UNEMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop colonne originali come da tuo codice
    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True)
    return df

def scale_and_encode(df):
    """One-Hot Encoding e Scaling rigoroso."""
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = [col for col in df.select_dtypes(include=['number']).columns 
                          if col not in ['ID', 'TARGET']]
    
    # Get dummies (drop_first=True per evitare multicollinearità)
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Scaling
    scaler = StandardScaler()
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    return df_encoded, scaler