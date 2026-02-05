# src/visualization.py
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_style():
    """Imposta lo stile base per i grafici."""
    sns.set_style("whitegrid")
    plt.rc('font', size=10)

def plot_numeric_distributions(df, exclude_cols=['ID', 'TARGET']):
    """Genera istogrammi per le feature numeriche."""
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude_cols]
    num_plots = len(numeric_cols)
    cols_per_row = 3
    rows = math.ceil(num_plots / cols_per_row)

    plt.figure(figsize=(15, rows * 4))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, cols_per_row, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuzione di {col}')
        plt.tight_layout()
    plt.show()

def plot_categorical_counts(df, exclude_cols=['ID']):
    """Genera countplots per le feature categoriche."""
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in exclude_cols]
    # Aggiungi colonne flag o custom se necessario
    if 'IS_UNEMPLOYED' in df.columns:
        categorical_cols.append('IS_UNEMPLOYED')

    num_plots = len(categorical_cols)
    cols_per_row = 3
    rows = math.ceil(num_plots / cols_per_row)

    plt.figure(figsize=(15, rows * 5))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(rows, cols_per_row, i)
        if col in df.columns:
            sns.countplot(data=df, y=col, order=df[col].value_counts().index)
            plt.title(f'Frequenza di {col}')
            plt.tight_layout()
    plt.show()

def plot_correlations(df, exclude_cols=['ID', 'TARGET']):
    """Calcola e visualizza la heatmap di correlazione."""
    numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude_cols]
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice di Correlazione Feature Numeriche')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """Visualizza l'importanza delle feature per modelli tree-based."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis', hue=importances.index, legend=False)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importanza')
    plt.tight_layout()
    plt.show()