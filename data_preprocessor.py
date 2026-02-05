"""
Data Preprocessor Module
Gestisce preprocessing, pulizia dati e imputazione valori mancanti
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe per preprocessing e pulizia dei dati"""
    
    def __init__(self):
        """Inizializza il preprocessor"""
        self.scaler = StandardScaler()
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names = []
        self.is_fitted = False
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'mode'
    ) -> pd.DataFrame:
        """
        Gestisce i valori mancanti
        
        Args:
            df: DataFrame input
            numeric_strategy: Strategia per valori numerici ('mean', 'median', 'drop')
            categorical_strategy: Strategia per categorici ('mode', 'drop', 'unknown')
            
        Returns:
            DataFrame con valori mancanti gestiti
        """
        df = df.copy()
        
        # Identifica colonne con valori mancanti
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("Nessun valore mancante da gestire")
            return df
        
        logger.info(f"Gestione valori mancanti in {len(missing_cols)} colonne")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Gestione valori numerici
        for col in missing_cols:
            if col in numeric_cols:
                if numeric_strategy == 'median':
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    logger.info(f"Colonna '{col}': imputata con mediana = {fill_value:.2f}")
                elif numeric_strategy == 'mean':
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    logger.info(f"Colonna '{col}': imputata con media = {fill_value:.2f}")
        
        # Gestione valori categorici
        for col in missing_cols:
            if col in categorical_cols:
                if categorical_strategy == 'mode':
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    logger.info(f"Colonna '{col}': imputata con moda = {fill_value}")
                elif categorical_strategy == 'unknown':
                    df[col].fillna('Unknown', inplace=True)
                    logger.info(f"Colonna '{col}': imputata con 'Unknown'")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove righe duplicate
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame senza duplicati
        """
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            logger.warning(f"Trovate {n_duplicates} righe duplicate. Rimozione...")
            df = df.drop_duplicates()
            logger.info(f"Duplicati rimossi. Shape finale: {df.shape}")
        else:
            logger.info("Nessun duplicato trovato")
        
        return df
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encoding delle variabili categoriche
        
        Args:
            df: DataFrame input
            columns: Lista colonne da encodare (None = tutte le categoriche)
            method: Metodo di encoding ('onehot', 'label')
            
        Returns:
            DataFrame con variabili encodate
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not columns:
            logger.info("Nessuna colonna categorica da encodare")
            return df
        
        logger.info(f"Encoding di {len(columns)} colonne categoriche con metodo: {method}")
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            logger.info(f"One-hot encoding completato. Nuove dimensioni: {df.shape}")
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Label encoding completato")
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scaling delle feature numeriche
        
        Args:
            df: DataFrame input
            columns: Lista colonne da scalare (None = tutte le numeriche)
            fit: Se True, fit del scaler. Se False, usa scaler già fittato
            
        Returns:
            DataFrame con feature scalate
        """
        df = df.copy()
        
        if columns is None:
            # Esclude ID e TARGET
            columns = [col for col in df.select_dtypes(include=['number']).columns 
                      if col not in ['ID', 'TARGET']]
        
        if not columns:
            logger.info("Nessuna colonna numerica da scalare")
            return df
        
        logger.info(f"Scaling di {len(columns)} colonne numeriche")
        
        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
            self.numeric_columns = columns
            self.is_fitted = True
            logger.info("Scaler fittato e applicato")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler non ancora fittato. Usa fit=True")
            df[columns] = self.scaler.transform(df[columns])
            logger.info("Scaler applicato")
        
        return df
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = 'TARGET',
        id_col: str = 'ID',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Pipeline completa di preprocessing
        
        Args:
            df: DataFrame input
            target_col: Nome colonna target
            id_col: Nome colonna ID
            fit: Se True, fit degli encoder/scaler
            
        Returns:
            Tuple (X preprocessato, y target) o solo X se target non presente
        """
        logger.info("Inizio pipeline di preprocessing")
        
        df = df.copy()
        
        # 1. Rimozione duplicati
        df = self.remove_duplicates(df)
        
        # 2. Gestione valori mancanti
        df = self.handle_missing_values(df)
        
        # 3. Separazione target (se presente)
        y = None
        if target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])
            logger.info(f"Target '{target_col}' separato")
        
        # 4. Rimozione ID
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
            logger.info(f"Colonna ID '{id_col}' rimossa")
        
        # 5. Identificazione colonne
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # 6. Encoding categoriche
        df = self.encode_categorical(df, columns=self.categorical_columns)
        
        # 7. Scaling numeriche
        df = self.scale_features(df, columns=self.numeric_columns, fit=fit)
        
        # 8. Salva feature names
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Preprocessing completato. Shape finale: {df.shape}")
        
        return df, y


if __name__ == "__main__":
    # Test
    from data_loader import DataLoader
    
    loader = DataLoader("../../data/raw/credit_scoring.csv")
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {preprocessor.feature_names[:10]}...")
