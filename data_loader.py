"""
Data Loader Module
Gestisce il caricamento e la validazione iniziale dei dati
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Classe per caricare e validare i dati di credit scoring"""
    
    REQUIRED_COLUMNS = [
        'ID', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE',
        'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Inizializza il DataLoader
        
        Args:
            data_path: Path al file CSV dei dati
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Carica i dati dal file CSV
        
        Args:
            data_path: Path al file CSV (opzionale se già specificato)
            
        Returns:
            DataFrame con i dati caricati
        """
        path = data_path or self.data_path
        
        if path is None:
            raise ValueError("Data path non specificato")
        
        logger.info(f"Caricamento dati da: {path}")
        
        try:
            self.df = pd.read_csv(path)
            logger.info(f"Dati caricati con successo: {self.df.shape[0]} righe, {self.df.shape[1]} colonne")
            
            # Validazione base
            self._validate_data()
            
            return self.df
            
        except FileNotFoundError:
            logger.error(f"File non trovato: {path}")
            raise
        except Exception as e:
            logger.error(f"Errore durante caricamento dati: {str(e)}")
            raise
    
    def _validate_data(self) -> None:
        """Valida che i dati abbiano le colonne richieste"""
        if self.df is None:
            raise ValueError("Nessun dato caricato")
        
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        
        if missing_cols:
            logger.warning(f"Colonne mancanti (verranno ignorate se non essenziali): {missing_cols}")
        
        logger.info("Validazione dati completata")
    
    def get_data_info(self) -> dict:
        """
        Restituisce informazioni sui dati caricati
        
        Returns:
            Dictionary con statistiche sui dati
        """
        if self.df is None:
            raise ValueError("Nessun dato caricato")
        
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.to_dict()
        }
    
    def check_target_distribution(self, target_col: str = 'TARGET') -> dict:
        """
        Verifica la distribuzione della variabile target
        
        Args:
            target_col: Nome della colonna target
            
        Returns:
            Dictionary con statistiche sulla distribuzione
        """
        if self.df is None:
            raise ValueError("Nessun dato caricato")
        
        if target_col not in self.df.columns:
            raise ValueError(f"Colonna {target_col} non trovata")
        
        value_counts = self.df[target_col].value_counts()
        value_proportions = self.df[target_col].value_counts(normalize=True)
        
        return {
            'counts': value_counts.to_dict(),
            'proportions': value_proportions.to_dict(),
            'class_ratio': value_counts.min() / value_counts.max()
        }
    
    @staticmethod
    def download_data(url: str, save_path: str) -> None:
        """
        Scarica i dati da URL
        
        Args:
            url: URL del dataset
            save_path: Path dove salvare il file
        """
        import urllib.request
        
        logger.info(f"Download dati da: {url}")
        
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"Dati salvati in: {save_path}")
        except Exception as e:
            logger.error(f"Errore durante download: {str(e)}")
            raise


if __name__ == "__main__":
    # Test
    loader = DataLoader()
    
    # Download se necessario
    # DataLoader.download_data(
    #     url="https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv",
    #     save_path="../../data/raw/credit_scoring.csv"
    # )
    
    # Load
    df = loader.load_data("../../data/raw/credit_scoring.csv")
    print(loader.get_data_info())
    print(loader.check_target_distribution())
