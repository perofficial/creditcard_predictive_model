"""
Feature Engineering Module
Gestisce la creazione di nuove feature derivate
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe per feature engineering"""
    
    def __init__(self):
        """Inizializza il FeatureEngineer"""
        self.created_features = []
    
    def create_age_features(self, df: pd.DataFrame, days_birth_col: str = 'DAYS_BIRTH') -> pd.DataFrame:
        """
        Crea feature relative all'età
        
        Args:
            df: DataFrame input
            days_birth_col: Nome colonna con giorni dalla nascita
            
        Returns:
            DataFrame con nuove feature età
        """
        df = df.copy()
        
        if days_birth_col not in df.columns:
            logger.warning(f"Colonna '{days_birth_col}' non trovata")
            return df
        
        # Età in anni
        df['AGE_YEARS'] = (-df[days_birth_col] / 365.25).astype(int)
        logger.info("Feature 'AGE_YEARS' creata")
        
        # Categorie età
        df['AGE_GROUP'] = pd.cut(
            df['AGE_YEARS'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        logger.info("Feature 'AGE_GROUP' creata")
        
        # Rimozione colonna originale
        df = df.drop(columns=[days_birth_col])
        
        self.created_features.extend(['AGE_YEARS', 'AGE_GROUP'])
        
        return df
    
    def create_employment_features(
        self, 
        df: pd.DataFrame, 
        days_employed_col: str = 'DAYS_EMPLOYED'
    ) -> pd.DataFrame:
        """
        Crea feature relative all'impiego
        
        Args:
            df: DataFrame input
            days_employed_col: Nome colonna con giorni dall'assunzione
            
        Returns:
            DataFrame con nuove feature impiego
        """
        df = df.copy()
        
        if days_employed_col not in df.columns:
            logger.warning(f"Colonna '{days_employed_col}' non trovata")
            return df
        
        # Anni di impiego (valori negativi indicano impiego, positivi disoccupazione)
        df['YEARS_EMPLOYED'] = df[days_employed_col].apply(
            lambda x: -x / 365.25 if x < 0 else 0
        ).astype(int)
        logger.info("Feature 'YEARS_EMPLOYED' creata")
        
        # Flag disoccupazione
        df['IS_UNEMPLOYED'] = (df[days_employed_col] > 0).astype(int)
        logger.info("Feature 'IS_UNEMPLOYED' creata")
        
        # Stabilità lavorativa
        df['EMPLOYMENT_STABILITY'] = pd.cut(
            df['YEARS_EMPLOYED'],
            bins=[-1, 0, 2, 5, 10, 100],
            labels=['Unemployed', '0-2y', '3-5y', '6-10y', '10+y']
        )
        logger.info("Feature 'EMPLOYMENT_STABILITY' creata")
        
        # Rimozione colonna originale
        df = df.drop(columns=[days_employed_col])
        
        self.created_features.extend(['YEARS_EMPLOYED', 'IS_UNEMPLOYED', 'EMPLOYMENT_STABILITY'])
        
        return df
    
    def create_income_features(
        self, 
        df: pd.DataFrame, 
        income_col: str = 'AMT_INCOME_TOTAL'
    ) -> pd.DataFrame:
        """
        Crea feature relative al reddito
        
        Args:
            df: DataFrame input
            income_col: Nome colonna reddito
            
        Returns:
            DataFrame con nuove feature reddito
        """
        df = df.copy()
        
        if income_col not in df.columns:
            logger.warning(f"Colonna '{income_col}' non trovata")
            return df
        
        # Log income (per gestire skewness)
        df['LOG_INCOME'] = np.log1p(df[income_col])
        logger.info("Feature 'LOG_INCOME' creata")
        
        # Fasce di reddito
        df['INCOME_BRACKET'] = pd.qcut(
            df[income_col],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
        logger.info("Feature 'INCOME_BRACKET' creata")
        
        # Reddito per membro famiglia
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['INCOME_PER_MEMBER'] = df[income_col] / (df['CNT_FAM_MEMBERS'] + 1)
            logger.info("Feature 'INCOME_PER_MEMBER' creata")
            self.created_features.append('INCOME_PER_MEMBER')
        
        self.created_features.extend(['LOG_INCOME', 'INCOME_BRACKET'])
        
        return df
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature relative alla famiglia
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame con nuove feature famiglia
        """
        df = df.copy()
        
        # Ha figli
        if 'CNT_CHILDREN' in df.columns:
            df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
            logger.info("Feature 'HAS_CHILDREN' creata")
            self.created_features.append('HAS_CHILDREN')
        
        # Dimensione famiglia
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['FAMILY_SIZE'] = pd.cut(
                df['CNT_FAM_MEMBERS'],
                bins=[0, 1, 2, 3, 10],
                labels=['Single', 'Small', 'Medium', 'Large']
            )
            logger.info("Feature 'FAMILY_SIZE' creata")
            self.created_features.append('FAMILY_SIZE')
        
        return df
    
    def create_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature relative ai beni posseduti
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame con nuove feature beni
        """
        df = df.copy()
        
        # Conteggio beni
        asset_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        available_cols = [col for col in asset_cols if col in df.columns]
        
        if available_cols:
            df['TOTAL_ASSETS'] = df[available_cols].sum(axis=1)
            logger.info("Feature 'TOTAL_ASSETS' creata")
            self.created_features.append('TOTAL_ASSETS')
        
        return df
    
    def create_contact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature relative ai contatti
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame con nuove feature contatti
        """
        df = df.copy()
        
        # Conteggio metodi di contatto
        contact_cols = ['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
        available_cols = [col for col in contact_cols if col in df.columns]
        
        if available_cols:
            df['CONTACT_METHODS'] = df[available_cols].sum(axis=1)
            logger.info("Feature 'CONTACT_METHODS' creata")
            self.created_features.append('CONTACT_METHODS')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature di interazione tra variabili
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame con feature di interazione
        """
        df = df.copy()
        
        # Age * Income
        if 'AGE_YEARS' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['AGE_INCOME_INTERACTION'] = df['AGE_YEARS'] * df['AMT_INCOME_TOTAL'] / 1000000
            logger.info("Feature 'AGE_INCOME_INTERACTION' creata")
            self.created_features.append('AGE_INCOME_INTERACTION')
        
        # Employment * Income
        if 'YEARS_EMPLOYED' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['EMPLOYMENT_INCOME_INTERACTION'] = df['YEARS_EMPLOYED'] * df['AMT_INCOME_TOTAL'] / 1000000
            logger.info("Feature 'EMPLOYMENT_INCOME_INTERACTION' creata")
            self.created_features.append('EMPLOYMENT_INCOME_INTERACTION')
        
        return df
    
    def apply_all_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica tutte le trasformazioni di feature engineering
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame con tutte le nuove feature
        """
        logger.info("Inizio feature engineering completo")
        
        df = self.create_age_features(df)
        df = self.create_employment_features(df)
        df = self.create_income_features(df)
        df = self.create_family_features(df)
        df = self.create_asset_features(df)
        df = self.create_contact_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering completato. {len(self.created_features)} nuove feature create")
        logger.info(f"Feature create: {self.created_features}")
        
        return df


if __name__ == "__main__":
    # Test
    from data_loader import DataLoader
    
    loader = DataLoader("../../data/raw/credit_scoring.csv")
    df = loader.load_data()
    
    engineer = FeatureEngineer()
    df_engineered = engineer.apply_all_transformations(df)
    
    print(f"Shape originale: {loader.df.shape}")
    print(f"Shape dopo engineering: {df_engineered.shape}")
    print(f"\nNuove colonne create:")
    print(df_engineered.columns.tolist())
