"""
Base Model Module
Classe astratta base per tutti i modelli di credit scoring
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Classe astratta base per modelli di credit scoring"""
    
    def __init__(self, model_name: str = "BaseModel"):
        """
        Inizializza il modello base
        
        Args:
            model_name: Nome del modello
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_time = None
        
    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        Costruisce l'architettura del modello
        
        Returns:
            Modello scikit-learn
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Addestra il modello
        
        Args:
            X_train: Feature di training
            y_train: Target di training
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dictionary con metriche di training
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Effettua predizioni
        
        Args:
            X: Feature per predizione
            
        Returns:
            Array con predizioni (0 o 1)
        """
        if not self.is_trained:
            raise ValueError(f"Modello {self.model_name} non ancora addestrato")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Effettua predizioni probabilistiche
        
        Args:
            X: Feature per predizione
            
        Returns:
            Array con probabilità per ogni classe
        """
        if not self.is_trained:
            raise ValueError(f"Modello {self.model_name} non ancora addestrato")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"Modello {self.model_name} non supporta predict_proba")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Restituisce l'importanza delle feature
        
        Returns:
            Series con importanza feature o None se non disponibile
        """
        if not self.is_trained:
            raise ValueError(f"Modello {self.model_name} non ancora addestrato")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            return importance
        elif hasattr(self.model, 'coef_'):
            # Per modelli lineari (es. Logistic Regression)
            importance = pd.Series(
                self.model.coef_[0],
                index=self.feature_names
            ).abs().sort_values(ascending=False)
            return importance
        else:
            logger.warning(f"Modello {self.model_name} non ha feature importance")
            return None
    
    def get_coefficients(self) -> Optional[pd.Series]:
        """
        Restituisce i coefficienti del modello (per modelli lineari)
        
        Returns:
            Series con coefficienti o None se non disponibile
        """
        if not self.is_trained:
            raise ValueError(f"Modello {self.model_name} non ancora addestrato")
        
        if hasattr(self.model, 'coef_'):
            coefficients = pd.Series(
                self.model.coef_[0],
                index=self.feature_names
            ).sort_values(ascending=False)
            return coefficients
        else:
            logger.warning(f"Modello {self.model_name} non ha coefficienti")
            return None
    
    def save_model(self, filepath: str) -> None:
        """
        Salva il modello su disco
        
        Args:
            filepath: Path dove salvare il modello
        """
        if not self.is_trained:
            raise ValueError(f"Modello {self.model_name} non ancora addestrato")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modello {self.model_name} salvato in: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carica il modello da disco
        
        Args:
            filepath: Path del modello da caricare
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.training_time = model_data.get('training_time')
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Modello {self.model_name} caricato da: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sul modello
        
        Returns:
            Dictionary con informazioni sul modello
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'training_time': self.training_time,
            'model_params': self.model.get_params() if self.model else None
        }


if __name__ == "__main__":
    # Questa è una classe astratta, non può essere istanziata direttamente
    # Verrà usata come base per modelli concreti
    print("BaseModel è una classe astratta - usa le classi derivate")
