"""
Random Forest Module
Implementazione del modello Random Forest per credit scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import time
import logging

from base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Modello Random Forest per credit scoring"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Inizializza Random Forest
        
        Args:
            n_estimators: Numero di alberi
            max_depth: Profondità massima alberi
            min_samples_split: Minimo campioni per split
            min_samples_leaf: Minimo campioni per foglia
            class_weight: Bilanciamento classi
            random_state: Seed per riproducibilità
            n_jobs: Numero di core (-1 = tutti)
        """
        super().__init__(model_name="RandomForest")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = self.build_model()
    
    def build_model(self) -> RandomForestClassifier:
        """
        Costruisce il modello Random Forest
        
        Returns:
            RandomForestClassifier configurato
        """
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        logger.info(f"Random Forest creato con {self.n_estimators} estimatori")
        return model
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Addestra il modello Random Forest
        
        Args:
            X_train: Feature di training
            y_train: Target di training
            verbose: Se True, stampa info durante training
            
        Returns:
            Dictionary con metriche di training
        """
        logger.info(f"Inizio training Random Forest con {X_train.shape[0]} campioni")
        
        # Salva nomi feature
        self.feature_names = X_train.columns.tolist()
        
        # Training
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        self.is_trained = True
        
        # Calcola metriche su training set
        train_predictions = self.model.predict(X_train)
        train_accuracy = (train_predictions == y_train).mean()
        
        if verbose:
            logger.info(f"Training completato in {self.training_time:.2f} secondi")
            logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        
        return {
            'training_time': self.training_time,
            'train_accuracy': train_accuracy,
            'n_estimators': self.n_estimators,
            'n_features': len(self.feature_names)
        }
    
    def get_oob_score(self) -> Optional[float]:
        """
        Restituisce l'OOB (Out-of-Bag) score se disponibile
        
        Returns:
            OOB score o None
        """
        if hasattr(self.model, 'oob_score_') and self.model.oob_score:
            return self.model.oob_score_
        else:
            logger.warning("OOB score non disponibile (imposta oob_score=True nel modello)")
            return None
    
    def get_tree_depths(self) -> Dict[str, float]:
        """
        Restituisce statistiche sulla profondità degli alberi
        
        Returns:
            Dictionary con statistiche profondità
        """
        if not self.is_trained:
            raise ValueError("Modello non ancora addestrato")
        
        depths = [estimator.get_depth() for estimator in self.model.estimators_]
        
        return {
            'min_depth': min(depths),
            'max_depth': max(depths),
            'mean_depth': np.mean(depths),
            'median_depth': np.median(depths)
        }
    
    def get_tree_leaves(self) -> Dict[str, float]:
        """
        Restituisce statistiche sul numero di foglie degli alberi
        
        Returns:
            Dictionary con statistiche foglie
        """
        if not self.is_trained:
            raise ValueError("Modello non ancora addestrato")
        
        n_leaves = [estimator.get_n_leaves() for estimator in self.model.estimators_]
        
        return {
            'min_leaves': min(n_leaves),
            'max_leaves': max(n_leaves),
            'mean_leaves': np.mean(n_leaves),
            'median_leaves': np.median(n_leaves)
        }


if __name__ == "__main__":
    # Test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Genera dati di esempio
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crea e addestra modello
    rf_model = RandomForestModel(n_estimators=100)
    train_metrics = rf_model.train(X_train, y_train)
    
    # Predizioni
    predictions = rf_model.predict(X_test)
    proba = rf_model.predict_proba(X_test)
    
    # Feature importance
    importance = rf_model.get_feature_importance()
    
    print("Train metrics:", train_metrics)
    print("\nTop 5 feature importances:")
    print(importance.head())
    
    print("\nTree statistics:")
    print("Depths:", rf_model.get_tree_depths())
    print("Leaves:", rf_model.get_tree_leaves())
