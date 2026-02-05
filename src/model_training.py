import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

class CreditScoringModel:
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state

    def train_random_forest(self, X, y, n_estimators=100):
        print(f"Training Random Forest con {n_estimators} estimatori...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            class_weight='balanced', 
            random_state=self.random_state, 
            n_jobs=-1
        )
        start = time.time()
        self.model.fit(X, y)
        print(f"Training completato in {time.time() - start:.2f}s")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob)
        }
        print("\nReport Classificazione:")
        print(classification_report(y_test, y_pred))
        return metrics

    def get_feature_importance(self, feature_names):
        importances = pd.Series(
            self.model.feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False)
        return importances
    
    def save_model(self, filepath='models/random_forest_model.pkl'):
        """Salva il modello allenato su disco."""
        if self.model is not None:
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Modello salvato con successo in: {filepath}")
        else:
            print("Errore: Nessun modello da salvare. Esegui prima il training.")