"""
Metrics Module
Calcolo e gestione delle metriche di valutazione dei modelli
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe per valutazione dei modelli"""
    
    def __init__(self):
        """Inizializza l'evaluator"""
        self.metrics_history = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Calcola tutte le metriche di valutazione
        
        Args:
            y_true: Etichette vere
            y_pred: Predizioni (0/1)
            y_pred_proba: Probabilità classe positiva
            model_name: Nome del modello
            
        Returns:
            Dictionary con tutte le metriche
        """
        logger.info(f"Calcolo metriche per {model_name}")
        
        # Metriche base
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Metriche probabilistiche (se disponibili)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Metriche derivate
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Salva in history
        self.metrics_history.append(metrics)
        
        logger.info(f"Metriche calcolate - F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Genera il classification report di sklearn
        
        Args:
            y_true: Etichette vere
            y_pred: Predizioni
            target_names: Nomi delle classi
            
        Returns:
            String con il classification report
        """
        if target_names is None:
            target_names = ['Low Risk', 'High Risk']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        return report
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        approval_cost: float = 100.0,
        default_cost: float = 5000.0,
        rejection_opportunity_cost: float = 200.0
    ) -> Dict[str, float]:
        """
        Calcola metriche di business per credit scoring
        
        Args:
            y_true: Etichette vere (1=buon cliente, 0=cattivo cliente)
            y_pred: Predizioni (1=approva, 0=rifiuta)
            approval_cost: Costo per approvare una richiesta
            default_cost: Costo se cliente fa default
            rejection_opportunity_cost: Costo opportunità di rifiutare buon cliente
            
        Returns:
            Dictionary con metriche di business
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calcola costi
        # TN: Correttamente rifiutati (cattivi clienti) - costo = 0
        # FP: Erroneamente approvati (cattivi clienti) - costo = approval_cost + default_cost
        # FN: Erroneamente rifiutati (buoni clienti) - costo = rejection_opportunity_cost
        # TP: Correttamente approvati (buoni clienti) - guadagno = revenue - approval_cost
        
        total_cost_fp = fp * (approval_cost + default_cost)  # Approvati ma fanno default
        total_cost_fn = fn * rejection_opportunity_cost       # Rifiutati ma erano buoni
        total_cost_tp = tp * approval_cost                    # Costo gestione buoni clienti
        
        total_cost = total_cost_fp + total_cost_fn + total_cost_tp
        
        # Calcolo revenue potenziale (assumendo che ogni buon cliente generi valore)
        revenue_per_good_customer = 1000.0  # Esempio
        total_revenue = tp * revenue_per_good_customer
        
        net_profit = total_revenue - total_cost
        
        return {
            'total_cost': total_cost,
            'cost_false_positives': total_cost_fp,
            'cost_false_negatives': total_cost_fn,
            'cost_true_positives': total_cost_tp,
            'total_revenue': total_revenue,
            'net_profit': net_profit,
            'profit_per_decision': net_profit / len(y_true)
        }
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calcola metriche per diversi threshold di classificazione
        
        Args:
            y_true: Etichette vere
            y_pred_proba: Probabilità classe positiva
            thresholds: Array di threshold da testare
            
        Returns:
            DataFrame con metriche per ogni threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        Trova il threshold ottimale basato su una metrica
        
        Args:
            y_true: Etichette vere
            y_pred_proba: Probabilità classe positiva
            metric: Metrica da ottimizzare ('f1_score', 'accuracy', etc.)
            
        Returns:
            Tuple (optimal_threshold, optimal_metric_value)
        """
        threshold_df = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        optimal_idx = threshold_df[metric].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
        optimal_value = threshold_df.loc[optimal_idx, metric]
        
        logger.info(f"Threshold ottimale per {metric}: {optimal_threshold:.2f} (valore: {optimal_value:.4f})")
        
        return optimal_threshold, optimal_value
    
    def compare_models(self, metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Confronta le metriche di più modelli
        
        Args:
            metrics_list: Lista di dictionary con metriche
            
        Returns:
            DataFrame comparativo
        """
        comparison_data = []
        
        for metrics in metrics_list:
            row = {
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
            }
            
            if 'roc_auc' in metrics:
                row['ROC-AUC'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def export_metrics(self, filepath: str) -> None:
        """
        Esporta le metriche in formato CSV
        
        Args:
            filepath: Path dove salvare il CSV
        """
        if not self.metrics_history:
            logger.warning("Nessuna metrica da esportare")
            return
        
        # Flatten confusion matrix per export
        export_data = []
        for metrics in self.metrics_history:
            flat_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            flat_metrics.update(metrics['confusion_matrix'])
            export_data.append(flat_metrics)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Metriche esportate in: {filepath}")


if __name__ == "__main__":
    # Test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Genera dati
    X, y = make_classification(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Addestra modello
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predizioni
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Valutazione
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba, "RandomForest")
    
    print("Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value}")
    
    print("\nClassification Report:")
    print(evaluator.get_classification_report(y_test, y_pred))
    
    print("\nBusiness Metrics:")
    business_metrics = evaluator.calculate_business_metrics(y_test, y_pred)
    for key, value in business_metrics.items():
        print(f"  {key}: {value:.2f}")
