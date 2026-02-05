"""
Training Script
Script principale per addestrare i modelli di credit scoring
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import time
import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.random_forest import RandomForestModel
from evaluation.metrics import ModelEvaluator

from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditScoringTrainer:
    """Classe principale per training pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il trainer
        
        Args:
            config_path: Path al file di configurazione YAML
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
    
    def _load_config(self, config_path: str) -> dict:
        """Carica configurazione da file YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configurazione caricata da: {config_path}")
        return config
    
    def _default_config(self) -> dict:
        """Configurazione di default"""
        return {
            'data': {
                'raw_path': 'data/raw/credit_scoring.csv',
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'missing_numeric': 'median',
                'missing_categorical': 'mode',
                'scaling': True
            },
            'feature_engineering': {
                'enabled': True
            },
            'models': {
                'random_forest': {
                    'enabled': True,
                    'n_estimators': 200,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            },
            'output': {
                'models_dir': 'models',
                'results_dir': 'reports/results'
            }
        }
    
    def load_and_prepare_data(self) -> None:
        """Carica e prepara i dati"""
        logger.info("="*50)
        logger.info("STEP 1: Caricamento e preparazione dati")
        logger.info("="*50)
        
        # 1. Caricamento
        data_path = self.config['data']['raw_path']
        df = self.data_loader.load_data(data_path)
        
        # Info sui dati
        logger.info(f"Dataset caricato: {df.shape}")
        target_dist = self.data_loader.check_target_distribution()
        logger.info(f"Distribuzione TARGET: {target_dist['counts']}")
        
        # 2. Feature Engineering
        if self.config['feature_engineering']['enabled']:
            logger.info("Applicazione feature engineering...")
            df = self.feature_engineer.apply_all_transformations(df)
        
        # 3. Preprocessing
        logger.info("Preprocessing dati...")
        X, y = self.preprocessor.preprocess_pipeline(df, fit=True)
        
        # 4. Train-Test Split
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info("Preparazione dati completata\n")
    
    def train_random_forest(self) -> None:
        """Addestra il modello Random Forest"""
        logger.info("="*50)
        logger.info("STEP 2: Training Random Forest")
        logger.info("="*50)
        
        rf_config = self.config['models']['random_forest']
        
        if not rf_config.get('enabled', True):
            logger.info("Random Forest disabilitato nella configurazione")
            return
        
        # Crea modello
        rf_model = RandomForestModel(
            n_estimators=rf_config.get('n_estimators', 200),
            class_weight=rf_config.get('class_weight', 'balanced'),
            n_jobs=rf_config.get('n_jobs', -1),
            random_state=self.config['data']['random_state']
        )
        
        # Training
        train_metrics = rf_model.train(self.X_train, self.y_train)
        
        # Predizioni su test set
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Valutazione
        metrics = self.evaluator.calculate_metrics(
            self.y_test,
            y_pred,
            y_pred_proba,
            model_name="Random Forest"
        )
        
        # Feature importance
        feature_importance = rf_model.get_feature_importance()
        
        # Salva risultati
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'train_metrics': train_metrics
        }
        
        # Log risultati
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        logger.info("\nTop 10 Feature Importance:")
        for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        logger.info("Training Random Forest completato\n")
    
    def save_models(self) -> None:
        """Salva i modelli addestrati"""
        logger.info("="*50)
        logger.info("STEP 3: Salvataggio modelli")
        logger.info("="*50)
        
        models_dir = Path(self.config['output']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = models_dir / f"{model_name}_model.pkl"
            model.save_model(str(filepath))
            logger.info(f"Modello '{model_name}' salvato in: {filepath}")
        
        logger.info("Salvataggio modelli completato\n")
    
    def save_results(self) -> None:
        """Salva i risultati della valutazione"""
        logger.info("="*50)
        logger.info("STEP 4: Salvataggio risultati")
        logger.info("="*50)
        
        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metriche comparative
        metrics_list = [result['metrics'] for result in self.results.values()]
        comparison_df = self.evaluator.compare_models(metrics_list)
        
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Confronto modelli salvato in: {comparison_path}")
        
        # 2. Feature importance per ogni modello
        for model_name, result in self.results.items():
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance_df = result['feature_importance'].reset_index()
                importance_df.columns = ['Feature', 'Importance']
                
                importance_path = results_dir / f'{model_name}_feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Feature importance '{model_name}' salvata in: {importance_path}")
        
        # 3. Report completo
        self._generate_report(results_dir)
        
        logger.info("Salvataggio risultati completato\n")
    
    def _generate_report(self, results_dir: Path) -> None:
        """Genera report testuale completo"""
        report_path = results_dir / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CREDIT SCORING MODEL - TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Number of features: {self.X_train.shape[1]}\n")
            f.write(f"Target distribution (train): {self.y_train.value_counts().to_dict()}\n\n")
            
            # Models results
            for model_name, result in self.results.items():
                f.write(f"\n{model_name.upper()} RESULTS\n")
                f.write("-"*70 + "\n")
                
                metrics = result['metrics']
                f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
                
                f.write("\nConfusion Matrix:\n")
                cm = metrics['confusion_matrix']
                f.write(f"  TN: {cm['true_negatives']:<6} FP: {cm['false_positives']}\n")
                f.write(f"  FN: {cm['false_negatives']:<6} TP: {cm['true_positives']}\n")
                
                if 'feature_importance' in result and result['feature_importance'] is not None:
                    f.write("\nTop 10 Feature Importance:\n")
                    for i, (feature, importance) in enumerate(result['feature_importance'].head(10).items(), 1):
                        f.write(f"  {i:2d}. {feature:<30} {importance:.4f}\n")
                
                f.write("\n")
        
        logger.info(f"Report completo salvato in: {report_path}")
    
    def run_full_pipeline(self) -> None:
        """Esegue la pipeline completa di training"""
        start_time = time.time()
        
        logger.info("\n" + "="*70)
        logger.info("INIZIO TRAINING PIPELINE CREDIT SCORING")
        logger.info("="*70 + "\n")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Train models
            self.train_random_forest()
            # Qui si possono aggiungere altri modelli
            
            # Step 3: Save models
            self.save_models()
            
            # Step 4: Save results
            self.save_results()
            
            # Summary
            elapsed_time = time.time() - start_time
            logger.info("="*70)
            logger.info("TRAINING COMPLETATO CON SUCCESSO!")
            logger.info(f"Tempo totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Errore durante training: {str(e)}", exc_info=True)
            raise


def main():
    """Funzione main"""
    parser = argparse.ArgumentParser(description='Training Credit Scoring Models')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path al file di configurazione YAML'
    )
    
    args = parser.parse_args()
    
    # Crea trainer ed esegui pipeline
    trainer = CreditScoringTrainer(config_path=args.config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
