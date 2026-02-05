# 🚀 Quick Start Guide

Guida rapida per iniziare a utilizzare il Credit Scoring Model in 5 minuti.

## ⚡ Setup Veloce

### 1. Clone & Install (2 minuti)

```bash
# Clone repository
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### 2. Download Dataset (1 minuto)

```bash
# Crea directory
mkdir -p data/raw

# Download dataset
wget https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv -P data/raw/
```

### 3. Train Model (2 minuti)

```bash
# Training con configurazione default
python scripts/train_model.py
```

Output atteso:
```
==================================================
STEP 1: Caricamento e preparazione dati
==================================================
Dataset caricato: (30000, 19)
...
==================================================
STEP 2: Training Random Forest
==================================================
F1-Score: 0.9234
ROC-AUC: 0.9456
...
TRAINING COMPLETATO CON SUCCESSO!
```

## 📊 Uso Base

### Predizione su Nuovi Dati

```python
from src.models.random_forest import RandomForestModel
import pandas as pd

# Carica modello salvato
model = RandomForestModel()
model.load_model("models/random_forest_model.pkl")

# Nuovi clienti
new_customers = pd.read_csv("new_customers.csv")

# Predizioni
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)

# Risultati
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    if pred == 1:
        print(f"Cliente {i}: APPROVATO (confidenza: {prob[1]:.2%})")
    else:
        print(f"Cliente {i}: RIFIUTATO (confidenza: {prob[0]:.2%})")
```

### Pipeline Completa da Python

```python
from scripts.train_model import CreditScoringTrainer

# Inizializza trainer
trainer = CreditScoringTrainer()

# Esegui pipeline completa
trainer.run_full_pipeline()

# Accedi ai risultati
best_model = trainer.models['random_forest']
metrics = trainer.results['random_forest']['metrics']

print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

## 🎯 Esempi Comuni

### Esempio 1: Valutare Singolo Cliente

```python
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.random_forest import RandomForestModel
import pandas as pd

# Dati cliente
client_data = {
    'CODE_GENDER': 'M',
    'FLAG_OWN_CAR': 'Y',
    'FLAG_OWN_REALTY': 'Y',
    'CNT_CHILDREN': 2,
    'AMT_INCOME_TOTAL': 65000,
    'NAME_INCOME_TYPE': 'Working',
    'NAME_EDUCATION_TYPE': 'Higher education',
    'NAME_FAMILY_STATUS': 'Married',
    'NAME_HOUSING_TYPE': 'House / apartment',
    'DAYS_BIRTH': -12000,
    'DAYS_EMPLOYED': -3000,
    'FLAG_MOBIL': 1,
    'FLAG_WORK_PHONE': 1,
    'FLAG_PHONE': 0,
    'FLAG_EMAIL': 1,
    'OCCUPATION_TYPE': 'Managers',
    'CNT_FAM_MEMBERS': 4
}

df_client = pd.DataFrame([client_data])

# Preprocessing
engineer = FeatureEngineer()
df_client = engineer.apply_all_transformations(df_client)

preprocessor = DataPreprocessor()
X_client, _ = preprocessor.preprocess_pipeline(df_client, fit=False)

# Predizione
model = RandomForestModel()
model.load_model("models/random_forest_model.pkl")

prediction = model.predict(X_client)[0]
probability = model.predict_proba(X_client)[0]

# Risultato
if prediction == 1:
    print(f"✅ Cliente APPROVATO")
    print(f"   Probabilità alta affidabilità: {probability[1]:.1%}")
else:
    print(f"❌ Cliente RIFIUTATO")
    print(f"   Probabilità bassa affidabilità: {probability[0]:.1%}")

# Top feature che hanno influenzato
importance = model.get_feature_importance()
print("\nTop 5 fattori decisivi:")
for i, (feature, imp) in enumerate(importance.head(5).items(), 1):
    print(f"{i}. {feature}: {imp:.3f}")
```

### Esempio 2: Batch Processing

```python
import pandas as pd
from src.models.random_forest import RandomForestModel

# Carica batch di clienti
clients_df = pd.read_csv("data/raw/new_applications.csv")

# Carica modello
model = RandomForestModel()
model.load_model("models/random_forest_model.pkl")

# Preprocessing (assumendo dati già preprocessati)
predictions = model.predict(clients_df)
probabilities = model.predict_proba(clients_df)[:, 1]

# Aggiungi risultati
clients_df['DECISION'] = ['APPROVED' if p == 1 else 'REJECTED' for p in predictions]
clients_df['CONFIDENCE'] = probabilities

# Salva risultati
clients_df.to_csv("results/credit_decisions.csv", index=False)

# Statistiche
print(f"Totale richieste: {len(clients_df)}")
print(f"Approvati: {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
print(f"Rifiutati: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
```

### Esempio 3: Analisi Feature Importance

```python
from src.models.random_forest import RandomForestModel
import matplotlib.pyplot as plt
import seaborn as sns

# Carica modello
model = RandomForestModel()
model.load_model("models/random_forest_model.pkl")

# Feature importance
importance = model.get_feature_importance()

# Plot top 15
plt.figure(figsize=(10, 6))
importance.head(15).plot(kind='barh')
plt.title('Top 15 Most Important Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('reports/figures/feature_importance.png')
plt.show()
```

### Esempio 4: Threshold Tuning

```python
from src.evaluation.metrics import ModelEvaluator
from src.models.random_forest import RandomForestModel
import numpy as np

# Carica modello e test data
model = RandomForestModel()
model.load_model("models/random_forest_model.pkl")

# ... (load X_test, y_test)

# Predizioni probabilistiche
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Trova threshold ottimale
evaluator = ModelEvaluator()
optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
    y_test, 
    y_pred_proba,
    metric='f1_score'
)

print(f"Threshold ottimale: {optimal_threshold:.2f}")
print(f"F1-Score con threshold ottimale: {optimal_f1:.4f}")

# Usa threshold custom
y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)
metrics = evaluator.calculate_metrics(y_test, y_pred_custom, y_pred_proba)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

## 🔧 Customizzazione

### Modificare Configurazione

Edita `config/config.yaml`:

```yaml
# Cambia numero alberi Random Forest
models:
  random_forest:
    n_estimators: 300  # Default: 200

# Cambia test size
data:
  test_size: 0.25  # Default: 0.2

# Disabilita feature engineering
feature_engineering:
  enabled: false
```

Poi esegui:
```bash
python scripts/train_model.py --config config/config.yaml
```

### Aggiungere Nuovo Modello

```python
# src/models/xgboost_model.py
from base_model import BaseModel
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    def build_model(self):
        return XGBClassifier(...)
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return {...}
```

## 📈 Monitoraggio Risultati

### Visualizza Report

```bash
# Report testuale
cat reports/results/training_report.txt

# Metriche CSV
cat reports/results/model_comparison.csv

# Feature importance
cat reports/results/random_forest_feature_importance.csv
```

### Logs

```bash
# Training logs
tail -f logs/training.log
```

## 🐛 Troubleshooting

### Problema: ModuleNotFoundError

```bash
# Soluzione: Installa in modalità editable
pip install -e .
```

### Problema: Dataset non trovato

```bash
# Soluzione: Verifica path
ls data/raw/credit_scoring.csv

# Se mancante, ri-download
wget https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv -P data/raw/
```

### Problema: Out of Memory

```yaml
# Soluzione: Riduci n_estimators in config.yaml
models:
  random_forest:
    n_estimators: 50
```

## 🎓 Prossimi Passi

1. ✅ **Completa Quick Start** ← Sei qui
2. 📚 **Leggi [README.md](README.md)** per dettagli completi
3. 🏗️ **Consulta [ARCHITECTURE.md](ARCHITECTURE.md)** per architettura
4. 🤝 **Leggi [CONTRIBUTING.md](CONTRIBUTING.md)** per contribuire
5. 📓 **Esplora notebooks/** per analisi dettagliate

## 💡 Tips

- Usa `jupyter notebook` per esplorare i notebooks in `notebooks/`
- I modelli salvati in `models/` sono riutilizzabili
- Modifica `config/config.yaml` invece di hardcodare valori
- Esegui test con `pytest tests/`

## 📞 Help

- 📖 **Documentazione**: Vedi README.md
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/credit-scoring-model/issues)
- 💬 **Discussioni**: [GitHub Discussions](https://github.com/your-username/credit-scoring-model/discussions)

---

**Buon Coding! 🚀**
