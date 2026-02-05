# Architecture Documentation

## 🏗️ Architettura del Sistema Credit Scoring

Questo documento descrive l'architettura tecnica del sistema di Credit Scoring.

## 📊 Overview

Il sistema è organizzato in una pipeline modulare che segue i principi di:
- **Separation of Concerns**: Ogni modulo ha una responsabilità ben definita
- **Reusability**: Componenti riutilizzabili e configurabili
- **Testability**: Codice facilmente testabile con unit tests
- **Scalability**: Architettura estendibile per nuovi modelli e feature

## 🔄 Pipeline Flow

```
┌─────────────────┐
│  Raw Data CSV   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │ ─────► Caricamento e validazione iniziale
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │ ─────► Creazione nuove feature derivate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessor    │ ─────► Preprocessing, encoding, scaling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train/Test Split│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ ─────► Training multipli modelli
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │ ─────► Metriche e valutazione
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Artifacts │ ─────► Salvataggio modelli e risultati
└─────────────────┘
```

## 📦 Componenti Principali

### 1. Data Layer (`src/data/`)

**Responsabilità**: Gestione dati raw e preprocessati

#### DataLoader (`data_loader.py`)
- Caricamento dataset da CSV
- Validazione schema dati
- Download automatico dataset
- Statistiche e info sui dati

```python
loader = DataLoader()
df = loader.load_data("data/raw/credit_scoring.csv")
info = loader.get_data_info()
```

#### DataPreprocessor (`data_preprocessor.py`)
- Gestione valori mancanti (imputazione)
- Rimozione duplicati
- Encoding variabili categoriche (One-Hot, Label)
- Scaling feature numeriche (StandardScaler)
- Pipeline completa preprocessing

```python
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess_pipeline(df)
```

**Design Pattern**: Pipeline Pattern

---

### 2. Feature Layer (`src/features/`)

**Responsabilità**: Feature engineering e selezione

#### FeatureEngineer (`feature_engineering.py`)
- Trasformazione DAYS_BIRTH → AGE_YEARS, AGE_GROUP
- Trasformazione DAYS_EMPLOYED → YEARS_EMPLOYED, IS_UNEMPLOYED
- Feature derivate da reddito (LOG_INCOME, INCOME_BRACKET)
- Feature famiglia (HAS_CHILDREN, FAMILY_SIZE)
- Feature beni (TOTAL_ASSETS)
- Feature contatti (CONTACT_METHODS)
- Feature di interazione (AGE*INCOME, etc.)

```python
engineer = FeatureEngineer()
df_engineered = engineer.apply_all_transformations(df)
```

**Design Pattern**: Builder Pattern

---

### 3. Model Layer (`src/models/`)

**Responsabilità**: Implementazione e training modelli ML

#### BaseModel (`base_model.py`)
Classe astratta base per tutti i modelli:
- Interface comune: `train()`, `predict()`, `predict_proba()`
- Gestione feature importance
- Serializzazione modelli (save/load)
- Metadata tracking

```python
class BaseModel(ABC):
    @abstractmethod
    def build_model(self): pass
    
    @abstractmethod
    def train(self, X, y): pass
```

#### RandomForestModel (`random_forest.py`)
Implementazione concreta Random Forest:
- Configurazione iperparametri
- Class balancing
- Feature importance nativa
- Statistiche alberi (depth, leaves)

```python
rf = RandomForestModel(n_estimators=200)
rf.train(X_train, y_train)
predictions = rf.predict(X_test)
```

**Altri Modelli** (pattern estendibile):
- `LogisticRegression` → Per interpretabilità lineare
- `DecisionTree` → Per visualizzazione regole
- `BaggingEnsemble` → Per riduzione variance

**Design Pattern**: Strategy Pattern, Template Method

---

### 4. Evaluation Layer (`src/evaluation/`)

**Responsabilità**: Valutazione performance modelli

#### ModelEvaluator (`metrics.py`)
- Calcolo metriche standard (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix e metriche derivate
- Business metrics (costi/ricavi)
- Threshold optimization
- Model comparison

```python
evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
comparison = evaluator.compare_models([metrics1, metrics2])
```

#### Visualization (`visualization.py`)
- Confusion matrix heatmaps
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Threshold analysis plots

**Design Pattern**: Observer Pattern

---

### 5. Explainability Layer (`src/explainability/`)

**Responsabilità**: Interpretabilità modelli

#### FeatureImportance
- Importanza globale feature
- SHAP values (opzionale)
- Coefficienti modelli lineari

#### ModelInterpreter
- Spiegazioni per singole predizioni
- "Why rejected/approved?" per clienti
- Top-N feature influenti per decisione

**Design Pattern**: Decorator Pattern

---

### 6. Utils Layer (`src/utils/`)

**Responsabilità**: Funzionalità di supporto

- **Logger**: Logging configurabile
- **Helpers**: Funzioni utility comuni
- **Config**: Gestione configurazioni

---

## 🎯 Training Pipeline (`scripts/train_model.py`)

Orchestrazione completa del training:

```python
class CreditScoringTrainer:
    def run_full_pipeline(self):
        1. Load and prepare data
        2. Train models
        3. Save models
        4. Save results
```

**Configurazione**: Via YAML (`config/config.yaml`)

---

## 💾 Data Flow

### Input
```
data/raw/credit_scoring.csv (30k rows × 19 cols)
```

### Processing
```
Feature Engineering → 25+ features
One-Hot Encoding → ~60 features
Scaling → Standardized
```

### Output
```
models/random_forest_model.pkl
reports/results/model_comparison.csv
reports/results/feature_importance.csv
reports/figures/confusion_matrix.png
```

---

## 🔐 Design Principles

### SOLID Principles

**Single Responsibility**
- Ogni classe ha una responsabilità unica
- `DataLoader`: solo caricamento
- `Preprocessor`: solo preprocessing

**Open/Closed**
- Estendibile senza modificare codice esistente
- Nuovi modelli: eredita da `BaseModel`

**Liskov Substitution**
- Tutti i modelli sono intercambiabili
- Interface comune via `BaseModel`

**Interface Segregation**
- Interface minimali e specifiche
- `predict()` separato da `predict_proba()`

**Dependency Inversion**
- Dipendenza da astrazioni (BaseModel)
- Non da implementazioni concrete

### DRY (Don't Repeat Yourself)
- Logica comune in BaseModel
- Utilities condivise in utils/
- Configurazione centralizzata

### KISS (Keep It Simple)
- Moduli piccoli e focalizzati
- Funzioni chiare e leggibili
- Evita over-engineering

---

## 📈 Scalability

### Nuovi Modelli
```python
class XGBoostModel(BaseModel):
    def build_model(self):
        return XGBClassifier(...)
    
    def train(self, X, y):
        self.model.fit(X, y)
```

### Nuove Feature
```python
# In FeatureEngineer
def create_risk_score(self, df):
    df['RISK_SCORE'] = ...
    return df
```

### Nuove Metriche
```python
# In ModelEvaluator
def calculate_profit_metric(self, y_true, y_pred):
    return profit_calculation(...)
```

---

## 🧪 Testing Strategy

```
tests/
├── test_data_loader.py       # Unit tests data layer
├── test_preprocessor.py       # Unit tests preprocessing
├── test_feature_engineering.py
├── test_models.py             # Unit tests modelli
└── test_integration.py        # Integration tests
```

**Coverage Target**: 80%+

---

## 🚀 Deployment Considerations

### Model Serving
- Serializzazione con joblib/pickle
- API REST per predizioni (futuro)
- Batch processing scripts

### Monitoring
- Model drift detection
- Performance degradation alerts
- Feature distribution monitoring

### Retraining
- Scheduled retraining pipeline
- Incremental learning (futuro)
- A/B testing framework

---

## 📚 Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Testing** | pytest |
| **Config** | YAML |
| **Serialization** | joblib |
| **Version Control** | Git |

---

## 🔄 CI/CD Pipeline (Future)

```yaml
# .github/workflows/ci.yml
- Lint (Black, Flake8)
- Unit Tests
- Integration Tests
- Coverage Report
- Model Training (on merge to main)
- Model Validation
- Artifact Upload
```

---

## 📖 References

- [Scikit-learn Best Practices](https://scikit-learn.org/stable/)
- [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Design Patterns](https://refactoring.guru/design-patterns)
- [ML System Design](https://www.amazon.com/Machine-Learning-Systems-Design/dp/1098107966)
