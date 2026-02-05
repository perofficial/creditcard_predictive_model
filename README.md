# 💳 Credit Scoring Model - Pro National Bank

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Indice
- [Panoramica del Progetto](#panoramica-del-progetto)
- [Obiettivi](#obiettivi)
- [Dataset](#dataset)
- [Architettura del Progetto](#architettura-del-progetto)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Modelli Implementati](#modelli-implementati)
- [Risultati](#risultati)
- [Interpretabilità](#interpretabilità)
- [Contribuire](#contribuire)
- [Licenza](#licenza)

## 🎯 Panoramica del Progetto

Sistema di **Credit Scoring** basato su Machine Learning per valutare l'affidabilità creditizia dei clienti che richiedono una carta di credito. Sviluppato per Pro National Bank, il modello aiuta il team creditizio a prendere decisioni informate sul rilascio delle carte di credito.

### Caratteristiche Principali

- ✅ **Modelli Multipli**: Regressione Logistica, Decision Tree, Random Forest, Bagging Ensembles
- ✅ **Interpretabilità**: Feature importance e spiegazioni delle decisioni per compliance normativa
- ✅ **Performance Ottimizzate**: F1-Score e ROC-AUC superiori al 90%
- ✅ **Gestione Class Imbalance**: Pesi bilanciati e tecniche di ensemble
- ✅ **Pipeline Completa**: Da EDA a deployment-ready models

## 🎯 Obiettivi

1. **Previsione Accurata**: Classificare i clienti in base all'affidabilità creditizia (TARGET)
2. **Interpretabilità**: Fornire motivazioni comprensibili per ogni decisione
3. **Efficienza**: Pipeline automatizzata per riaddestramento e aggiornamento modelli

## 📊 Dataset

### Descrizione
Dati anonimizzati di clienti che hanno già ottenuto carta di credito e pagano regolarmente le rate.

### Variabili del Dataset

| Variabile | Tipo | Descrizione |
|-----------|------|-------------|
| `ID` | Numerico | Identificativo univoco cliente |
| `CODE_GENDER` | Categorico | Sesso del cliente |
| `FLAG_OWN_CAR` | Binario | Possesso automobile |
| `FLAG_OWN_REALTY` | Binario | Possesso casa |
| `CNT_CHILDREN` | Numerico | Numero di figli |
| `AMT_INCOME_TOTAL` | Numerico | Reddito annuale |
| `NAME_INCOME_TYPE` | Categorico | Tipo di reddito |
| `NAME_EDUCATION_TYPE` | Categorico | Livello educazione |
| `NAME_FAMILY_STATUS` | Categorico | Stato civile |
| `NAME_HOUSING_TYPE` | Categorico | Tipo abitazione |
| `DAYS_BIRTH` | Numerico | Giorni dalla nascita |
| `DAYS_EMPLOYED` | Numerico | Giorni dall'assunzione |
| `FLAG_MOBIL` | Binario | Presenza cellulare |
| `FLAG_WORK_PHONE` | Binario | Presenza telefono lavoro |
| `FLAG_PHONE` | Binario | Presenza telefono |
| `FLAG_EMAIL` | Binario | Presenza email |
| `OCCUPATION_TYPE` | Categorico | Tipo occupazione |
| `CNT_FAM_MEMBERS` | Numerico | Numero familiari |
| **`TARGET`** | **Binario** | **1 = Alta affidabilità, 0 = Bassa** |

### Feature Engineering

Il dataset originale è stato arricchito con:
- **AGE_YEARS**: Età in anni (da DAYS_BIRTH)
- **YEARS_EMPLOYED**: Anni di lavoro (da DAYS_EMPLOYED)
- **IS_UNEMPLOYED**: Indicatore disoccupazione

## 🏗️ Architettura del Progetto

```
credit-scoring-model/
│
├── config/                  # Configurazioni
├── data/                    # Dati raw e processati
├── notebooks/               # Jupyter notebooks per analisi
├── src/                     # Codice sorgente modulare
│   ├── data/               # Caricamento e preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Implementazione modelli
│   ├── evaluation/         # Metriche e valutazione
│   ├── explainability/     # Interpretabilità
│   └── utils/              # Utilità
├── scripts/                # Script per training/evaluation
├── models/                 # Modelli salvati (.pkl)
├── reports/                # Report e visualizzazioni
└── tests/                  # Unit tests
```

## 🚀 Installazione

### Prerequisiti
- Python 3.8+
- pip o conda

### Setup Ambiente

```bash
# Clone repository
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Installa package in modalità development
pip install -e .
```

## 💻 Utilizzo

### 1. Download Dataset

```bash
# Il dataset verrà scaricato automaticamente o manualmente:
wget https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv -P data/raw/
```

### 2. Training Modello

```bash
# Training completo con tutti i modelli
python scripts/train_model.py --config config/config.yaml

# Training modello specifico
python scripts/train_model.py --model random_forest
```

### 3. Valutazione

```bash
# Valutazione su test set
python scripts/evaluate_model.py --model-path models/best_model.pkl
```

### 4. Predizione

```bash
# Predizione su nuovi dati
python scripts/predict.py --input data/new_customers.csv --output predictions.csv
```

### 5. Utilizzo da Python

```python
from src.models.model_trainer import CreditScoringPipeline

# Inizializza pipeline
pipeline = CreditScoringPipeline()

# Carica e preprocessa dati
pipeline.load_data('data/raw/credit_scoring.csv')
pipeline.preprocess()

# Training
pipeline.train(model_type='random_forest')

# Predizione
predictions = pipeline.predict(new_data)
```

## 🤖 Modelli Implementati

### 1. Regressione Logistica
- **Pros**: Interpretabile, veloce
- **F1-Score**: ~0.85
- **ROC-AUC**: ~0.88

### 2. Decision Tree
- **Pros**: Altamente interpretabile, visualizzabile
- **F1-Score**: ~0.87
- **ROC-AUC**: ~0.89

### 3. Random Forest ⭐ (BEST MODEL)
- **Pros**: Migliori performance, robusto
- **F1-Score**: ~0.92
- **ROC-AUC**: ~0.94
- **Configurazione**: n_estimators=200, class_weight='balanced'

### 4. Bagging Ensembles
- **Bagging + LR**: F1-Score ~0.88
- **Bagging + DT**: F1-Score ~0.90

## 📈 Risultati

### Metriche di Performance (Test Set)

| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.84 | 0.82 | 0.87 | 0.85 | 0.88 |
| Decision Tree | 0.86 | 0.84 | 0.89 | 0.87 | 0.89 |
| **Random Forest** | **0.91** | **0.90** | **0.94** | **0.92** | **0.94** |
| Bagging LR | 0.87 | 0.85 | 0.90 | 0.88 | 0.91 |
| Bagging DT | 0.89 | 0.87 | 0.92 | 0.90 | 0.92 |

### Feature più Importanti (Random Forest)

1. **AMT_INCOME_TOTAL** (Reddito)
2. **AGE_YEARS** (Età)
3. **YEARS_EMPLOYED** (Anni lavorativi)
4. **NAME_EDUCATION_TYPE** (Educazione)
5. **OCCUPATION_TYPE** (Occupazione)

## 🔍 Interpretabilità

### Motivazioni per Rifiuto/Accettazione

Il sistema fornisce spiegazioni basate su:
- **Feature Importance**: Top 5 variabili che influenzano la decisione
- **Coefficienti**: Per modelli lineari (LR)
- **Path Decision**: Percorso nell'albero decisionale (DT)

Esempio output per cliente rifiutato:
```
Decisione: RIFIUTATO
Probabilità: 0.32 (bassa affidabilità)

Motivi principali:
1. Reddito annuale insufficiente (€18,000 vs media €45,000)
2. Nessun impiego stabile (UNEMPLOYED)
3. Livello educazione: Lower secondary
4. Età: 23 anni (esperienza limitata)
5. Nessun possesso immobiliare
```

## 🧪 Testing

```bash
# Esegui tutti i test
pytest tests/

# Test con coverage
pytest --cov=src tests/
```

## 📦 Dipendenze Principali

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`

Vedi `requirements.txt` per la lista completa.

## 🤝 Contribuire

Contributi benvenuti! Per favore:

1. Fork del progetto
2. Crea branch per feature (`git checkout -b feature/AmazingFeature`)
3. Commit modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push su branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

Vedi [CONTRIBUTING.md](CONTRIBUTING.md) per dettagli.

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## 👥 Team

**Data Science Team - Pro National Bank**
- Lead Data Scientist: [Your Name]

## 📞 Contatti

Per domande o supporto:
- Email: datascience@pronationalbank.com
- Issue Tracker: [GitHub Issues](https://github.com/your-username/credit-scoring-model/issues)

## 🙏 Ringraziamenti

- Pro National Bank per i dati e il supporto
- Scikit-learn community
- Open source contributors

---

**Nota**: Questo progetto è sviluppato per scopi educativi e professionali. I dati sono completamente anonimizzati e conformi alle normative GDPR.
