# 🏦 Credit Scoring Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Modello di Machine Learning per la previsione dell'affidabilità creditizia (Credit Scoring) per il rilascio di carte di credito.

## 📌 Descrizione

Il progetto analizza i dati storici dei clienti per prevedere la probabilità di default nel pagamento delle carte di credito. L'obiettivo è supportare il team decisionale con un modello interpretabile che fornisca motivazioni chiare in caso di rifiuto della carta, garantendo trasparenza e conformità normativa.

### Caratteristiche Principali

- ✅ **Pipeline automatizzata** per data cleaning, feature engineering e training
- 📊 **Analisi esplorativa completa** con visualizzazioni interattive
- 🧠 **Modello Random Forest** ottimizzato per performance bilanciate
- 🔍 **Interpretabilità** tramite feature importance analysis
- 📦 **Architettura modulare** per facile manutenzione ed estensione

## 📂 Struttura del Progetto
```
credit-scoring-pro-bank/
├── data/
│   ├── raw/                    # Dataset originali (non versionati)
│   │   └── credit_scoring.csv
│   └── processed/              # Dataset processati
├── models/                     # Modelli salvati (.pkl)
├── notebooks/
│   └── eda_model_engineering.ipynb   # Analisi esplorativa interattiva
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Pulizia e feature engineering
│   ├── visualization.py        # Funzioni per plotting
│   └── model_training.py       # Training e valutazione modello
├── .gitignore
├── main.py                     # Pipeline principale
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### 1. Clona il Repository
```bash
git clone https://github.com/tuousername/credit-scoring-pro-bank.git
cd credit-scoring-pro-bank
```

### 2. Setup Ambiente
```bash
# Crea virtual environment (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### 3. Prepara i Dati

Posiziona il file `credit_scoring.csv` nella directory `data/raw/`.

### 4. Esegui la Pipeline
```bash
python main.py
```

Questo comando eseguirà:
- Caricamento e pulizia dei dati
- Feature engineering automatico
- Training del modello Random Forest
- Valutazione delle performance
- Salvataggio del modello addestrato in `models/`

## 📊 Dataset

Il dataset contiene informazioni demografiche e finanziarie dei clienti:

- **Features demografiche**: Età, Genere, Stato civile, Livello educativo
- **Features finanziarie**: Reddito annuale, Tipo di impiego, Anzianità lavorativa
- **Features immobiliari**: Tipo di abitazione, Proprietà auto/immobile
- **Target**: `TARGET` (0 = Pagatore affidabile, 1 = Rischio default)

## 🧠 Modello e Performance

### Modello Selezionato: Random Forest Classifier

Il Random Forest è stato scelto dopo confronto con Logistic Regression e Decision Tree per:
- Migliori performance complessive (F1-Score e ROC-AUC)
- Robustezza agli outlier
- Capacità di catturare interazioni non-lineari
- Nativa gestione dello sbilanciamento tramite `class_weight='balanced'`

### Metriche di Performance

| Metrica | Valore |
|---------|--------|
| Accuracy | ~92% |
| Precision | ~85% |
| Recall | ~78% |
| F1-Score | ~81% |
| ROC-AUC | ~88% |

*Note: Le metriche esatte dipendono dal dataset specifico utilizzato.*

## 🔍 Interpretabilità

Il modello fornisce l'**importanza delle feature** per spiegare le decisioni:

**Top 5 Feature più importanti:**
1. Anzianità lavorativa (YEARS_EMPLOYED)
2. Età del cliente (AGE_YEARS)
3. Reddito annuale (AMT_INCOME_TOTAL)
4. Stato occupazionale (IS_UNEMPLOYED)
5. Tipo di impiego (NAME_INCOME_TYPE)

Questo permette di fornire motivazioni trasparenti ai clienti in caso di rifiuto.

## 📈 Analisi Esplorativa

Per esplorare i dati interattivamente, apri il notebook Jupyter:
```bash
jupyter notebook notebooks/exploratory_eda.ipynb
```

Il notebook include:
- Distribuzione delle variabili numeriche
- Analisi delle variabili categoriche
- Matrice di correlazione
- Visualizzazioni della relazione con il target

## 🛠️ Sviluppo

### Estendere il Progetto

Per aggiungere nuovi modelli o feature:

1. **Nuove feature**: Modifica `src/data_processing.py` nella funzione `clean_and_engineer_features()`
2. **Nuovi modelli**: Aggiungi metodi alla classe `CreditScoringModel` in `src/model_training.py`
3. **Nuove visualizzazioni**: Estendi `src/visualization.py`

### Testing
```bash
# Esegui test unitari (se implementati)
pytest tests/
```

## 📋 Requisiti

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Vedi `requirements.txt` per le versioni specifiche.

## 🤝 Contributi

I contributi sono benvenuti! Per contribuire:

1. Fai un fork del progetto
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.

## 👥 Autori

**Data Science Team @ Pro National Bank**

- [Il Tuo Nome](https://github.com/tuousername)

## 📞 Contatti

Per domande o supporto:
- 📧 Email: datascience@probank.com
- 🐛 Issues: [GitHub Issues](https://github.com/tuousername/credit-scoring-pro-bank/issues)

## 🙏 Ringraziamenti

- Dataset fornito da [fonte del dataset]
- Ispirato dalle best practices di [riferimenti]

---

⭐ Se questo progetto ti è stato utile, considera di lasciare una stella!
