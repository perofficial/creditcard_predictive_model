# Data Directory

Questa directory contiene i dati utilizzati per il progetto Credit Scoring.

## 📁 Struttura

```
data/
├── raw/              # Dati grezzi originali (NON modificare)
│   └── credit_scoring.csv
├── processed/        # Dati elaborati e preprocessati
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
└── README.md         # Questo file
```

## 📥 Download Dataset

Il dataset può essere scaricato da:
```
https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv
```

Oppure utilizzando wget:
```bash
wget https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv -P data/raw/
```

## 📊 Descrizione Dataset

### File: credit_scoring.csv

Dataset con informazioni anonimizzate di clienti che hanno richiesto carta di credito.

**Dimensioni**: ~30,000 righe × 19 colonne

**Colonne**:
- `ID`: Identificativo univoco cliente
- `CODE_GENDER`: Sesso (M/F)
- `FLAG_OWN_CAR`: Possesso automobile (Y/N)
- `FLAG_OWN_REALTY`: Possesso casa (Y/N)
- `CNT_CHILDREN`: Numero figli
- `AMT_INCOME_TOTAL`: Reddito annuale totale
- `NAME_INCOME_TYPE`: Tipo reddito (Working, Commercial, etc.)
- `NAME_EDUCATION_TYPE`: Livello educazione
- `NAME_FAMILY_STATUS`: Stato civile
- `NAME_HOUSING_TYPE`: Tipo abitazione
- `DAYS_BIRTH`: Giorni dalla nascita (negativo)
- `DAYS_EMPLOYED`: Giorni dall'assunzione (negativo=impiegato, positivo=disoccupato)
- `FLAG_MOBIL`: Presenza cellulare (0/1)
- `FLAG_WORK_PHONE`: Presenza telefono lavoro (0/1)
- `FLAG_PHONE`: Presenza telefono (0/1)
- `FLAG_EMAIL`: Presenza email (0/1)
- `OCCUPATION_TYPE`: Tipo occupazione
- `CNT_FAM_MEMBERS`: Numero membri famiglia
- `TARGET`: **Variabile target** (1=alta affidabilità, 0=bassa affidabilità)

## 🔒 Privacy e Sicurezza

⚠️ **IMPORTANTE**:
- I dati sono **completamente anonimizzati**
- NON versionare file dati su Git (vedi `.gitignore`)
- Rispettare GDPR e normative privacy
- NON condividere dati al di fuori del progetto

## 📝 Note

- I file nella cartella `raw/` non devono mai essere modificati
- I file `processed/` sono generati automaticamente dalla pipeline
- Per ricaricare i dati, eliminare `processed/` e rieseguire preprocessing
