# Contributing to Credit Scoring Model

Grazie per il tuo interesse nel contribuire al progetto Credit Scoring Model! Questa guida ti aiuterà a iniziare.

## 📋 Indice
- [Code of Conduct](#code-of-conduct)
- [Come Contribuire](#come-contribuire)
- [Setup Ambiente di Sviluppo](#setup-ambiente-di-sviluppo)
- [Processo di Sviluppo](#processo-di-sviluppo)
- [Linee Guida per il Codice](#linee-guida-per-il-codice)
- [Testing](#testing)
- [Documentazione](#documentazione)

## 📜 Code of Conduct

Ci aspettiamo che tutti i contributori:
- Siano rispettosi e professionali
- Accettino critiche costruttive
- Si concentrino su ciò che è meglio per il progetto
- Mostrino empatia verso altri membri della community

## 🤝 Come Contribuire

### Segnalazione Bug

Se trovi un bug, apri una issue includendo:
- Descrizione chiara del problema
- Passi per riprodurre il bug
- Comportamento atteso vs comportamento attuale
- Screenshot (se applicabile)
- Informazioni sull'ambiente (OS, Python version, etc.)

### Suggerimento Feature

Per suggerire nuove feature:
- Apri una issue con tag "enhancement"
- Descrivi la feature e il problema che risolve
- Fornisci esempi di utilizzo
- Discuti alternative considerate

### Pull Requests

1. **Fork del repository**
2. **Crea un branch** per la tua feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Sviluppa la tua feature** seguendo le linee guida

4. **Aggiungi test** per il nuovo codice

5. **Esegui tutti i test**:
   ```bash
   pytest tests/
   ```

6. **Commit delle modifiche**:
   ```bash
   git commit -m "Add amazing feature"
   ```

7. **Push al branch**:
   ```bash
   git push origin feature/amazing-feature
   ```

8. **Apri una Pull Request**

## 🛠️ Setup Ambiente di Sviluppo

### Prerequisiti
- Python 3.8+
- pip o conda
- git

### Installazione

```bash
# Clone repository
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze di sviluppo
pip install -r requirements.txt
pip install -e ".[dev]"

# Installa pre-commit hooks
pre-commit install
```

### Struttura del Progetto

```
credit-scoring-model/
│
├── src/                  # Codice sorgente
│   ├── data/            # Moduli per dati
│   ├── features/        # Feature engineering
│   ├── models/          # Implementazione modelli
│   ├── evaluation/      # Valutazione modelli
│   └── utils/           # Utilities
│
├── scripts/             # Script eseguibili
├── tests/               # Test suite
├── notebooks/           # Jupyter notebooks
├── config/              # File configurazione
└── docs/                # Documentazione
```

## 🔄 Processo di Sviluppo

### Workflow Git

1. **Sync con main**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Crea feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Sviluppa e testa**:
   ```bash
   # Fai modifiche
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Commit incrementali**:
   ```bash
   git add .
   git commit -m "Descrizione chiara delle modifiche"
   ```

5. **Push e PR**:
   ```bash
   git push origin feature/your-feature
   # Apri PR su GitHub
   ```

### Tipi di Branch

- `feature/` - Nuove feature
- `bugfix/` - Fix di bug
- `hotfix/` - Fix urgenti per produzione
- `refactor/` - Refactoring codice
- `docs/` - Modifiche documentazione

### Commit Messages

Segui il formato:
```
<tipo>: <descrizione breve>

<descrizione estesa (opzionale)>

<footer (opzionale)>
```

Tipi:
- `feat`: Nuova feature
- `fix`: Bug fix
- `docs`: Modifiche documentazione
- `style`: Formattazione codice
- `refactor`: Refactoring senza cambiare funzionalità
- `test`: Aggiunta o modifica test
- `chore`: Manutenzione

Esempio:
```
feat: add SHAP explainability module

Implements SHAP values calculation for model interpretability.
Includes visualization functions and feature contribution analysis.

Closes #42
```

## 📝 Linee Guida per il Codice

### Style Guide

Seguiamo [PEP 8](https://www.python.org/dev/peps/pep-0008/) con alcune personalizzazioni:

#### Formattazione
- Usa **Black** per formattazione automatica
- Lunghezza linea: 100 caratteri
- Indentazione: 4 spazi (no tab)

```bash
# Applica Black
black src/ tests/
```

#### Naming Conventions
- **Classi**: PascalCase (`DataLoader`, `RandomForestModel`)
- **Funzioni/Metodi**: snake_case (`load_data`, `calculate_metrics`)
- **Costanti**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`, `DEFAULT_THRESHOLD`)
- **Variabili private**: _leading_underscore (`_validate_data`)

#### Docstrings

Usa Google style docstrings:

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcola le metriche di valutazione del modello.
    
    Args:
        y_true: Array con etichette vere
        y_pred: Array con predizioni
        
    Returns:
        Dictionary con metriche calcolate
        
    Raises:
        ValueError: Se gli array hanno dimensioni diverse
        
    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.95
    """
    pass
```

#### Type Hints

Usa sempre type hints:

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    pass
```

### Best Practices

#### Codice Pulito
- Funzioni piccole e focalizzate (< 50 linee idealmente)
- Nomi descrittivi per variabili e funzioni
- Evita magic numbers (usa costanti)
- DRY (Don't Repeat Yourself)

#### Error Handling
```python
# Buono
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise

# Cattivo
try:
    result = risky_operation()
except:  # Evita catch-all generici
    pass
```

#### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Usa logging invece di print
logger.info("Processing started")
logger.warning("Low memory detected")
logger.error("Failed to load data", exc_info=True)
```

## 🧪 Testing

### Esegui Test

```bash
# Tutti i test
pytest tests/

# Con coverage
pytest --cov=src tests/

# Test specifico
pytest tests/test_data_loader.py

# Verbose
pytest -v tests/
```

### Scrivi Test

Ogni nuovo modulo deve avere test corrispondenti:

```python
# tests/test_feature_engineering.py
import pytest
from src.features.feature_engineering import FeatureEngineer

def test_create_age_features():
    """Test creazione feature età"""
    engineer = FeatureEngineer()
    df = pd.DataFrame({'DAYS_BIRTH': [-10000, -15000]})
    
    result = engineer.create_age_features(df)
    
    assert 'AGE_YEARS' in result.columns
    assert result['AGE_YEARS'].iloc[0] > 0
```

### Coverage Minima

- Moduli core: 80%+ coverage
- Utilities: 70%+ coverage
- Nuove feature: 80%+ coverage

## 📚 Documentazione

### README Updates

Se aggiungi feature significative, aggiorna:
- README.md
- Sezione "Utilizzo"
- Esempi di codice

### Docstring

Ogni funzione/classe pubblica deve avere docstring:
- Descrizione
- Args
- Returns
- Raises (se applicabile)
- Example (per funzioni complesse)

### Notebooks

Se aggiungi analisi:
- Crea notebook in `notebooks/`
- Aggiungi descrizioni markdown
- Pulisci output prima del commit

## 🔍 Code Review

Le Pull Request saranno revisionate per:
- ✅ Correttezza funzionale
- ✅ Test coverage adeguato
- ✅ Stile codice (Black, Flake8)
- ✅ Documentazione chiara
- ✅ Nessuna regressione
- ✅ Performance accettabili

## 📧 Contatti

Per domande:
- Apri una issue su GitHub
- Email: datascience@pronationalbank.com

## 🙏 Riconoscimenti

Tutti i contributori saranno riconosciuti nel README.

---

Grazie per contribuire al progetto Credit Scoring Model! 🎉
