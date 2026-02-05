from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def train_best_rf(X_train, y_train):
    """Addestra il miglior modello Random Forest configurato."""
    # Usiamo i parametri che hanno dato i risultati migliori nel tuo test
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def save_model(model, path='models/best_model.pkl'):
    joblib.dump(model, path)