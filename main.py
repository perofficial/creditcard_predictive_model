import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_data, feature_engineering, prepare_for_model
from src.models import train_best_rf, save_model
from sklearn.metrics import classification_report

def run_pipeline():
    print("🚀 Avvio pipeline di Credit Scoring...")
    
    # 1. Caricamento
    df = pd.read_csv('credit_scoring.csv')
    
    # 2. Preprocessing & Engineering
    df = clean_data(df)
    df = feature_engineering(df)
    df_final = prepare_for_model(df)
    
    # 3. Split
    X = df_final.drop(columns=['ID', 'TARGET'])
    y = df_final['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Training
    model = train_best_rf(X_train, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test)
    print("\n✅ Report finale del modello selezionato:")
    print(classification_report(y_test, y_pred))
    
    # 6. Export
    save_model(model)
    print("\n💾 Modello salvato nella cartella 'models/'")

if __name__ == "__main__":
    run_pipeline()