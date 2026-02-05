from src.data_processing import load_data, clean_and_engineer_features, preprocess_for_model
from src.model_training import CreditScoringModel
from sklearn.model_selection import train_test_split

def main():
    # 1. Pipeline Dati
    df = load_data('data/raw/credit_scoring.csv')
    df = clean_and_engineer_features(df)
    df_processed, feature_names = preprocess_for_model(df)

    # 2. Split
    X = df_processed.drop(columns=['TARGET'])
    y = df_processed['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Training (Usiamo il Random Forest vincitore)
    learner = CreditScoringModel()
    learner.train_random_forest(X_train, y_train, n_estimators=200)

    # 4. Valutazione
    metrics = learner.evaluate(X_test, y_test)
    print(f"Metriche Finali: {metrics}")

    # 5. Interpretabilità (Bonus)
    top_features = learner.get_feature_importance(X.columns)
    print("\nTop 5 Fattori di Rischio:")
    print(top_features.head(5))

if __name__ == "__main__":
    main()