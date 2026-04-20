from src.cleaning import load_data, clean_data, get_info
from src.visualization import visualize_all
from src.model import prepare_data, train_model, evaluate_model, save_model

def main():
    print("🚀 Démarrage du pipeline\n")

    # Étape 1 - Chargement
    print("📂 Chargement des données...")
    df = load_data('data/diabetes.csv')

    # Étape 2 - Informations
    print("\n📊 Informations sur les données...")
    get_info(df)

    # Étape 3 - Nettoyage
    print("\n🧹 Nettoyage des données...")
    df = clean_data(df)

    # Étape 4 - Visualisation
    print("\n🎨 Génération des graphiques...")
    visualize_all(df)

    # Étape 5 - Préparation
    print("\n⚙️ Préparation des données...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Étape 6 - Entraînement
    print("\n🤖 Entraînement du modèle...")
    model = train_model(X_train, y_train)

    # Étape 7 - Évaluation
    print("\n📈 Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)

    # Étape 8 - Sauvegarde
    print("\n💾 Sauvegarde du modèle...")
    save_model(model)

    print("\n✅ Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()