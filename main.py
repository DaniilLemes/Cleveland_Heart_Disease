from heart_disease_loader.loader import HeartDiseaseLoader
from bayes.nb_classifier import NaiveBayesModel


def main():
    # Wczytanie i przygotowanie danych
    loader = HeartDiseaseLoader("data/processed.cleveland.data")
    loader.load_clean_data()
    X_train, X_test, y_train, y_test = loader.prepare_data(stratify=True)

    # Inicjalizacja modelu Bayesowskiego
    nb_model = NaiveBayesModel(res_folder="results_bayes")
    nb_model.train(X_train, y_train)

    # Ewaluacja (zapisuje plik tekstowy)
    nb_model.evaluate(X_test, y_test)

    # Grafiki (zapisuje PNG w folderze results_bayes)
    nb_model.plot_diagnostics(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
