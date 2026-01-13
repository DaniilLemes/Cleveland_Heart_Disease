from heart_disease_loader.loader import HeartDiseaseLoader
from bayes.nb_classifier import NaiveBayesModel
from svm.svm_classifier import SVMModel


def main():
    loader = HeartDiseaseLoader("data/processed.cleveland.data")
    loader.load_clean_data()
    X_train, X_test, y_train, y_test = loader.prepare_data(stratify=True)

    # --- NAIVE BAYES ---
    # nb_model = NaiveBayesModel(res_folder="results_bayes")
    # nb_model.train(X_train, y_train)
    # nb_model.evaluate(X_test, y_test)
    # nb_model.plot_diagnostics(X_train, y_train, X_test, y_test)

    # --- SVM ---
    svm_model = SVMModel(res_folder="results_svm")
    svm_model.train(X_train, y_train)
    svm_model.evaluate(X_test, y_test)
    svm_model.plot_diagnostics(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
