import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve


class RandomForestModel:
    def __init__(self, res_folder="rf_res", seed: int = 42):
        """
        Random Forest dla klasyfikacji wieloklasowej 0–4.
        Skalowanie nie jest wymagane (drzewa nie są wrażliwe na skalę cech).
        """
        self.res_folder = res_folder
        self.model_name = "Random_Forest"
        self.seed = seed

        # Spójne nazwy klas do wykresów
        self.class_names = ["Zdrowy", "St1", "St2", "St3", "St4"]

        # Model bazowy (bez tuningu)
        # class_weight="balanced" pomaga przy nierównowadze klas
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        )

        if not os.path.exists(self.res_folder):
            os.mkdir(self.res_folder)

    def train(self, x_train, y_train, tune: bool = True):
        print(f"--- Trenowanie modelu {self.model_name} ---")

        if not tune:
            self.model.fit(x_train, y_train)
            return

        # Siatka tuningu (sensowna dla datasetu tej wielkości)
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        search.fit(x_train, y_train)

        print("Najlepsze parametry:", search.best_params_)
        print("Najlepszy wynik CV (f1_macro):", search.best_score_)

        # Podmieniamy model na najlepszy
        self.model = search.best_estimator_

        # Zapis parametrów do pliku
        params_path = os.path.join(self.res_folder, f"{self.model_name}_best_params.txt")
        with open(params_path, "w", encoding="utf-8") as f:
            f.write("Best params:\n")
            f.write(str(search.best_params_) + "\n\n")
            f.write("Best CV score (f1_macro):\n")
            f.write(str(search.best_score_) + "\n")

        print(f"Parametry zapisane w: {params_path}")

    def evaluate(self, x_test, y_test):
        """
        Zapisuje classification_report do pliku txt (jak w NB/SVM).
        """
        y_pred = self.model.predict(x_test)

        report = classification_report(
            y_test, y_pred,
            target_names=[f"Stage {i}" for i in range(5)]
        )

        file_path = os.path.join(self.res_folder, f"{self.model_name}_report.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Raport klasyfikacji dla {self.model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)

        print(f"Raport zapisany w: {file_path}")
        return y_pred

    def plot_diagnostics(self, x_train, y_train, x_test, y_test):
        """
        Generuje:
        - confusion matrix (PNG)
        - ROC curves (PNG)
        - Precision-Recall curves (PNG)
        """
        print(f"--- Generowanie wykresów dla {self.model_name} ---")

        # --- MACIERZ POMYŁEK ---
        plt.figure(figsize=(10, 8))
        y_pred = self.model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywista")
        plt.tight_layout()
        plt.savefig(os.path.join(self.res_folder, f"{self.model_name}_cm.png"))
        plt.close()

        # --- KRZYWA ROC ---
        roc = ROCAUC(
            self.model,
            classes=self.class_names,
            title=f"ROC Curves - {self.model_name}"
        )
        roc.fit(x_train, y_train)
        roc.score(x_test, y_test)
        roc.show(outpath=os.path.join(self.res_folder, f"{self.model_name}_roc.png"))
        plt.close()

        # --- KRZYWA PRECISION-RECALL ---
        pr = PrecisionRecallCurve(
            self.model,
            classes=self.class_names,
            title=f"P-R Curves - {self.model_name}"
        )
        pr.fit(x_train, y_train)
        pr.score(x_test, y_test)
        pr.show(outpath=os.path.join(self.res_folder, f"{self.model_name}_pr_curve.png"))
        plt.close()

    def plot_feature_importance(self, feature_names, top_n: int = 13):
        """
        Opcjonalnie: wykres ważności cech (bardzo dobrze wygląda w raporcie).
        """
        if not hasattr(self.model, "feature_importances_"):
            print("Model nie ma feature_importances_. Upewnij się, że to RandomForestClassifier.")
            return

        importances = self.model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
        plt.title(f"Feature Importances - {self.model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.res_folder, f"{self.model_name}_feature_importances.png"))
        plt.close()
