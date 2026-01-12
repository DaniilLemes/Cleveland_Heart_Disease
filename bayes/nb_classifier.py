import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve


class NaiveBayesModel:
    def __init__(self, res_folder="nb_res"):
        self.model = GaussianNB()
        self.res_folder = res_folder
        self.model_name = 'Naive_Bayes'
        # Mapowanie klas dla Yellowbrick
        self.label_map = {0: 'Zdrowy', 1: 'St1', 2: 'St2', 3: 'St3', 4: 'St4'}

        if not os.path.exists(self.res_folder):
            os.mkdir(self.res_folder)

    def train(self, x_train, y_train):
        print(f"--- Trenowanie modelu {self.model_name} ---")
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=[f"Stage {i}" for i in range(5)])

        file_path = os.path.join(self.res_folder, f"{self.model_name}_report.txt")
        with open(file_path, "w") as f:
            f.write(f"Raport klasyfikacji dla {self.model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)

        print(f"Raport zapisany w: {file_path}")
        return y_pred

    def plot_diagnostics(self, x_train, y_train, x_test, y_test):
        print(f"--- Generowanie wykresów dla {self.model_name} ---")

        # Definiuje nazwy klas raz dla wszystkich wykresów
        class_names = ['Zdrowy', 'St1', 'St2', 'St3', 'St4']

        # --- MACIERZ POMYŁEK ---
        plt.figure(figsize=(10, 8))
        y_pred = self.model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.savefig(os.path.join(self.res_folder, f"{self.model_name}_cm.png"))
        plt.close()

        # --- KRZYWA ROC ---
        visualizer = ROCAUC(self.model, classes=class_names, title=f"ROC Curves - {self.model_name}")
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.show(outpath=os.path.join(self.res_folder, f"{self.model_name}_roc.png"))
        plt.close()

        # --- KRZYWA PRECISION-RECALL ---
        pr_curve = PrecisionRecallCurve(self.model, classes=class_names, title=f"P-R Curves - {self.model_name}")
        pr_curve.fit(x_train, y_train)
        pr_curve.score(x_test, y_test)
        pr_curve.show(outpath=os.path.join(self.res_folder, f"{self.model_name}_pr_curve.png"))
        plt.close()
