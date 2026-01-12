import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HeartDiseaseLoader:
    def __init__(self, file_path: str, seed: int = 42, test_size: float = 0.25):
        self.file_path = file_path
        self.seed = seed
        self.test_size = test_size

        self.column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]

        self.data = None

        # Split data holders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Scaler for standardized data (e.g., SVM)
        self.scaler = StandardScaler()

    def load_clean_data(self) -> pd.DataFrame:
        """Wczytuje dane, obsługuje brakujące wartości i konwertuje typy."""
        try:
            # UCI Cleveland: brak nagłówka, separator = ','
            self.data = pd.read_csv(
                self.file_path,
                header=None,
                names=self.column_names,
                na_values="?",
                sep=",",
                engine="python"
            )

            # Konwersja wszystkich kolumn na numeric (gdzie się nie da -> NaN)
            for c in self.column_names:
                self.data[c] = pd.to_numeric(self.data[c], errors="coerce")

            before = len(self.data)
            if self.data.isnull().values.any():
                self.data = self.data.dropna()
            after = len(self.data)

            print(f"--- Dane wczytane: {self.file_path} ---")
            print(f"--- Rekordy: {before} -> {after} po usunięciu braków ---")

        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Nieoczekiwany błąd podczas wczytywania danych: {e}")

        return self.data

    def prepare_data(self, stratify: bool = True):
        """Dzieli dane na X i y oraz na zbiory treningowe/testowe (75/25). MULTI-CLASS 0–4."""
        if self.data is None:
            raise ValueError("Brak danych. Najpierw uruchom load_clean_data().")

        X = self.data.drop(columns=["target"])
        y = self.data["target"].astype(int)  # 0..4

        strat = y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=42,
            stratify=strat
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_standardized_data(self):
        """Zwraca dane przeskalowane (StandardScaler) - przydatne np. dla SVM."""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Brak podziału train/test. Najpierw uruchom prepare_data().")

        X_train_std = self.scaler.fit_transform(self.X_train)
        X_test_std = self.scaler.transform(self.X_test)
        return X_train_std, X_test_std

    def print_data_summary(self):
        """Wyświetla sformatowane informacje o zbiorze w terminalu."""
        if self.data is None:
            print("Brak danych do wyświetlenia. Najpierw użyj load_clean_data().")
            return

        print("\n" + "=" * 70)
        print("PIERWSZE 5 REKORDÓW:")
        print("=" * 70)
        print(self.data.head())

        print("\n" + "=" * 70)
        print("INFORMACJE O TYPACH I BRAKACH:")
        print("=" * 70)
        # info() drukuje do stdout i zwraca None
        self.data.info()

        print("\n" + "=" * 70)
        print("ROZKŁAD KLAS (target: 0 zdrowy, 1-4 stadia):")
        print("=" * 70)
        print(self.data["target"].astype(int).value_counts().sort_index())
        print("=" * 70 + "\n")
