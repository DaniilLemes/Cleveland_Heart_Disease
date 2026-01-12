import pandas as pd


class HeartDiseaseLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        self.data = None

    def load_clean_data(self):
        """Wczytuje dane, obsługuje brakujące wartości i konwertuje typy."""
        try:
            # Wczytanie danych - '?' traktowane jako NaN
            self.data = pd.read_csv(self.file_path, names=self.column_names, na_values='?')

            if self.data.isnull().values.any():
                self.data = self.data.dropna()

            self.data['ca'] = pd.to_numeric(self.data['ca'])
            self.data['thal'] = pd.to_numeric(self.data['thal'])

            print(f"--- Dane wczytane pomyślnie z: {self.file_path} ---")
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku {self.file_path}")
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd: {e}")

        return self.data

    def print_data_summary(self):
        """Wyświetla sformatowane informacje o zbiorze w terminalu."""
        if self.data is not None:
            print("\n" + "=" * 50)
            print("PIERWSZE 5 REKORDÓW:")
            print("=" * 50)
            print(self.data.head())

            print("\n" + "=" * 50)
            print("INFORMACJE O TYPACH I BRAKACH:")
            print("=" * 50)
            print(self.data.info())

            print("\n" + "=" * 50)
            print("ROZKŁAD KLAS (0-zdrowy, 1-4 stadia choroby):")
            print("=" * 50)
            print(self.data['target'].value_counts().sort_index())
            print("=" * 50 + "\n")
        else:
            print("Brak danych do wyświetlenia. Najpierw użyj load_clean_data().")
