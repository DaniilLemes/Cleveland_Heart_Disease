from heart_disease_loader.loader import HeartDiseaseLoader


def main():
    path = "data/processed.cleveland.data"

    loader = HeartDiseaseLoader(path)
    df = loader.load_clean_data()
    loader.print_data_summary()


if __name__ == "__main__":
    main()