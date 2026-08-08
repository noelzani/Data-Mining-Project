import pandas as pd

COLUMNS = [
    "Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash",
    "Magnesium", "Total_phenols", "Flavanoids",
    "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "OD280/OD315", "Proline"
]

def load_wine_data(path="data/wine.data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df
