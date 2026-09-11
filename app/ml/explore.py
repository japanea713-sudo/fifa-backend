
import pandas as pd
from pathlib import Path

# This builds a path relative to THIS file, no matter where you run it from
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # goes up to backend/
DATA_PATH = BASE_DIR / "data" / "players_25.csv"

df = pd.read_csv(DATA_PATH)

print(df.shape) 
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.isnull().sum())