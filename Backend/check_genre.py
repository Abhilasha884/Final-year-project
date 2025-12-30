import pandas as pd
from collections import Counter
from genre_mapper import map_to_main_genre, MAIN_GENRES

# # 🔹 Load dataset
# DATASET_PATH = "..data/labels.csv"   # <-- change if needed
CSV_FILE = "../data/labels.csv"
df = pd.read_csv(CSV_FILE)

# 🔹 Detect genre column automatically
genre_col = None
for col in df.columns:
    if "genre" in col.lower():
        genre_col = col
        break

if genre_col is None:
    raise ValueError("❌ No genre column found in dataset")

# 🔹 Map genres
mapped_genres = []

for g in df[genre_col]:
    main = map_to_main_genre(str(g))
    if main:
        mapped_genres.append(main)

# 🔹 Count genres
counts = Counter(mapped_genres)

print("\n📊 Genre distribution (mapped to MAIN genres):")
for genre in MAIN_GENRES:
    print(f"{genre}: {counts.get(genre, 0)}")

print("\n🔍 Total mapped songs:", sum(counts.values()))
print("🔍 Total rows in dataset:", len(df))
