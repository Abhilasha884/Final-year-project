import pandas as pd
from genre_mapper_full import map_to_main_genre

INPUT_CSV = "Data/labels.csv"
OUTPUT_CSV = "Data/labels_mapped.csv"

print("Loading labels.csv...")
df = pd.read_csv(INPUT_CSV)

print("Mapping sub-genres to main genres...")
df["main_genre"] = df["genre"].apply(map_to_main_genre)

df.to_csv(OUTPUT_CSV, index=False)

print("Done! New file saved as Data/labels_mapped.csv")
