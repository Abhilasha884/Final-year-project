import pandas as pd

df = pd.read_csv("Data/labels_mapped.csv", encoding="latin1")

song_col = df.columns[0]  # take first column as song_id

replacements = {
    "â€™": "’",
    "â": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”"
}

for bad, good in replacements.items():
    df[song_col] = df[song_col].astype(str).str.replace(bad, good, regex=False)

df.to_csv("Data/labels_mapped.csv", index=False, encoding="utf-8-sig")

print("Encoding fixed in first column:", song_col)
