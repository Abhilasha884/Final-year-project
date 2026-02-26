import pandas as pd

# ---------------- FILE PATHS ----------------
muse_path = "data/muse_v3.csv"
labels_path = "data/labels.csv"

# ---------------- LOAD DATA ----------------
muse = pd.read_csv(muse_path)
labels = pd.read_csv(labels_path)

# ---------------- EXTRACT EXISTING SONG TITLES ----------------
labels['title'] = labels['song_id'].apply(
    lambda x: x.split("_", 1)[1].replace("_", " ") if "_" in x else x
)

existing_titles = set(labels['title'].str.lower())

# ---------------- FILTER ELECTRONIC + SUBGENRES ----------------
electronic_df = muse[muse['genre'].str.contains(
    "electronic|edm|techno|house|trance|ambient|dubstep|drum|synth",
    case=False, na=False
)].copy()

# ---------------- REMOVE SONGS ALREADY IN LABELS ----------------
electronic_df = electronic_df[
    ~electronic_df['track'].str.lower().isin(existing_titles)
]

# ---------------- REMOVE DUPLICATES ----------------
electronic_df.drop_duplicates(subset=["track", "artist"], inplace=True)

# ---------------- REMOVE MISSING EMOTION VALUES ----------------
electronic_df = electronic_df.dropna(subset=["valence_tags", "arousal_tags"])

print("Available Electronic Songs:", len(electronic_df))

# ---------------- SELECT 1000 ----------------
electronic_1000 = electronic_df.tail(1000)

# ---------------- FORMAT OUTPUT ----------------
formatted = []

for _, row in electronic_1000.iterrows():
    formatted.append(
        f'("{row["track"]}", "{row["artist"]}", {round(row["valence_tags"],2)}, {round(row["arousal_tags"],2)}),'
    )

# ---------------- SAVE FILE ----------------
with open("electronic_1000_list.py", "w", encoding="utf-8") as f:
    f.write("\n".join(formatted))

print("✅ 1000 Electronic songs saved to electronic_1000_list.py")