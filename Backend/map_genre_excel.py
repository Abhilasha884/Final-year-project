import pandas as pd

# =========================
# FILE PATHS
# =========================
INPUT_CSV = "../data/labels.csv"          # original file
OUTPUT_CSV = "../data/labels_mapped.csv" # new file with mapped genre

# =========================
# GENRE MAPPING FUNCTION
# =========================
def map_to_main_genre(genre: str):
    if not isinstance(genre, str):
        return None

    g = genre.lower()

    # ROCK
    if any(x in g for x in ["rock", "metal", "grunge", "alternative"]):
        return "Rock"

    # POP
    if any(x in g for x in ["pop", "k-pop", "dance pop", "synth", "romantic", "love"]):
        return "Pop"

    # HIP-HOP
    if any(x in g for x in ["hip", "rap", "trap"]):
        return "Hip-Hop"

    # R&B / SOUL
    if any(x in g for x in ["r&b", "soul", "neo-soul", "funk"]):
        return "R&B"

    # ELECTRONIC
    if any(x in g for x in ["edm", "electronic", "dance", "house", "techno", "trance", "dubstep"]):
        return "Electronic"

    # JAZZ
    if any(x in g for x in ["jazz", "blues"]):
        return "Jazz"

    # CLASSICAL / INDIAN
    if any(x in g for x in [
        "classical", "instrumental", "orchestral",
        "hindustani", "carnatic", "ghazal", "semi-classical"
    ]):
        return "Classical"

    # COUNTRY / FOLK / PATRIOTIC
    if any(x in g for x in ["country", "folk", "patriotic", "devotional"]):
        return "Country"

    return None


# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":

    print(" Loading dataset...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    if "genre" not in df.columns:
        raise ValueError("❌ 'genre' column not found in dataset")

    print(" Mapping genres to MAIN genres...")
    df["mapped_genre"] = df["genre"].apply(map_to_main_genre)

    # Stats
    total_rows = len(df)
    mapped_rows = df["mapped_genre"].notna().sum()

    print("\n Mapping summary:")
    print(df["mapped_genre"].value_counts(dropna=False))

    print(f"\n Mapped songs: {mapped_rows}/{total_rows}")

    # Save new CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n New dataset saved as: {OUTPUT_CSV}")
