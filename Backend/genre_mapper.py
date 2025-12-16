# genre_mapper.py

# =========================
# MAIN GENRES (used for training + UI)
# =========================
MAIN_GENRES = [
    "Pop",
    "Rock",
    "Hip-Hop",
    "R&B",
    "Electronic",
    "Jazz",
    "Classical",
    "Country"
]


# =========================
# Fine → Main genre mapping
# =========================
GENRE_MAPPING = {
    # POP
    "Pop": "Pop",
    "Pop Rock": "Pop",
    "Pop ballad": "Pop",
    "Pop Ballad": "Pop",
    "Pop/Dance": "Pop",
    "Pop/Electronic": "Pop",
    "Synthpop": "Pop",
    "K-Pop": "Pop",

    # ROCK
    "Rock": "Rock",
    "Classic Rock": "Rock",
    "Hard Rock": "Rock",
    "Alternative Rock": "Rock",
    "Indie Rock": "Rock",
    "Punk Rock": "Rock",

    # HIP-HOP
    "Hip-Hop": "Hip-Hop",
    "Rap": "Hip-Hop",
    "Hip Pop": "Hip-Hop",

    # R&B
    "R&B": "R&B",
    "R&B Pop": "R&B",
    "Soul": "R&B",

    # ELECTRONIC
    "Electronic": "Electronic",
    "EDM": "Electronic",
    "Dance": "Electronic",

    # JAZZ
    "Jazz": "Jazz",
    "Soul/Jazz": "Jazz",

    # CLASSICAL
    "Classical": "Classical",
    "Classic": "Classical",

    # COUNTRY
    "Country": "Country",
    "Country Pop": "Country",
}


def map_to_main_genre(genre: str):
    """
    Maps fine-grained genre to main genre.
    Returns None if no mapping found.
    """
    if not genre:
        return None

    genre = genre.strip()
    return GENRE_MAPPING.get(genre, None)
