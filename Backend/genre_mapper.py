# genre_mapper.py

# =========================
# MAIN GENRES (MODEL OUTPUT CLASSES)
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

GENRE_TO_IDX = {g: i for i, g in enumerate(MAIN_GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}


# =========================
# FINE → MAIN GENRE MAPPING
# =========================
GENRE_MAPPING = {

    # ---------- POP ----------
    "Pop": "Pop",
    "Pop Rock": "Pop",
    "Pop ballad": "Pop",
    "Pop Ballad": "Pop",
    "Pop/Dance": "Pop",
    "Pop/Electronic": "Pop",
    "Synthpop": "Pop",
    "K-Pop": "Pop",
    "Dance Pop": "Pop",
    "Indie Pop": "Pop",
    "Bollywood Pop": "Pop",
    "Romantic": "Pop",
    "Love Song": "Pop",

    # ---------- ROCK ----------
    "Rock": "Rock",
    "Classic Rock": "Rock",
    "Hard Rock": "Rock",
    "Alternative Rock": "Rock",
    "Indie Rock": "Rock",
    "Punk Rock": "Rock",
    "Soft Rock": "Rock",
    "Pop-Rock": "Rock",
    "Progressive Rock": "Rock",

    # ---------- HIP-HOP ----------
    "Hip-Hop": "Hip-Hop",
    "Rap": "Hip-Hop",
    "Hip Pop": "Hip-Hop",
    "Trap": "Hip-Hop",
    "Desi Hip-Hop": "Hip-Hop",

    # ---------- R&B ----------
    "R&B": "R&B",
    "R&B Pop": "R&B",
    "Soul": "R&B",
    "Neo Soul": "R&B",
    "Funk": "R&B",

    # ---------- ELECTRONIC ----------
    "Electronic": "Electronic",
    "EDM": "Electronic",
    "Dance": "Electronic",
    "House": "Electronic",
    "Techno": "Electronic",
    "Trance": "Electronic",
    "Dubstep": "Electronic",

    # ---------- JAZZ ----------
    "Jazz": "Jazz",
    "Smooth Jazz": "Jazz",
    "Soul/Jazz": "Jazz",
    "Blues": "Jazz",

    # ---------- CLASSICAL ----------
    "Classical": "Classical",
    "Classic": "Classical",
    "Indian Classical": "Classical",
    "Hindustani Classical": "Classical",
    "Carnatic": "Classical",
    "Semi-Classical": "Classical",
    "Folk": "Classical",
    "Folk/Pop": "Classical",
    "Instrumental": "Classical",
    "Ghazan": "Classical",
    "Emotional": "Classical",
    "Devotional": "Classical",

    # ---------- COUNTRY ----------
    "Country": "Country",
    "Country Pop": "Country",
    "Country Rock": "Country",

    # ---------- SPECIAL / EDGE ----------
    "Patriotic": "Classical",
    "Soundtrack": "Classical",
    "Film": "Pop",
    "Bollywood": "Pop",
}


# def map_to_main_genre(genre: str):
#     """
#     Maps fine-grained genre to one of the 8 MAIN genres.
#     Returns None if unmapped.
#     """
#     if not genre:
#         return None

#     genre = genre.strip()
#     return GENRE_MAPPING.get(genre, None)

def map_to_main_genre(genre: str):
    if not genre:
        return None

    g = genre.lower()

    # ROCK
    if any(x in g for x in ["rock", "metal", "grunge", "alternative"]):
        return "Rock"

    # POP
    if any(x in g for x in ["pop", "k-pop", "dance pop", "synth"]):
        return "Pop"

    # HIP-HOP
    if any(x in g for x in ["hip", "rap", "trap"]):
        return "Hip-Hop"

    # R&B / SOUL
    if any(x in g for x in ["r&b", "soul", "neo-soul"]):
        return "R&B"

    # ELECTRONIC
    if any(x in g for x in ["edm", "electronic", "dance", "house", "techno"]):
        return "Electronic"

    # JAZZ
    if "jazz" in g:
        return "Jazz"

    # CLASSICAL / INDIAN
    if any(x in g for x in ["classical", "instrumental", "orchestral"]):
        return "Classical"

    # COUNTRY / FOLK / PATRIOTIC
    if any(x in g for x in ["country", "folk", "patriotic", "devotional"]):
        return "Country"

    return None

