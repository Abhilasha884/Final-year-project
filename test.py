import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL

# ===============================
# CONFIGURATION
# ===============================
GENIUS_TOKEN = "-nlF8VErk0OJ8kDZoKgJXAe4j78CSWZ2VLt_RydUHA2co1K8dmQ-om2THc9ugPHV"
dataset_folder = "data"
audio_folder = os.path.join(dataset_folder, "audio")
lyrics_folder = os.path.join(dataset_folder, "lyrics")
labels_csv = os.path.join(dataset_folder, "labels.csv")
failed_log = os.path.join(dataset_folder, "failed_songs.txt")

os.makedirs(audio_folder, exist_ok=True)
os.makedirs(lyrics_folder, exist_ok=True)

# ===============================
# HELPER: Safe filename
# ===============================
def safe_filename(name):
    """Remove invalid characters from filenames (Windows-safe)."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

# ===============================
# SONGS LIST WITH YOUTUBE URLS
# ===============================
songs_info = [

    ("Tujhe Dekha To", "Kumar Sanu", "Hindi", 0.65, 0.60, "Romantic Melody",
[
    "https://www.youtube.com/watch?v=cNV5hLSa9H8",
    "https://soundcloud.com/kumarsanu/tujhe-dekha-to",
    "https://en.wikipedia.org/wiki/Dilwale_Dulhania_Le_Jayenge"
]
),

("Phir Le Aaya Dil", "Arijit Singh", "Hindi", 0.30, 0.55, "Sad Romantic",
[
    "https://www.youtube.com/watch?v=5iZt7ZQ1XwQ",
    "https://soundcloud.com/arijitsinghofficial/phir-le-aaya-dil",
    "https://en.wikipedia.org/wiki/Phir_Le_Aaya_Dil"
]
),

("Aaj Kal Zindagi", "Shankar Mahadevan", "Hindi", 0.75, 0.70, "Feel Good Pop",
[
    "https://www.youtube.com/watch?v=K2s6Z9yVxTQ",
    "https://soundcloud.com/shankarehsaanloy/aaj-kal-zindagi",
    "https://en.wikipedia.org/wiki/Wake_Up_Sid"
]
),

("Abhi Mujh Mein Kahin", "Sonu Nigam", "Hindi", 0.25, 0.45, "Emotional Ballad",
[
    "https://www.youtube.com/watch?v=oWKgpB2zpgw",
    "https://soundcloud.com/sonunigam/abhi-mujh-mein-kahin",
    "https://en.wikipedia.org/wiki/Agneepath_(2012_film)"
]
),

("Iktara", "Kavita Seth", "Hindi", 0.55, 0.50, "Indie Romantic",
[
    "https://www.youtube.com/watch?v=fSS_R91Nimw",
    "https://soundcloud.com/kavitaseth/iktara",
    "https://en.wikipedia.org/wiki/Iktara"
]
),

("Dil Chahta Hai", "Shankar Mahadevan", "Hindi", 0.85, 0.80, "Friendship Pop",
[
    "https://www.youtube.com/watch?v=0u7Y8zjUu2M",
    "https://soundcloud.com/shankarehsaanloy/dil-chahta-hai",
    "https://en.wikipedia.org/wiki/Dil_Chahta_Hai_(song)"
]
),

("Yeh Haseen Wadiyan", "S. P. Balasubrahmanyam", "Hindi", 0.70, 0.65, "Melodic Romance",
[
    "https://www.youtube.com/watch?v=8dA2xZ7z3hM",
    "https://soundcloud.com/arrahman/yeh-haseen-wadiyan",
    "https://en.wikipedia.org/wiki/Roja_(1992_film)"
]
),

("Roja Jaaneman", "S. P. Balasubrahmanyam", "Hindi", 0.60, 0.60, "Romantic Melody",
[
    "https://www.youtube.com/watch?v=Z0B2Y2qYx9U",
    "https://soundcloud.com/arrahman/roja-jaaneman",
    "https://en.wikipedia.org/wiki/Roja_(1992_film)"
]
),

("Taal Se Taal Mila", "Alka Yagnik", "Hindi", 0.90, 0.85, "Classical Fusion",
[
    "https://www.youtube.com/watch?v=5Kp9KpJ1KqY",
    "https://soundcloud.com/arrahman/taal-se-taal-mila",
    "https://en.wikipedia.org/wiki/Taal_(film)"
]
),

("Pani Da Rang", "Ayushmann Khurrana", "Hindi", 0.40, 0.55, "Indie Folk",
[
    "https://www.youtube.com/watch?v=Y2zg9a6FqJ4",
    "https://soundcloud.com/ayushmannkhurrana/pani-da-rang",
    "https://en.wikipedia.org/wiki/Pani_Da_Rang"
]
),

("Aye Khuda", "Salim Merchant", "Hindi", 0.20, 0.50, "Spiritual Sad",
[
    "https://www.youtube.com/watch?v=ZzQ9n3gTn8g",
    "https://soundcloud.com/salimmerchant/aye-khuda",
    "https://en.wikipedia.org/wiki/Murder_2"
]
),

("Tu Hi Re", "Hariharan", "Hindi", 0.55, 0.60, "Romantic Classic",
[
    "https://www.youtube.com/watch?v=E5bRk6Zzq7A",
    "https://soundcloud.com/arrahman/tu-hi-re",
    "https://en.wikipedia.org/wiki/Bombay_(1995_film)"
]
),

("Maa", "Shankar Mahadevan", "Hindi", 0.30, 0.45, "Emotional",
[
    "https://www.youtube.com/watch?v=Y2FJYzP5yF4",
    "https://soundcloud.com/shankarehsaanloy/maa",
    "https://en.wikipedia.org/wiki/Taare_Zameen_Par"
]
),

("Jashn-e-Bahara", "Javed Ali", "Hindi", 0.65, 0.60, "Classical Romance",
[
    "https://www.youtube.com/watch?v=0X5Yp2QZ7ZQ",
    "https://soundcloud.com/arrahman/jashn-e-bahara",
    "https://en.wikipedia.org/wiki/Jodhaa_Akbar"
]
),

("Saibo", "Shreya Ghoshal", "Hindi", 0.45, 0.55, "Soft Romantic",
[
    "https://www.youtube.com/watch?v=G6sZp5H9n8E",
    "https://soundcloud.com/shreyaghoshal/saibo",
    "https://en.wikipedia.org/wiki/Shor_in_the_City"
]
),

("Tere Bina", "A. R. Rahman", "Hindi", 0.35, 0.50, "Romantic Sufi",
[
    "https://www.youtube.com/watch?v=9JDSGhhiOwI",
    "https://soundcloud.com/arrahman/tere-bina",
    "https://en.wikipedia.org/wiki/Guru_(2007_film)"
]
),

("Luka Chuppi", "Lata Mangeshkar", "Hindi", 0.15, 0.40, "Emotional Lullaby",
[
    "https://www.youtube.com/watch?v=_ikZtc6G9SU",
    "https://soundcloud.com/arrahman/luka-chuppi",
    "https://en.wikipedia.org/wiki/Rang_De_Basanti"
]
),

("Masakali", "Mohit Chauhan", "Hindi", 0.88, 0.82, "Indie Pop",
[
    "https://www.youtube.com/watch?v=SS3lIQdKP-A",
    "https://soundcloud.com/arrahman/masakali",
    "https://en.wikipedia.org/wiki/Delhi-6"
]
),

("Khaabon Ke Parinday", "Mohit Chauhan", "Hindi", 0.85, 0.78, "Travel Pop",
[
    "https://www.youtube.com/watch?v=R0XjwtP_iTY",
    "https://soundcloud.com/shankarehsaanloy/khaabon-ke-parinday",
    "https://en.wikipedia.org/wiki/Zindagi_Na_Milegi_Dobara"
]
),

("Zara Zara", "Bombay Jayashri", "Hindi", 0.45, 0.55, "Romantic Ghazal",
[
    "https://www.youtube.com/watch?v=Y0Q9qkXz1F8",
    "https://soundcloud.com/bombayjayashri/zara-zara",
    "https://en.wikipedia.org/wiki/Rehnaa_Hai_Terre_Dil_Mein"
]
)


]

# ===============================
# INITIALIZE GENIUS API
# ===============================
genius = lyricsgenius.Genius(GENIUS_TOKEN, timeout=15, retries=3)

# ===============================
# FUNCTION: Fetch Lyrics
# ===============================
def fetch_lyrics(song_title, artist_name):
    try:
        song = genius.search_song(song_title, artist_name)
        if song and song.lyrics:
            return song.lyrics
    except Exception as e:
        print(f"❌ Error fetching lyrics for {song_title}: {e}")
    return None

# ===============================
# FUNCTION: Download Audio (60-sec)
# ===============================
def download_audio(youtube_url, filepath, duration=60):
    folder, base = os.path.split(filepath)
    base = safe_filename(base)
    full_path = os.path.join(folder, base)

    print(f"➡️ Downloading trimmed audio to: {full_path}")

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': full_path,
            'quiet': True,
            'noplaylist': True,
            'geo_bypass': True,
            'user_agent': 'Mozilla/5.0',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-t', str(duration)  # limit duration in seconds
            ]
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"✅ 60-sec audio downloaded: {full_path}")
        return full_path
    except Exception as e:
        print(f"❌ Error downloading {youtube_url}: {e}")
        return None


# ===============================
# MAIN LOOP
# ===============================
data = []
failed_songs = []

for song_title, artist, language, valence, arousal, genre, youtube_url in songs_info:
    song_id = f"{language.lower()}_{safe_filename(song_title.replace(' ', '_'))}"

    try:
        # --- Fetch Lyrics ---
        lyrics = fetch_lyrics(song_title, artist)
        if lyrics:
            lyrics_file = os.path.join(lyrics_folder, f"{song_id}.txt")
            with open(lyrics_file, "w", encoding="utf-8") as f:
                f.write(lyrics)
            print(f"📝 Lyrics saved for {song_title}")
        else:
            print(f"⚠️ Lyrics not found for {song_title}")

        # --- Download 60-sec Audio ---
        audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
        downloaded_path = download_audio(youtube_url, audio_file)
        if not downloaded_path:
            failed_songs.append(song_title)
            continue

        # --- Add to dataset ---
        data.append([song_id, language, artist, valence, arousal, genre])

    except Exception as e:
        print(f"❌ Unexpected error processing {song_title}: {e}")
        failed_songs.append(song_title)

# ===============================
# SAVE LABELS CSV
# ===============================
df = pd.DataFrame(data, columns=["song_id", "language", "singer", "valence", "arousal", "genre"])
if os.path.exists(labels_csv):
    old_df = pd.read_csv(labels_csv)
    combined_df = pd.concat([old_df, df], ignore_index=True)
    combined_df.drop_duplicates(subset=["song_id"], keep="first", inplace=True)
    combined_df.to_csv(labels_csv, index=False)
else:
    df.to_csv(labels_csv, index=False)

# ===============================
# SAVE FAILED SONGS LOG
# ===============================
if failed_songs:
    with open(failed_log, "w", encoding="utf-8") as f:
        f.write("\n".join(failed_songs))
    print(f"⚠️ Some songs failed and were logged to {failed_log}")

print("✅ Dataset creation complete! Audio downloaded (60-sec), lyrics saved, and CSV updated.")
