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

    # 💖 Emotional Pop Ballads
    ("My Heart Will Go On", "Celine Dion", "English", 0.4, 0.5, "Pop Ballad", "https://youtu.be/A3QAqZQYLIQ?si=g3Cj6YqhATw8NEMb"),
    ("Because You Loved Me", "Celine Dion", "English", 0.55, 0.5, "Pop Ballad", "https://www.youtube.com/watch?v=JDcuRgk-Jd0"),
    ("I Will Always Love You", "Whitney Houston", "English", 0.3, 0.4, "Soul Ballad", "https://www.youtube.com/watch?v=FxYw0XPEoKE"),
    ("Un-break My Heart", "Toni Braxton", "English", 0.35, 0.45, "R&B Ballad", "https://www.youtube.com/watch?v=p2Rch6WvPJE"),
    ("You’re Beautiful", "James Blunt", "English", 0.5, 0.4, "Pop Ballad", "https://www.youtube.com/watch?v=nX1VeFBo9AQ"),

    # 🎉 Feel-good / Funky Pop
    ("Mambo No.5", "Lou Bega", "English", 0.95, 0.9, "Latin Pop", "https://www.youtube.com/watch?v=4kHl4FoK1Ys"),
    ("Livin’ la Vida Loca", "Ricky Martin", "English", 0.9, 0.9, "Pop", "https://www.youtube.com/watch?v=4kHl4FoK1Ys"),
    ("Can’t Get You Out of My Head", "Kylie Minogue", "English", 0.85, 0.8, "Dance Pop", "https://www.youtube.com/watch?v=c18441Eh_WE"),
    ("Hey Ya!", "Outkast", "English", 0.95, 0.85, "Funk Pop", "https://www.youtube.com/watch?v=PWgvGjAhvIw"),
    ("Crazy in Love", "Beyoncé ft. Jay-Z", "English", 0.9, 0.9, "R&B Pop", "https://www.youtube.com/watch?v=ViwtNLUqkMY"),

    # 🎸 Pop Rock / Feel-good Rock
    ("It’s My Life", "Bon Jovi", "English", 0.8, 0.85, "Pop Rock", "https://www.youtube.com/watch?v=vx2u5uUu3DE"),
    ("Complicated", "Avril Lavigne", "English", 0.75, 0.7, "Pop Rock", "https://www.youtube.com/watch?v=5NPBIwQyPWE"),
    ("Smooth", "Santana ft. Rob Thomas", "English", 0.9, 0.85, "Latin Rock", "https://www.youtube.com/watch?v=6Whgn_iE5uc"),
    ("Viva La Vida", "Coldplay", "English", 0.8, 0.6, "Pop Rock", "https://www.youtube.com/watch?v=dvgZkm1xWPE"),
    ("Drops of Jupiter", "Train", "English", 0.7, 0.6, "Pop Rock", "https://www.youtube.com/watch?v=7Xf-Lesrkuc"),

    # 💃 Dance / Disco Revival
    ("Hung Up", "Madonna", "English", 0.9, 0.85, "Dance Pop", "https://www.youtube.com/watch?v=EDwb9jOVRtU"),
    ("Don’t Stop Movin’", "S Club 7", "English", 0.9, 0.8, "Dance Pop", "https://www.youtube.com/watch?v=5s9Xl5v7VnI"),
    ("Rock Your Body", "Justin Timberlake", "English", 0.85, 0.9, "Funk Pop", "https://www.youtube.com/watch?v=TSVHoHyErBQ"),
    ("Can’t Get Enough of You Baby", "Smash Mouth", "English", 0.85, 0.75, "Pop Rock", "https://www.youtube.com/watch?v=8e2xBf0tV2o"),
    ("Don’t Stop Believin’", "Journey", "English", 0.9, 0.85, "Rock", "https://www.youtube.com/watch?v=1k8craCGpgs")

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
