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

# Optional: SOCKS5 proxy for free VPN (Windscribe / ProtonVPN)
# Format: "socks5://username:password@host:port"
VPN_PROXY = None  # e.g., "socks5://username:password@us-free.windscribe.com:1080"

# ===============================
# HELPER: Safe filename
# ===============================
def safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# ===============================
# SONGS LIST WITH CORRECT YOUTUBE LINKS ONLY
# ===============================
songs_info = [

    (
    "Flowers", "Miley Cyrus", "English", 0.8, 0.6, "Pop",
    [
        "https://www.youtube.com/watch?v:G7KNmW9a75Y",
        "https://en.wikipedia.org/wiki/Flowers_(Miley_Cyrus_song)"
    ]
),

(
    "Anti-Hero", "Taylor Swift", "English", 0.55, 0.45, "Pop",
    [
        "https://www.youtube.com/watch?v=b1kbLwvqugk",
        "https://en.wikipedia.org/wiki/Anti-Hero_(Taylor_Swift_song)"
    ]
),

(
    "Levitating", "Dua Lipa", "English", 0.8, 0.8, "Pop/Disco",
    [
        "https://www.youtube.com/watch?v=TUVcZfQe-Kw",
        "https://en.wikipedia.org/wiki/Levitating_(song)"
    ]
),

(
    "As It Was", "Harry Styles", "English", 0.8, 0.7, "Pop",
    [
        "https://www.youtube.com/watch?v=H5v3kku4y6Q",
        "https://en.wikipedia.org/wiki/As_It_Was"
    ]
),

(
    "Save Your Tears", "The Weeknd", "English", 0.5, 0.4, "Synthpop",
    [
        "https://www.youtube.com/watch?v=XXYlFuWEuKI",
        "https://en.wikipedia.org/wiki/Save_Your_Tears"
    ]
),

(
    "Good 4 U", "Olivia Rodrigo", "English", 0.9, 0.9, "Pop/Rock",
    [
        "https://www.youtube.com/watch?v=gNi_6U5Pm_o",
        "https://en.wikipedia.org/wiki/Good_4_U"
    ]
),

(
    "Stay", "The Kid LAROI, Justin Bieber", "English", 0.8, 0.9, "Pop",
    [
        "https://www.youtube.com/watch?v=kTJczUoc26U",
        "https://en.wikipedia.org/wiki/Stay_(The_Kid_Laroi_and_Justin_Bieber_song)"
    ]
),

(
    "Shivers", "Ed Sheeran", "English", 0.85, 0.8, "Pop",
    [
        "https://www.youtube.com/watch?v=Il0S8BoucSA",
        "https://en.wikipedia.org/wiki/Shivers_(Ed_Sheeran_song)"
    ]
),

(
    "Unholy", "Sam Smith, Kim Petras", "English", 0.6, 0.85, "Pop",
    [
        "https://www.youtube.com/watch?v=JnBMVQzA4Co",
        "https://en.wikipedia.org/wiki/Unholy_(Sam_Smith_and_Kim_Petras_song)"
    ]
),

(
    "Watermelon Sugar", "Harry Styles", "English", 0.9, 0.7, "Pop",
    [
        "https://www.youtube.com/watch?v=E07s5ZYygMg",
        "https://en.wikipedia.org/wiki/Watermelon_Sugar"
    ]
),

(
    "Peaches", "Justin Bieber ft. Daniel Caesar, Giveon", "English", 0.7, 0.5, "R&B",
    [
        "https://www.youtube.com/watch?v=tQ0yjYUFKAE",
        "https://en.wikipedia.org/wiki/Peaches_(Justin_Bieber_song)"
    ]
),

(
    "Industry Baby", "Lil Nas X, Jack Harlow", "English", 0.8, 0.9, "Hip-Hop/Pop",
    [
        "https://www.youtube.com/watch?v=UTHLKHL_whs",
        "https://en.wikipedia.org/wiki/Industry_Baby"
    ]
),

(
    "Bad Habits", "Ed Sheeran", "English", 0.7, 0.8, "Pop",
    [
        "https://www.youtube.com/watch?v=orJSJGHjBLI",
        "https://en.wikipedia.org/wiki/Bad_Habits_(Ed_Sheeran_song)"
    ]
),

(
    "abcdefu", "GAYLE", "English", 0.7, 0.9, "Pop/Punk",
    [
        "https://www.youtube.com/watch?v=NaFd8ucHLuo",
        "https://en.wikipedia.org/wiki/Abcdefu"
    ]
),

(
    "Sunflower", "Post Malone, Swae Lee", "English", 0.7, 0.6, "Hip-Hop/Pop",
    [
        "https://www.youtube.com/watch?v=ApXoWvfEYVU",
        "https://en.wikipedia.org/wiki/Sunflower_(Post_Malone_and_Swae_Lee_song)"
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
# FUNCTION: Download Audio (60-sec) with YouTube
# ===============================
def download_audio(url_list, filepath, duration=60, proxy=None):
    folder, base = os.path.split(filepath)
    base = safe_filename(base)
    full_path = os.path.join(folder, base)
    print(f"➡️ Downloading trimmed audio to: {full_path}")

    for url in url_list:
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
                'postprocessor_args': ['-t', str(duration)]  # limit duration in seconds
            }
            if proxy:
                ydl_opts['proxy'] = proxy
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"✅ 60-sec audio downloaded from {url}")
            return full_path
        except Exception as e:
            print(f"⚠️ Failed for {url}: {e}")
            continue
    return None

# ===============================
# MAIN LOOP
# ===============================
data = []
failed_songs = []

for song_title, artist, language, valence, arousal, genre, url_list in songs_info:
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
        downloaded_path = download_audio(url_list, audio_file, proxy=VPN_PROXY)
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
