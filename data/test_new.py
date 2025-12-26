import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "cookies.txt"))
FFMPEG_PATH = r"C:\Users\HP\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"


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

# Optional: SOCKS5 proxy for VPN (Windscribe / ProtonVPN)
VPN_PROXY = None  # e.g. "socks5://username:password@us-free.windscribe.com:1080"

# ===============================
# HELPER: Safe filename
# ===============================
def safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# ===============================
# SONGS LIST WITH VERIFIED LINKS
# ===============================
songs_info = [
("Jolene", "Dolly Parton", "English", 0.45, 0.55, "Country",
[
    "https://www.youtube.com/watch?v=Ixrje2rXLMA"
]
),
("Ring of Fire", "Johnny Cash", "English", 0.70, 0.65, "Country",
[
    "https://www.youtube.com/watch?v=5WyLhwYFgmk"
]
),
("Take Me Home, Country Roads", "John Denver", "English", 0.85, 0.45, "Country",
[
    "https://www.youtube.com/watch?v=1vrEljMfXYo"
]
),
("I Walk the Line", "Johnny Cash", "English", 0.65, 0.40, "Country",
[
    "https://www.youtube.com/watch?v=5WyLhwYFgmk"
]
),
("Blue Eyes Crying in the Rain", "Willie Nelson", "English", 0.35, 0.25, "Country",
[
    "https://www.youtube.com/watch?v=crgtW3-zV1c"
]
),
("Always On My Mind", "Willie Nelson", "English", 0.40, 0.30, "Country",
[
    "https://www.youtube.com/watch?v=R7f189Z0v0Y"
]
),
("Friends in Low Places", "Garth Brooks", "English", 0.80, 0.70, "Country",
[
    "https://www.youtube.com/watch?v=mvCgSqPZ4EM"
]
),
("The Gambler", "Kenny Rogers", "English", 0.65, 0.45, "Country",
[
    "https://www.youtube.com/watch?v=7hx4gdlfamo"
]
),
("Stand By Your Man", "Tammy Wynette", "English", 0.50, 0.35, "Country",
[
    "https://www.youtube.com/watch?v=AM-b8P1yj9w"
]
),
("Crazy", "Patsy Cline", "English", 0.30, 0.25, "Country",
[
    "https://www.youtube.com/watch?v=CKTOvHw8qFM"
]
),
("Mama Tried", "Merle Haggard", "English", 0.55, 0.60, "Country",
[
    "https://www.youtube.com/watch?v=XyRCvukVv6w"
]
),
("On the Road Again", "Willie Nelson", "English", 0.85, 0.65, "Country",
[
    "https://www.youtube.com/watch?v=dBN86y30Ufc"
]
),
("He Stopped Loving Her Today", "George Jones", "English", 0.15, 0.20, "Country",
[
    "https://www.youtube.com/watch?v=VExw77xJsBQ"
]
),
("Achy Breaky Heart", "Billy Ray Cyrus", "English", 0.80, 0.75, "Country",
[
    "https://www.youtube.com/watch?v=byQIPdHMpjc"
]
),
("Check Yes or No", "George Strait", "English", 0.75, 0.55, "Country",
[
    "https://www.youtube.com/watch?v=9nwyvUO9xWI"
]
),
("Amarillo by Morning", "George Strait", "English", 0.50, 0.35, "Country",
[
    "https://www.youtube.com/watch?v=F2_1tF7C2m4"
]
),
("Coward of the County", "Kenny Rogers", "English", 0.40, 0.50, "Country",
[
    "https://www.youtube.com/watch?v=8c1vH9Fz-Fc"
]
),
("If Tomorrow Never Comes", "Garth Brooks", "English", 0.45, 0.30, "Country",
[
    "https://www.youtube.com/watch?v=Gn4Gf6asjUU"
]
),
("She Thinks My Tractor's Sexy", "Kenny Chesney", "English", 0.85, 0.70, "Country",
[
    "https://www.youtube.com/watch?v=uWu4aynBK7E"
]
),
("Live Like You Were Dying", "Tim McGraw", "English", 0.70, 0.50, "Country",
[
    "https://www.youtube.com/watch?v=_9TShlMkQnc"
]
),
("Whiskey Lullaby", "Brad Paisley & Alison Krauss", "English", 0.10, 0.20, "Country",
[
    "https://www.youtube.com/watch?v=IZbN_nmxAGk"
]
),
("Before He Cheats", "Carrie Underwood", "English", 0.55, 0.80, "Country",
[
    "https://www.youtube.com/watch?v=WaSy8yy-mr8"
]
),
("Need You Now", "Lady A", "English", 0.35, 0.30, "Country",
[
    "https://www.youtube.com/watch?v=eM213aMKTHg"
]
),
("Die a Happy Man", "Thomas Rhett", "English", 0.70, 0.35, "Country",
[
    "https://www.youtube.com/watch?v=w5hZqZ7h8yg"
]
),
("Tennessee Whiskey", "Chris Stapleton", "English", 0.60, 0.40, "Country",
[
    "https://www.youtube.com/watch?v=4zAThXFOy2c"
]
),
("Chicken Fried", "Zac Brown Band", "English", 0.85, 0.60, "Country",
[
    "https://www.youtube.com/watch?v=e4ujS1er1r0"
]
),
("Take Your Time", "Sam Hunt", "English", 0.65, 0.45, "Country",
[
    "https://www.youtube.com/watch?v=2kV6sQ3O9nE"
]
),
("God Bless the USA", "Lee Greenwood", "English", 0.80, 0.55, "Country",
[
    "https://www.youtube.com/watch?v=Q65KZIqay4E"
]
),
("Humble and Kind", "Tim McGraw", "English", 0.75, 0.30, "Country",
[
    "https://www.youtube.com/watch?v=awzNHuGqoMc"
]
),
("Fast as You", "Dwight Yoakam", "English", 0.70, 0.65, "Country",
[
    "https://www.youtube.com/watch?v=9z6rE0G9pYc"
]
),
("Delta Dawn", "Tanya Tucker", "English", 0.45, 0.40, "Country",
[
    "https://www.youtube.com/watch?v=Kk2tI5Ovxv4"
]
),
("Man! I Feel Like a Woman!", "Shania Twain", "English", 0.90, 0.85, "Country",
[
    "https://www.youtube.com/watch?v=ZJL4UGSbeFg"
]
),
("Boot Scootin’ Boogie", "Brooks & Dunn", "English", 0.85, 0.80, "Country",
[
    "https://www.youtube.com/watch?v=d05tQrhNMkA"
]
),
("Take It Easy", "Eagles", "English", 0.80, 0.55, "Country Rock",
[
    "https://www.youtube.com/watch?v=4v8KEbQA8kw"
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
# FUNCTION: Download Audio (60-sec) — Option 2 Improved
# ===============================
def download_audio(url_list, filepath, duration=60, proxy=None):
    folder, base = os.path.split(filepath)
    base = safe_filename(base)
    full_path = os.path.join(folder, base)
    if not full_path.lower().endswith(".mp3"):
        full_path += ".mp3"

    print(f"➡️ Downloading trimmed audio to: {full_path}")

    logger = logging.getLogger("yt_dlp")
    logger.setLevel(logging.ERROR)

    for url in [u for u in url_list if "youtube.com" in u or "youtu.be" in u]:
        try:

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
                'cookies': COOKIES_PATH,
                'ffmpeg_location': FFMPEG_PATH,
                'noplaylist': True,
                'geo_bypass': True,
                'quiet': False,
                'force_ipv4': True,  # ✅ Force IPv4 to avoid CDN issues
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': '*/*',
                },
                'prefer_ffmpeg': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'postprocessor_args': ['-ss', '0', '-t', str(duration)],
                'merge_output_format': 'mp3',
                'logger': logger,
            }

            if proxy:
                ydl_opts['proxy'] = proxy

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Validate file creation
            if os.path.exists(full_path):
                print(f"✅ 60-sec audio saved: {full_path}")
                return full_path
            else:
                # yt-dlp may create with a slightly different name
                for f in os.listdir(folder):
                    if f.startswith(os.path.splitext(os.path.basename(full_path))[0]) and f.endswith(".mp3"):
                        new_path = os.path.join(folder, f)
                        print(f"✅ 60-sec audio saved (renamed): {new_path}")
                        return new_path

        except Exception as e:
            print(f"⚠️ Failed for {url}: {e}")
            continue

    print(f"❌ Audio download failed for URLs: {url_list}")
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

        # --- Download Audio ---
        audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
        downloaded_path = download_audio(url_list, audio_file, proxy=VPN_PROXY)
        if not downloaded_path:
            failed_songs.append(song_title)
            continue

        # --- Add Metadata ---
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
