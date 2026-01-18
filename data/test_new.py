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
(
 "Embraceable You", "Ella Fitzgerald", "English", 0.36, 0.30, "Jazz Vocal",
 ["https://www.youtube.com/watch?v=RF1yQMPMEMo&pp=ygUfRWxsYSBGaXR6Z2VyYWxkIEVtYnJhY2VhYmxlIFlvdQ%3D%3D"]
),

(
 "You Go to My Head", "Billie Holiday", "English", 0.33, 0.27, "Jazz Ballad",
 ["https://www.youtube.com/watch?v=V7xQd3yt590&pp=ygUgQmlsbGllIEhvbGlkYXkgWW91IEdvIHRvIE15IEhlYWQ%3D"]
),

(
 "When Sunny Gets Blue", "Nancy Wilson", "English", 0.35, 0.29, "Jazz Ballad",
 ["https://www.youtube.com/watch?v=vVJyqIiEHto&pp=ygUhTmFuY3kgV2lsc29uIFdoZW4gU3VubnkgR2V0cyBCbHVl"]
),

(
 "Almost Like Being in Love", "Ella Fitzgerald", "English", 0.70, 0.65, "Swing Jazz",
 ["https://www.youtube.com/watch?v=1sTe80kHhyw&pp=ygUpRWxsYSBGaXR6Z2VyYWxkIEFsbW9zdCBMaWtlIEJlaW5nIGluIExvdmU%3D"]
),

(
 "Detour Ahead", "Sarah Vaughan", "English", 0.30, 0.25, "Jazz Ballad",
 ["https://www.youtube.com/watch?v=1Hyps3oh9tc&pp=ygUaU2FyYWggVmF1Z2hhbiBEZXRvdXIgQWhlYWQ%3D"]
),

(
 "I'm Glad There Is You", "Julie London", "English", 0.32, 0.26, "Cool Jazz",
 ["https://www.youtube.com/watch?v=qcwHFY_6PyE&pp=ygUiSnVsaWUgTG9uZG9uIEknbSBHbGFkIFRoZXJlIElzIFlvdQ%3D%3D"]
),

(
 "East of the Sun", "Frank Sinatra", "English", 0.38, 0.32, "Jazz Vocal",
 ["https://www.youtube.com/watch?v=1CvZqCNyEBc&pp=ygUdRnJhbmsgU2luYXRyYSBFYXN0IG9mIHRoZSBTdW4%3D"]
),

(
 "Black Coffee", "Peggy Lee", "English", 0.29, 0.23, "Jazz Ballad",
 ["https://www.youtube.com/watch?v=_udW-G6VEhg&pp=ygUWUGVnZ3kgTGVlIEJsYWNrIENvZmZlZdIHCQlPCgGHKiGM7w%3D%3D"]
),

(
 "But Not for Me", "Chet Baker", "English", 0.34, 0.28, "Jazz Vocal",
 ["https://www.youtube.com/watch?v=QwAwtMt8t4s&pp=ygUZQ2hldCBCYWtlciBCdXQgTm90IGZvciBNZQ%3D%3D"]
),

(
 "You're My Thrill", "Billie Holiday", "English", 0.31, 0.25, "Jazz Ballad",
 ["https://www.youtube.com/watch?v=tZJoVPFoYPM&pp=ygUfQmlsbGllIEhvbGlkYXkgWW91J3JlIE15IFRocmlsbA%3D%3D"]
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
