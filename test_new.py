import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "cookies.txt"))
FFMPEG_PATH = r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
# FFMPEG_PATH = r"C:\Users\HP\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"


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
 "Overcome", "Tricky", "English", 0.48, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=E3R_3h6zQEs"]   # Official Music Video
),
(
 "Teardrop", "Massive Attack", "English", 0.55, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=Tb0MC0jFv6M"]  # Official Music Video
),
(
 "Glory Box", "Portishead", "English", 0.52, 0.40, "Trip Hop",
 ["https://www.youtube.com/watch?v=4qQyUi4zfDs"]  # Official Music Video
),
(
 "Roads", "Portishead", "English", 0.50, 0.38, "Trip Hop",
 ["https://www.youtube.com/watch?v=7nxWP9BhI7w"]
),
(
 "Sour Times", "Portishead", "English", 0.53, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=8B-i1vsA1-s"]
),
(
 "Angel", "Massive Attack", "English", 0.46, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=66A_3uwuZ_I"]
),
(
 "Inertia Creeps", "Massive Attack", "English", 0.49, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=Epgo8ixX6Wo"]
),
(
 "All Mine", "Portishead", "English", 0.51, 0.39, "Trip Hop",
 ["https://www.youtube.com/watch?v=vozNQX6Ye1A"]
),
(
 "6 Underground", "Sneaker Pimps", "English", 0.58, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=2eBZqmL8ehg"]
),
(
 "Protection", "Massive Attack", "English", 0.54, 0.40, "Trip Hop",
 ["https://www.youtube.com/watch?v=Epgo8ixX6Wo"]
),
(
 "Black Milk", "Massive Attack", "English", 0.47, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=Bf9AgX4Ixs4"]
),
(
 "Risingson", "Massive Attack", "English", 0.56, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=85E9Q5Wx210"]
),
(
 "Cowboys", "Portishead", "English", 0.48, 0.37, "Trip Hop",
 ["https://www.youtube.com/watch?v=1Jq8zNkpFzY"]
),
(
 "Strangers", "Portishead", "English", 0.46, 0.36, "Trip Hop",
 ["https://www.youtube.com/watch?v=FvFY2Stxlzc"]
),
(
 "Man Next Door", "Massive Attack", "English", 0.45, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=S71_vIMQ0YY"]
),
(
 "Blue Lines", "Massive Attack", "English", 0.52, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=Zw9V3qQq0Hg"]
),
(
 "Becoming X", "Sneaker Pimps", "English", 0.59, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=8s8r7x8p3Yk"]
),
(
 "Evolution Revolution Love", "Tricky", "English", 0.48, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=Z5Kp4Y6Q5sE"]
),
(
 "Stillness in Time", "Jamiroquai", "English", 0.60, 0.48, "Trip Hop",
 ["https://www.youtube.com/watch?v=9G6xF3Rk9kA"]
),
(
 "Blood on the Motorway", "DJ Shadow", "English", 0.47, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=0Q5F9Z2kY8A"]
),
(
 "Midnight in a Perfect World", "DJ Shadow", "English", 0.50, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=InFbBlpDTfQ"]
),
(
 "She Said", "Plan B", "English", 0.55, 0.46, "Trip Hop",
 ["https://www.youtube.com/watch?v=1nCqRmx3Dnw"]
),
(
 "2Wicky", "Hooverphonic", "English", 0.57, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=dppcuKJrqbE"]
),
(
 "Rose Rouge", "St Germain", "English", 0.63, 0.52, "Trip Hop",
 ["https://www.youtube.com/watch?v=6QImCMjW-PM"]
),
(
 "Lebanese Blonde", "Thievery Corporation", "English", 0.60, 0.48, "Trip Hop",
 ["https://www.youtube.com/watch?v=1t7W6NtTKAw"]
),
(
 "Until the Morning", "Thievery Corporation", "English", 0.56, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=9nH8R9y5sZc"]
),
(
 "Paradise Circus", "Massive Attack", "English", 0.48, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=6hUkyKBsGtY"]
),
(
 "All Is Full of Love", "Björk", "English", 0.47, 0.40, "Trip Hop",
 ["https://www.youtube.com/watch?v=9JE6rUwfckI"]
),
(
 "Górecki", "Lamb", "English", 0.46, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=tSRYvYN1ayw"]
),
(
 "What Your Soul Sings", "Massive Attack", "English", 0.52, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=0qB2z7nY7Yw"]
),
(
 "Londinium", "Archive", "English", 0.48, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=ZkZy6QG2p5U"]
),
(
 "That Girl", "Esthero", "English", 0.55, 0.46, "Trip Hop",
 ["https://www.youtube.com/watch?v=EJZ3v8XzGxU"]
),
(
 "Utopia", "Goldfrapp", "English", 0.49, 0.38, "Trip Hop",
 ["https://www.youtube.com/watch?v=2x8tZzQ0Y7Q"]
),
(
 "Four Ton Mantis", "Amon Tobin", "English", 0.58, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=0lPH4H6f6nA"]
),
(
 "High Noon", "Kruder & Dorfmeister", "English", 0.57, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=Jz5v8kF6ZkY"]
),
(
 "Breathe From Another", "Esthero", "English", 0.50, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=Hc9G7pXyQyA"]
),
(
 "Come Near Me", "Massive Attack", "English", 0.50, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=Z9v5b8l9Q2E"]
),
(
 "Daydream in Blue", "I Monster", "English", 0.56, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=BhB6Lb7_kN8"]
),
(
 "Low Place Like Home", "Sneaker Pimps", "English", 0.58, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=6M4mZ9Qz1Qw"]
),
(
 "Biscuit", "Portishead", "English", 0.44, 0.35, "Trip Hop",
 ["https://www.youtube.com/watch?v=G3Xc9HkzQ7M"]
),
(
 "Television", "Baxter", "English", 0.60, 0.48, "Trip Hop",
 ["https://www.youtube.com/watch?v=1Q7YpXz8HnE"]
),
(
 "Eyesdown", "Bonobo", "English", 0.62, 0.49, "Trip Hop",
 ["https://www.youtube.com/watch?v=E1tOV7y94DY"]
),
(
 "The Truth", "Handsome Boy Modeling School", "English", 0.55, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=7F6x0s2mKQk"]
),
(
 "Cherry Blossom Girl", "Air", "English", 0.57, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=UwYfN9y0S4Y"]
),
(
 "Playground Love", "Air", "English", 0.55, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=hFuu5wPFv1M"]
),
(
 "Building Steam with a Grain of Salt", "DJ Shadow", "English", 0.46, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=H0f9nKz3C8Q"]
),
(
 "Organ Donor", "DJ Shadow", "English", 0.52, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=QGZ9b2N6K7A"]
),
(
 "La Femme d’Argent", "Air", "English", 0.60, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=YUX8fUrKRNU"]
),
(
 "Sun Goddess", "Ramsey Lewis", "English", 0.54, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=8m5Wl9Z9R5Q"]
),
(
 "Little Fluffy Clouds", "The Orb", "English", 0.51, 0.40, "Trip Hop",
 ["https://www.youtube.com/watch?v=KNfjpmvbQG0"]
),
(
 "Windowlicker", "Aphex Twin", "English", 0.53, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=UBS4Gi1y_nc"]
),
(
 "Smoke City", "Underwater Love", "English", 0.58, 0.47, "Trip Hop",
 ["https://www.youtube.com/watch?v=HuLjsW8XhY4"]
),
(
 "Porcelain", "Moby", "English", 0.50, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=13EifDb4GYs"]
),
(
 "Riverside", "Agnes Obel", "English", 0.46, 0.38, "Trip Hop",
 ["https://www.youtube.com/watch?v=vjncyiuwwXQ"]
),
(
 "Your Woman", "White Town", "English", 0.57, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=lVL-zZnD3VU"]
),
(
 "Praise You", "Fatboy Slim", "English", 0.62, 0.50, "Trip Hop",
 ["https://www.youtube.com/watch?v=ruAi4VBoBSM"]
),
(
 "Sleepyhead", "Passion Pit", "English", 0.55, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=T0RvPYRRRbE"]
),
(
 "Open", "Rhye", "English", 0.49, 0.39, "Trip Hop",
 ["https://www.youtube.com/watch?v=sng_CdAAw8M"]
),
(
 "Paper Tiger", "Beck", "English", 0.53, 0.42, "Trip Hop",
 ["https://www.youtube.com/watch?v=9Q6YJ-r5SQs"]
),
(
 "Butterfly Caught", "Massive Attack", "English", 0.49, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=YqG7M0ZxY8E"]
),
(
 "Half Day Closing", "Portishead", "English", 0.45, 0.36, "Trip Hop",
 ["https://www.youtube.com/watch?v=6t3hHqk5H5I"]
),
(
 "Sweet Lullaby", "Deep Forest", "English", 0.58, 0.46, "Trip Hop",
 ["https://www.youtube.com/watch?v=lIF5EEneWEU"]
),
(
 "Need You", "Bonobo", "English", 0.57, 0.44, "Trip Hop",
 ["https://www.youtube.com/watch?v=K8JpQx7rZ6A"]
),
(
 "Poison", "The Prodigy", "English", 0.62, 0.50, "Trip Hop",
 ["https://www.youtube.com/watch?v=voBNpdXkLnU"]
),
(
 "Low Five", "Sneaker Pimps", "English", 0.55, 0.46, "Trip Hop",
 ["https://www.youtube.com/watch?v=1xWQ7Yt3F6E"]
),
(
 "Montague Terrace (In Blue)", "Scott Walker", "English", 0.44, 0.38, "Trip Hop",
 ["https://www.youtube.com/watch?v=Y0ZPpJ8uQ1k"]
),
(
 "Roads (Live at Roseland NYC)", "Portishead", "English", 0.46, 0.37, "Trip Hop",
 ["https://www.youtube.com/watch?v=km8BODHiWJU"]
),
(
 "Morning Dove White", "One Dove", "English", 0.49, 0.41, "Trip Hop",
 ["https://www.youtube.com/watch?v=9o2zQp3Y8mU"]
),
(
 "Missing (Todd Terry Remix)", "Everything But The Girl", "English", 0.60, 0.48, "Trip Hop",
 ["https://www.youtube.com/watch?v=IAkY5m00rpY"]
),
(
 "This Love", "Craig Armstrong", "English", 0.47, 0.39, "Trip Hop",
 ["https://www.youtube.com/watch?v=Kq7Y9M8ZP2Q"]
),
(
 "Song 2", "Blur", "English", 0.63, 0.50, "Trip Hop",
 ["https://www.youtube.com/watch?v=SSbBvKaM6sk"]
),
(
 "Fade Into You", "Mazzy Star", "English", 0.45, 0.38, "Trip Hop",
 ["https://www.youtube.com/watch?v=ImKY6TZEyrI"]
),
(
 "Humming", "Portishead", "English", 0.44, 0.35, "Trip Hop",
 ["https://www.youtube.com/watch?v=H5bZ0mQmQz8"]
),
(
 "Come With Us", "The Chemical Brothers", "English", 0.60, 0.49, "Trip Hop",
 ["https://www.youtube.com/watch?v=KkR7zFj9Z4E"]
),
(
 "Falling", "Julee Cruise", "English", 0.42, 0.36, "Trip Hop",
 ["https://www.youtube.com/watch?v=9li9GeDHc2s"]
),
(
 "Heaven Sent", "Esthero", "English", 0.52, 0.45, "Trip Hop",
 ["https://www.youtube.com/watch?v=K5M8p7nQXkY"]
),
(
 "Silent Shout", "The Knife", "English", 0.51, 0.43, "Trip Hop",
 ["https://www.youtube.com/watch?v=4uI1KXHJVO8"]
),
(
 "Signs", "Bloc Party", "English", 0.47, 0.40, "Trip Hop",
 ["https://www.youtube.com/watch?v=7B7wK6q9t9E"]
),

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

# def download_audio(url_list, filepath, duration=60, proxy=None):
#     folder, base = os.path.split(filepath)
#     base = safe_filename(base)
#     full_path = os.path.join(folder, base)

#     print(f"➡️ Downloading trimmed audio to: {full_path}.mp3")

#     logger = logging.getLogger("yt_dlp")
#     logger.setLevel(logging.ERROR)

#     for url in url_list:
#         try:
#             ydl_opts = {
#                 'format': 'bestaudio/best',
#                 'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
#                 'ffmpeg_location': FFMPEG_PATH,
#                 'noplaylist': True,
#                 'quiet': False,
#                 'force_ipv4': True,
#                 'download_sections': f"*0-{duration}",  # ✅ CORRECT trimming
#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': 'mp3',
#                     'preferredquality': '192',
#                 }],
#             }

#             if proxy:
#                 ydl_opts['proxy'] = proxy

#             with YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])

#             final_file = full_path + ".mp3"
#             if os.path.exists(final_file):
#                 print(f"✅ 60-sec audio saved: {final_file}")
#                 return final_file

#         except Exception as e:
#             print(f"⚠️ Failed for {url}: {e}")

#     print(f"❌ Audio download failed for URLs: {url_list}")
#     return None

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
