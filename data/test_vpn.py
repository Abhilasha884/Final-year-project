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



    ("Light My Fire", "The Doors", "English", 0.85, 0.9, "Psychedelic Rock",
     [
         "https://open.spotify.com/track/6wBPS4f8YsM2TwZKdE68Hk",
         "https://music.apple.com/us/album/light-my-fire/160003879",
         "https://music.youtube.com/watch?v=deB_u-to-IE"
     ]
    ),
    ("Respect Yourself", "The Staple Singers", "English", 0.75, 0.6, "Soul",
     [
         "https://open.spotify.com/track/4y142Le7YaWaq8meNKTwzr",
         "https://music.apple.com/us/album/respect-yourself/1427888540",
         "https://music.youtube.com/watch?v=8Oipm2WHcZ8"
     ]
    ),
    ("Valerie", "Amy Winehouse", "English", 0.8, 0.65, "Soul/Pop",
     [
         "https://open.spotify.com/track/4AjCzRfXhw7Ffr9fScTIGT",
         "https://music.apple.com/us/album/valerie/887367604",
         "https://music.youtube.com/watch?v=4HLY1NTe04M"
     ]
    ),
    ("Clint Eastwood", "Gorillaz", "English", 0.7, 0.6, "Alternative Hip Hop",
     [
         "https://open.spotify.com/track/7xKdsLyu3PswQ7g1ppx0oQ",
         "https://music.apple.com/us/album/clint-eastwood/610253645",
         "https://music.youtube.com/watch?v=R6Csv9RBxb8"
     ]
    ),
    ("Breathe", "Pink Floyd", "English", 0.4, 0.45, "Progressive Rock",
     [
         "https://open.spotify.com/track/2SIwoFWhDmtz85XtxtBFsT",
         "https://music.apple.com/us/album/breathe/323079377",
         "https://music.youtube.com/watch?v=0pxP-T3V_wY"
     ]
    ),
    ("Hotline Bling", "Drake", "English", 0.75, 0.7, "Hip-Hop/R&B",
     [
         "https://open.spotify.com/track/3xKsf9qdS1CyvXSMEid6gA",
         "https://music.apple.com/us/album/hotline-bling/1146133985",
         "https://music.youtube.com/watch?v=uxpDa-c-4Mc"
     ]
    ),
    ("Royals", "Lorde", "English", 0.8, 0.6, "Alternative Pop",
     [
         "https://open.spotify.com/track/1W9qAR63VYwGQqkxd5z8nh",
         "https://music.apple.com/us/album/royals/1440833198",
         "https://music.youtube.com/watch?v=nlcIKh6sBtc"
     ]
    ),
    ("Juicy", "Notorious B.I.G.", "English", 0.75, 0.8, "Hip-Hop",
     [
         "https://open.spotify.com/track/4NkXGc8Es8oeNfD9VURmtH",
         "https://music.apple.com/us/album/juicy/206975892",
         "https://music.youtube.com/watch?v=_JZom_gVfuw"
     ]
    ),
    ("Riders on the Storm", "The Doors", "English", 0.6, 0.55, "Psychedelic Rock",
     [
         "https://open.spotify.com/track/4gkyFXrqVv9t3hvdNfS6TJ",
         "https://music.apple.com/us/album/riders-on-the-storm/1056775777",
         "https://music.youtube.com/watch?v=lS-af9Q-zvQ"
     ]
    ),
    ("Firework", "Katy Perry", "English", 0.9, 0.85, "Pop",
     [
         "https://open.spotify.com/track/6JmAs510HQZnq9kL2ZO61w",
         "https://music.apple.com/us/album/firework/420075073",
         "https://music.youtube.com/watch?v=QGJuMBdaqIw"
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
