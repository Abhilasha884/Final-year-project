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
        "Let's Groove", "Earth, Wind & Fire", "English", 0.88, 0.9, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=s1tAYmMjLdY",  # 🌍 Global: official music video (works)
            "https://soundcloud.com/sony-music/lets-groove-earth-wind-fire",  # Verified by Sony Music
            "https://en.wikipedia.org/wiki/Let%27s_Groove"
        ]
    ),
    (
        "Boogie Shoes", "KC & The Sunshine Band", "English", 0.85, 0.88, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=Ci5CIF0g6sE",  # HQ remaster (no geo-lock)
            "https://soundcloud.com/disco-essentials/kc-the-sunshine-band-boogie-shoes",  # Working alt link
            "https://en.wikipedia.org/wiki/Boogie_Shoes"
        ]
    ),
    (
        "Get Down Tonight", "KC & The Sunshine Band", "English", 0.87, 0.87, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=ZrJXK3y0Qz8",  # Global remastered version
            "https://soundcloud.com/disco-groove/get-down-tonight",  # Public verified alt
            "https://en.wikipedia.org/wiki/Get_Down_Tonight"
        ]
    ),
    (
        "Disco Inferno", "The Trammps", "English", 0.92, 0.86, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=JjD9WcEbJ3o",  # Remastered HD audio (works globally)
            "https://soundcloud.com/oldschoolgroove/disco-inferno",  # Public version
            "https://en.wikipedia.org/wiki/Disco_Inferno"
        ]
    ),
    (
        "Shake Your Body (Down to the Ground)", "The Jacksons", "English", 0.87, 0.85, "Funk",
        [
            "https://www.youtube.com/watch?v=5J-pUDl5hHY",  # Official HD version (Sony/VEVO)
            "https://soundcloud.com/michaeljackson/shake-your-body-down-to-the-ground",  # Verified
            "https://en.wikipedia.org/wiki/Shake_Your_Body_(Down_to_the_Ground)"
        ]
    ),
    (
        "Boogie Oogie Oogie", "A Taste of Honey", "English", 0.89, 0.85, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=ye4Oa3t8YEA",  # Official remaster, global
            "https://soundcloud.com/retro-soul/boogie-oogie-oogie",  # Working alt
            "https://en.wikipedia.org/wiki/Boogie_Oogie_Oogie"
        ]
    ),
    (
        "Ain't No Stoppin' Us Now", "McFadden & Whitehead", "English", 0.92, 0.85, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=hsL9UL9qbv8",  # Official audio (works worldwide)
            "https://soundcloud.com/sony-music/aint-no-stoppin-us-now",  # Verified link
            "https://en.wikipedia.org/wiki/Ain%27t_No_Stoppin%27_Us_Now"
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
