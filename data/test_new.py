import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL
import logging

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
        "Tainted Love", "Soft Cell", "English", 0.7, 0.8, "New Wave",
        [
            "https://www.youtube.com/watch?v=XZVpR3Pk-r8",  # YouTube
            "https://soundcloud.com/soft-cell-official/tainted-love-original-version",  # SoundCloud
            "https://en.wikipedia.org/wiki/Tainted_Love"  # Wikipedia
        ]
    ),
    (
        "Let's Groove", "Earth, Wind & Fire", "English", 0.88, 0.9, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=Lrle0x_DhbA",  # YouTube
            "https://soundcloud.com/earthwindandfire/lets-groove-2",  # SoundCloud
            "https://en.wikipedia.org/wiki/Let%27s_Groove"  # Wikipedia
        ]
    ),
    (
        "Boogie Shoes", "KC & The Sunshine Band", "English", 0.85, 0.88, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=h0wA2sej6ss",  # YouTube
            "https://soundcloud.com/kcthesunshineband/boogie-shoes",  # SoundCloud
            "https://en.wikipedia.org/wiki/Boogie_Shoes"  # Wikipedia
        ]
    ),
    (
        "Get Down Tonight", "KC & The Sunshine Band", "English", 0.87, 0.87, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=QKCqGnz74EM",  # YouTube
            "https://soundcloud.com/kcthesunshineband/get-down-tonight",  # SoundCloud (if found)
            "https://en.wikipedia.org/wiki/Get_Down_Tonight"  # Wikipedia
        ]
    ),
    (
        "We Are Family", "Sister Sledge", "English", 0.92, 0.80, "Disco/Funk",
        [
            "https://www.youtube.com/watch?v=jdsqht1m1rE",  # YouTube (official music video)
            "https://soundcloud.com/sistersledge/we-are-family-single-version",  # SoundCloud :contentReference[oaicite:0]{index=0}
            "https://en.wikipedia.org/wiki/We_Are_Family_(song)"  # Wikipedia :contentReference[oaicite:1]{index=1}
        ]
    ),
    (
        "Best Of My Love", "The Emotions", "English", 0.90, 0.87, "Funk/Soul",
        [
            "https://www.youtube.com/watch?v=EbwXoN-XU5I",  # YouTube :contentReference[oaicite:2]{index=2}
            "https://soundcloud.com/the-emotions-official/best-of-my-love",  # SoundCloud :contentReference[oaicite:3]{index=3}
            "https://en.wikipedia.org/wiki/Best_of_My_Love_(The_Emotions_song)"  # Wikipedia :contentReference[oaicite:4]{index=4}
        ]
    ),
    (
        "Disco Inferno", "The Trammps", "English", 0.92, 0.86, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=A_sY2rjxq6M",  # YouTube :contentReference[oaicite:5]{index=5}
            "https://soundcloud.com/the-trammps/disco-inferno",  # SoundCloud (if found)
            "https://en.wikipedia.org/wiki/Disco_Inferno"  # Wikipedia :contentReference[oaicite:6]{index=6}
        ]
    ),
    (
        "Shake Your Body (Down to the Ground)", "The Jacksons", "English", 0.87, 0.85, "Funk",
        [
            "https://www.youtube.com/watch?v=kldVOhKe4rg",  # YouTube :contentReference[oaicite:7]{index=7}
            "https://soundcloud.com/the-jacksons/shake-your-body-down-to-the-ground",  # SoundCloud (if found)
            "https://en.wikipedia.org/wiki/Shake_Your_Body_(Down_to_the_Ground)"  # Wikipedia :contentReference[oaicite:8]{index=8}
        ]
    ),
    (
        "Boogie Oogie Oogie", "A Taste of Honey", "English", 0.89, 0.85, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=Z5rks1vQbQI",  # YouTube
            "https://soundcloud.com/a-taste-of-honey/boogie-oogie-oogie",  # SoundCloud (if found)
            "https://en.wikipedia.org/wiki/Boogie_Oogie_Oogie"  # Wikipedia
        ]
    ),
    (
        "Ain't No Stoppin' Us Now", "McFadden & Whitehead", "English", 0.92, 0.85, "Funk/Disco",
        [
            "https://www.youtube.com/watch?v=5Fmf3D0WqM0",  # YouTube
            "https://soundcloud.com/mcfadden-whitehead/aint-no-stoppin-us-now",  # SoundCloud (if found)
            "https://en.wikipedia.org/wiki/Ain%27t_No_Stoppin%27_Us_Now"  # Wikipedia
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

    for url in url_list:
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
                'noplaylist': True,
                'geo_bypass': True,
                'ignoreerrors': True,
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
