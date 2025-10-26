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

    ("Midnight City", "M83", "English", 0.8, 0.7, "Synth-pop", 
        ["https://www.youtube.com/watch?v=dX3k_QDnzHE",
         "https://m.soundcloud.com/m83/midnight-city",
         "https://en.wikipedia.org/wiki/Midnight_City"]
    ),
    ("Electric Feel", "MGMT", "English", 0.85, 0.75, "Electronic Rock", 
        ["https://www.youtube.com/watch?v=MmZexg8sxyk",
         "https://soundcloud.com/mgmt/electric-feel",
         "https://en.wikipedia.org/wiki/Electric_Feel"]
    ),
    ("Dog Days Are Over", "Florence + The Machine", "English", 0.9, 0.8, "Indie Rock", 
        ["https://www.youtube.com/watch?v=iWOyfLBYtuU",
         "https://soundcloud.com/florenceandthemachine/dog-days-are-over",
         "https://en.wikipedia.org/wiki/Dog_Days_Are_Over"]
    ),
    ("Kids", "MGMT", "English", 0.85, 0.75, "Indie Rock", 
        ["https://www.youtube.com/watch?v=b2cv8kcKBR8",
         "https://soundcloud.com/mgmt/kids",
         "https://en.wikipedia.org/wiki/Kids_(MGMT_song)"]
    ),
    ("Young Folks", "Peter Bjorn and John", "English", 0.8, 0.6, "Indie Pop", 
        ["https://www.youtube.com/watch?v=51V1VMkuyx0",
         "https://soundcloud.com/peterbjornandjohn/young-folks",
         "https://en.wikipedia.org/wiki/Young_Folks"]
    ),
    ("Pumped Up Kicks", "Foster the People", "English", 0.75, 0.65, "Indie Pop", 
        ["https://www.youtube.com/watch?v=SDTZ7iX4vTQ",
         "https://soundcloud.com/fosterthepeople/pumped-up-kicks",
         "https://en.wikipedia.org/wiki/Pumped_Up_Kicks"]
    ),
    ("Take a Chance on Me", "ABBA", "English", 0.85, 0.7, "Disco", 
        ["https://www.youtube.com/watch?v=pFru1e72UKI",
         "https://soundcloud.com/abba-official/take-a-chance-on-me",
         "https://en.wikipedia.org/wiki/Take_a_Chance_on_Me"]
    ),
    ("Chasing Cars", "Snow Patrol", "English", 0.75, 0.6, "Alternative Rock", 
        ["https://www.youtube.com/watch?v=GemKqzILV4w",
         "https://soundcloud.com/snowpatrol/chasing-cars",
         "https://en.wikipedia.org/wiki/Chasing_Cars"]
    ),
    ("Somebody That I Used to Know", "Gotye feat. Kimbra", "English", 0.3, 0.4, "Indie Pop", 
        ["https://www.youtube.com/watch?v=8UVNT4wvIGY",
         "https://soundcloud.com/gotye/somebody-that-i-used-to-know",
         "https://en.wikipedia.org/wiki/Somebody_That_I_Used_to_Know"]
    ),
    ("Ho Hey", "The Lumineers", "English", 0.7, 0.5, "Folk Rock", 
        ["https://www.youtube.com/watch?v=zvCBSSwgtg4",
         "https://soundcloud.com/thelumineers/ho-hey",
         "https://en.wikipedia.org/wiki/Ho_Hey"]
    ),
    ("Sex on Fire", "Kings of Leon", "English", 0.85, 0.8, "Alternative Rock", 
        ["https://www.youtube.com/watch?v=RF0HhrwIwp0",
         "https://soundcloud.com/kings-of-leon/sex-on-fire",
         "https://en.wikipedia.org/wiki/Sex_on_Fire"]
    ),
    ("Electric Lady", "Janelle Monáe", "English", 0.9, 0.75, "R&B", 
        ["https://www.youtube.com/watch?v=DJ8wYz7cuSk",
         "https://soundcloud.com/janellemonae/electric-lady",
         "https://en.wikipedia.org/wiki/Electric_Lady"]
    ),
    ("Heaven", "Avicii", "English", 0.85, 0.7, "Electronic", 
        ["https://www.youtube.com/watch?v=9ajv8AHMku4",
         "https://soundcloud.com/avicii/heaven",
         "https://en.wikipedia.org/wiki/Heaven_(Avicii_song)"]
    ),
    ("I Bet You Look Good on the Dancefloor", "Arctic Monkeys", "English", 0.8, 0.9, "Indie Rock", 
        ["https://www.youtube.com/watch?v=pK7egZaT3hs",
         "https://soundcloud.com/arcticmonkeys/i-bet-you-look-good-on-the-dancefloor",
         "https://en.wikipedia.org/wiki/I_Bet_You_Look_Good_on_the_Dancefloor"]
    ),
    ("Radioactive", "Imagine Dragons", "English", 0.6, 0.9, "Rock", 
        ["https://www.youtube.com/watch?v=ktvTqknDobU",
         "https://soundcloud.com/imagine-dragons/radioactive",
         "https://en.wikipedia.org/wiki/Radioactive_(Imagine_Dragons_song)"]
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
