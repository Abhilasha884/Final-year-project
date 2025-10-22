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


    ("Tum Hi Ho", "Arijit Singh", "Hindi", 0.6, 0.4, "Romantic",
     [
         "https://www.youtube.com/watch?v=Umqb9KENgmk",
         "https://soundcloud.com/arijit-singh/tum-hi-ho",
         "https://en.wikipedia.org/wiki/Tum_Hi_Ho"
     ]
    ),
    ("Kesariya", "Arijit Singh", "Hindi", 0.8, 0.6, "Bollywood Pop",
     [
         "https://www.youtube.com/watch?v=BddP6PYo2gs",
         "https://soundcloud.com/arijit-singh/kesariya",
         "https://en.wikipedia.org/wiki/Kesariya_(song)"
     ]
    ),
    ("Apna Bana Le", "Arijit Singh", "Hindi", 0.75, 0.55, "Romantic",
     [
         "https://www.youtube.com/watch?v=E0PbIbS5A3c",
         "https://soundcloud.com/arijit-singh/apna-bana-le",
         "https://en.wikipedia.org/wiki/Apna_Bana_Le"
     ]
    ),
    ("Ghungroo", "Arijit Singh, Shilpa Rao", "Hindi", 0.85, 0.7, "Dance Pop",
     [
         "https://www.youtube.com/watch?v=E0PbIbS5A3c",
         "https://soundcloud.com/arijit-singh/ghungroo",
         "https://en.wikipedia.org/wiki/Ghungroo_(song)"
     ]
    ),
    ("Shayad", "Arijit Singh", "Hindi", 0.7, 0.5, "Romantic",
     [
         "https://www.youtube.com/watch?v=Jx6Gnu0WbJ4",
         "https://soundcloud.com/arijit-singh/shayad",
         "https://en.wikipedia.org/wiki/Shayad_(song)"
     ]
    ),
    ("Tujhe Kitna Chahne Lage", "Arijit Singh", "Hindi", 0.65, 0.45, "Sad Romantic",
     [
         "https://www.youtube.com/watch?v=AgX2IIvzdSk",
         "https://soundcloud.com/arijit-singh/tujhe-kitna-chahne-lage",
         "https://en.wikipedia.org/wiki/Tujhe_Kitna_Chahne_Lage"
     ]
    ),
    ("Bekhayali", "Sachet Tandon", "Hindi", 0.4, 0.6, "Emotional Rock",
     [
         "https://www.youtube.com/watch?v=Vzv8cXqK1kA",
         "https://soundcloud.com/sachet-tandon/bekhayali",
         "https://en.wikipedia.org/wiki/Bekhayali"
     ]
    ),
    ("Jai Jai Shivshankar", "Vishal Dadlani, Benny Dayal", "Hindi", 0.9, 0.85, "Dance",
     [
         "https://www.youtube.com/watch?v=rP7XQH-8Eqw",
         "https://soundcloud.com/vishal-dadlani/jai-jai-shivshankar",
         "https://en.wikipedia.org/wiki/Jai_Jai_Sivshankar"
     ]
    ),
    ("Chogada", "Darshan Raval", "Hindi", 0.85, 0.9, "Garba",
     [
         "https://www.youtube.com/watch?v=I6fcpfgHbjo",
         "https://soundcloud.com/darshan-raval/chogada",
         "https://en.wikipedia.org/wiki/Chogada_(song)"
     ]
    ),
    ("Malang Title Track", "Ved Sharma", "Hindi", 0.7, 0.75, "Rock",
     [
         "https://www.youtube.com/watch?v=ek1ePFp-nBI",
         "https://soundcloud.com/ved-sharma/malang-title-track",
         "https://en.wikipedia.org/wiki/Malang_(song)"
     ]
    ),
    ("Senorita", "Farhan Akhtar, Hrithik Roshan, Abhay Deol", "Hindi", 0.9, 0.8, "Latin Pop",
     [
         "https://www.youtube.com/watch?v=wAiwXMJ704Y",
         "https://soundcloud.com/farhan-akhtar/senorita",
         "https://en.wikipedia.org/wiki/Senorita_(song)"
     ]
    ),
    ("Kabira", "Tochi Raina, Rekha Bhardwaj", "Hindi", 0.7, 0.55, "Sufi Pop",
     [
         "https://www.youtube.com/watch?v=jHNNMj5bNQw",
         "https://soundcloud.com/tochi-raina/kabira",
         "https://en.wikipedia.org/wiki/Kabira_(song)"
     ]
    ),
    ("Dil Diyan Gallan", "Atif Aslam", "Hindi", 0.75, 0.6, "Romantic",
     [
         "https://www.youtube.com/watch?v=SaCheA6Njc4",
         "https://soundcloud.com/atif-aslam/dil-diyan-gallan",
         "https://en.wikipedia.org/wiki/Dil_Diyan_Gallan"
     ]
    ),
    ("Kala Chashma", "Amar Arshi, Badshah, Neha Kakkar", "Hindi", 0.9, 0.85, "Dance Pop",
     [
         "https://www.youtube.com/watch?v=k4yXQkG2s1E",
         "https://soundcloud.com/amar-arshi/kala-chashma",
         "https://en.wikipedia.org/wiki/Kala_Chashma"
     ]
    ),
    ("Tareefan", "Badshah", "Hindi", 0.85, 0.75, "Hip Hop",
     [
         "https://www.youtube.com/watch?v=FsHZy2Zl1cQ",
         "https://soundcloud.com/badshah/tareefan",
         "https://en.wikipedia.org/wiki/Tareefan"
     ]
    ),
    ("Dilbar", "Neha Kakkar, Dhvani Bhanushali", "Hindi", 0.8, 0.8, "Dance Pop",
     [
         "https://www.youtube.com/watch?v=JFtij9uYx9s",
         "https://soundcloud.com/neha-kakkar/dilbar",
         "https://en.wikipedia.org/wiki/Dilbar_(song)"
     ]
    ),
    ("Agar Tum Saath Ho", "Alka Yagnik, Arijit Singh", "Hindi", 0.5, 0.4, "Sad Romantic",
     [
         "https://www.youtube.com/watch?v=sK7riqg2mr4",
         "https://soundcloud.com/alka-yagnik/agar-tum-saath-ho",
         "https://en.wikipedia.org/wiki/Agar_Tum_Saath_Ho"
     ]
    ),
    ("Rata Lambiyaan", "Jubin Nautiyal, Asees Kaur", "Hindi", 0.75, 0.65, "Romantic",
     [
         "https://www.youtube.com/watch?v=xoq5sYv0pJc",
         "https://soundcloud.com/jubin-nautiyal/rata-lambiyaan",
         "https://en.wikipedia.org/wiki/Rata_Lambiyaan"
     ]
    ),
    ("Duniya", "Akhil, Dhvani Bhanushali", "Hindi", 0.8, 0.6, "Bollywood Pop",
     [
         "https://www.youtube.com/watch?v=gvyUuxdRdR4",
         "https://soundcloud.com/akhil/duniya",
         "https://en.wikipedia.org/wiki/Duniya_(song)"
     ]
    ),
    ("Balam Pichkari", "Vishal Dadlani, Shalmali Kholgade", "Hindi", 0.9, 0.9, "Dance",
     [
         "https://www.youtube.com/watch?v=3rqqMlhzfn8",
         "https://soundcloud.com/vishal-dadlani/balam-pichkari",
         "https://en.wikipedia.org/wiki/Balam_Pichkari"
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
