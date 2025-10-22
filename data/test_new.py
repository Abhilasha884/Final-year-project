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
    return re.sub(r'[\\/*?:"<>|]', "", name)

# ===============================
# SONGS LIST WITH MULTIPLE URLS (YouTube / SoundCloud / Bandcamp)
# ===============================
songs_info = [

    ("Pehla Nasha", "Udit Narayan & Sadhana Sargam", "Hindi", 0.85, 0.55, "Romantic", 
     ["https://www.youtube.com/watch?v=Qm5DwmH7u8E",
      "https://en.wikipedia.org/wiki/Pehla_Nasha"]),
    ("Chaiyya Chaiyya", "Sukhwinder Singh & Sapna Agarwal", "Hindi", 0.9, 0.8, "Dance/Pop", 
     ["https://www.youtube.com/watch?v=PQm-w_Q5vZI",
      "https://en.wikipedia.org/wiki/Chaiyya_Chaiyya"]),
    ("Kuch Kuch Hota Hai", "Udit Narayan & Alka Yagnik", "Hindi", 0.8, 0.65, "Bollywood Pop/Romantic", 
     ["https://www.youtube.com/watch?v=VL1K4aV4aHk",
      "https://en.wikipedia.org/wiki/Kuch_Kuch_Hota_Hai_(song)"]),
    ("Dil To Pagal Hai", "Lata Mangeshkar & Udit Narayan", "Hindi", 0.75, 0.6, "Romantic/Dance", 
     ["https://www.youtube.com/watch?v=T_UCfYjc-Dw",
      "https://en.wikipedia.org/wiki/Dil_To_Pagal_Hai_(song)"])


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
# FUNCTION: Download Audio (60-sec) with multi-platform support
# ===============================
def download_audio(url_list, filepath, duration=60):
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
                'postprocessor_args': [
                    '-t', str(duration)  # limit duration in seconds
                ]
            }
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
        downloaded_path = download_audio(url_list, audio_file)
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
