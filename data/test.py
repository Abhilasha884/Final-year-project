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
    """Remove invalid characters from filenames (Windows-safe)."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

# ===============================
# SONGS LIST WITH YOUTUBE URLS
# ===============================
songs_info = [

    # --- ENGLISH SONGS ---
    ("Blinding Lights", "The Weeknd", "English", 0.9, 0.8, "Synthpop", "https://www.youtube.com/watch?v=fHI8X4OXluQ"),
    ("Someone You Loved", "Lewis Capaldi", "English", 0.4, 0.3, "Pop Ballad", "https://www.youtube.com/watch?v=zABLecsR5UE"),
    ("Memories", "Maroon 5", "English", 0.6, 0.4, "Pop", "https://www.youtube.com/watch?v=SlPhetQn5pk"),
    ("Watermelon Sugar", "Harry Styles", "English", 0.9, 0.7, "Pop", "https://www.youtube.com/watch?v=E07s5ZYEXw4"),
    ("Dynamite", "BTS", "English", 0.9, 0.8, "K-Pop", "https://www.youtube.com/watch?v=gdZLi9oWNZg"),
    ("Dance Monkey", "Tones and I", "English", 0.85, 0.8, "Electropop", "https://www.youtube.com/watch?v=q0hyYWKXF0Q"),
    ("Bad Guy", "Billie Eilish", "English", 0.6, 0.7, "Pop", "https://www.youtube.com/watch?v=DyDfgMOUjCI"),
    ("Lovely", "Billie Eilish, Khalid", "English", 0.2, 0.25, "Indie Pop", "https://www.youtube.com/watch?v=V1Pl8CzNzCw"),
    ("Old Town Road", "Lil Nas X ft. Billy Ray Cyrus", "English", 0.8, 0.8, "Country Rap", "https://www.youtube.com/watch?v=w2Ov5jzm3j8"),
    ("Señorita", "Shawn Mendes, Camila Cabello", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=Pkh8UtuejGw"),
    ("Goosebumps", "Travis Scott", "English", 0.6, 0.9, "Hip-Hop", "https://www.youtube.com/watch?v=_EyZUTDAH0U"),
    ("Save Your Tears", "The Weeknd", "English", 0.5, 0.4, "Synthpop", "https://www.youtube.com/watch?v=u6lihZAcy4s"),
    ("Mood", "24kGoldn ft. Iann Dior", "English", 0.7, 0.75, "Hip-Hop", "https://www.youtube.com/watch?v=GrAchTdepsU"),
    ("Therefore I Am", "Billie Eilish", "English", 0.5, 0.6, "Pop", "https://www.youtube.com/watch?v=RUQl6YcMalg"),
    ("Positions", "Ariana Grande", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=tcYodQoapMg"),
    ("Willow", "Taylor Swift", "English", 0.6, 0.4, "Indie Pop", "https://www.youtube.com/watch?v=RsEZmictANA"),
    ("Leave The Door Open", "Silk Sonic", "English", 0.8, 0.5, "R&B", "https://www.youtube.com/watch?v=adLGH_gtjmA"),
    ("Kiss Me More", "Doja Cat ft. SZA", "English", 0.8, 0.7, "Pop/R&B", "https://www.youtube.com/watch?v=0EVVKs6DQLo"),
    ("Montero (Call Me By Your Name)", "Lil Nas X", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=6swmTBVI8Dk"),

    # --- HINDI SONGS ---
    ("Dil Bechara", "A.R. Rahman", "Hindi", 0.4, 0.3, "Romantic", "https://www.youtube.com/watch?v=GODAlxW5Pes"),
    ("Makhna", "Tanishk Bagchi, Yasser Desai, Asees Kaur", "Hindi", 0.9, 0.8, "Dance", "https://www.youtube.com/watch?v=0jU5Jd7TLcY"),
    ("Ghungroo", "Arijit Singh, Shilpa Rao", "Hindi", 0.9, 0.8, "Dance", "https://www.youtube.com/watch?v=7SHohKbEKog"),
    ("Duniyaa", "Akhil, Dhvani Bhanushali", "Hindi", 0.85, 0.5, "Romantic", "https://www.youtube.com/watch?v=7-QVFY0T6cc"),
    ("Hawayein", "Arijit Singh, Pritam", "Hindi", 0.8, 0.4, "Romantic", "https://www.youtube.com/watch?v=1tVL11ULjYY"),
    ("Tera Fitoor", "Arijit Singh", "Hindi", 0.85, 0.6, "Romantic", "https://www.youtube.com/watch?v=3mMX3NAi7gM"),
    ("Laal Ishq", "Arijit Singh", "Hindi", 0.7, 0.3, "Romantic", "https://www.youtube.com/watch?v=3gy7vVt3YxY"),
    ("Pachtaoge", "Arijit Singh", "Hindi", 0.2, 0.4, "Sad", "https://www.youtube.com/watch?v=PQmrmVs10X8"),
    ("Chedkhaniyan", "Arijit Singh, Nikhita Gandhi", "Hindi", 0.7, 0.6, "Fun", "https://www.youtube.com/watch?v=840WJ5Q8f6Q"),
    ("Main Tumhara", "Jonita Gandhi, Hriday Gattani", "Hindi", 0.6, 0.4, "Romantic", "https://www.youtube.com/watch?v=Z3vWD-7oduo"),
    ("Tujhe Kitna Chahne Lage", "Arijit Singh", "Hindi", 0.7, 0.4, "Romantic", "https://www.youtube.com/watch?v=POamd_-8zO0"),
    ("Namo Namo", "Amit Trivedi", "Hindi", 0.6, 0.5, "Spiritual", "https://www.youtube.com/watch?v=yE0Y8JYB9gs"),
    ("Shayad", "Arijit Singh", "Hindi", 0.85, 0.5, "Romantic", "https://www.youtube.com/watch?v=Jq_sh1yHkY4"),
    ("O Saki Saki", "Neha Kakkar, Tulsi Kumar, B Praak", "Hindi", 0.85, 0.9, "Dance", "https://www.youtube.com/watch?v=PVxc5mIHVuQ"),
    ("Garmi", "Badshah, Neha Kakkar", "Hindi", 0.9, 0.9, "Dance", "https://www.youtube.com/watch?v=v4nxMRA0z8g"),
    ("Teri Mitti", "B Praak", "Hindi", 0.3, 0.5, "Patriotic", "https://www.youtube.com/watch?v=B3oLqDoyW8w"),
    ("Ve Maahi", "Arijit Singh, Asees Kaur", "Hindi", 0.85, 0.5, "Romantic", "https://www.youtube.com/watch?v=YO4h9HhkV0Y"),
    ("Pal Pal Dil Ke Paas", "Arijit Singh, Parampara Thakur", "Hindi", 0.8, 0.5, "Romantic", "https://www.youtube.com/watch?v=ABF8-DxJcA0"),
    ("Mere Sohneya", "Sachet Tandon, Parampara Thakur", "Hindi", 0.85, 0.6, "Romantic", "https://www.youtube.com/watch?v=M9mO8bL5s7o"),
    ("Chashni", "Vishal & Shekhar, Abhijeet Srivastava", "Hindi", 0.75, 0.4, "Romantic", "https://www.youtube.com/watch?v=O9n2k4ooMhw"),
    ("The Jawaani Song", "Vishal Dadlani, Payal Dev", "Hindi", 0.9, 0.8, "Dance", "https://www.youtube.com/watch?v=BKQGthZP-nc"),
    ("Nachde Ne Saare", "Jasleen Royal, Harshdeep Kaur, Siddharth Mahadevan", "Hindi", 0.9, 0.8, "Dance", "https://www.youtube.com/watch?v=7M-b0bA0r8A"),


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
# FUNCTION: Download Audio (60-sec)
# ===============================
def download_audio(youtube_url, filepath, duration=60):
    folder, base = os.path.split(filepath)
    base = safe_filename(base)
    full_path = os.path.join(folder, base)

    print(f"➡️ Downloading trimmed audio to: {full_path}")

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': full_path,
            'quiet': True,
            'noplaylist': True,
            'geo_bypass': True,
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
            ydl.download([youtube_url])
        print(f"✅ 60-sec audio downloaded: {full_path}")
        return full_path
    except Exception as e:
        print(f"❌ Error downloading {youtube_url}: {e}")
        return None

# ===============================
# MAIN LOOP
# ===============================
data = []
failed_songs = []

for song_title, artist, language, valence, arousal, genre, youtube_url in songs_info:
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
        downloaded_path = download_audio(youtube_url, audio_file)
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
