import os
import re
import pandas as pd
import lyricsgenius
import requests
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ===============================
# CONFIGURATION
# ===============================
GENIUS_TOKEN = "-nlF8VErk0OJ8kDZoKgJXAe4j78CSWZ2VLt_RydUHA2co1K8dmQ-om2THc9ugPHV"
dataset_folder = "data"
audio_folder = os.path.join(dataset_folder, "audio")
lyrics_folder = os.path.join(dataset_folder, "lyrics")
labels_csv = os.path.join(dataset_folder, "labels.csv")

os.makedirs(audio_folder, exist_ok=True)
os.makedirs(lyrics_folder, exist_ok=True)

# ===============================
# SONGS LIST WITH YOUTUBE URLS
# ===============================
# songs_info = [
#     ("Shape of You", "Ed Sheeran", "English", "Happy", 0.9, 0.8, "https://www.youtube.com/watch?v=JGwWNGJdvx8"),
#     ("Someone Like You", "Adele", "English", "Sad", 0.2, 0.3, "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),

    # ("Tere Bina", "A.R. Rahman, Chinmayi", "Hindi", 0.6, 0.7, "Romantic/Semi-classical", "https://www.youtube.com/watch?v=9JDSGhhiOwI"),
    # ("Pehla Nasha", "Udit Narayan, Sadhana Sargam", "Hindi", 0.7, 0.8, "Romantic/Classic", "https://www.youtube.com/watch?v=SBfPs-PMGTA"),

# ]


songs_info = [
    
    ("Cry_Baby", "Clean Bandit, Anne-Marie, David Guetta", "English", 0.8, 0.7, "EDM/Dance-Pop", "https://www.youtube.com/watch?v=P6OgpKqb-HA"),
    # ("Alone", "The Cure", "English", 0.9, 0.8, "Gothic Rock", "https://www.youtube.com/watch?v=sx9SVAtMkJM"),
    # ("My_Oh_My", "Kylie Minogue, Bebe Rexha, Tove Lo", "English", 0.7, 0.6, "Dance-Pop", "https://www.youtube.com/watch?v=h8rhLGhsa2M"),
    # ("Free", "Calvin Harris & Ellie Goulding", "English", 0.9, 0.8, "Dance-Pop", "https://www.youtube.com/watch?v=NAVv00ZbAdc"),
    # ("Peggy", "Ceechynaa", "English", 0.7, 0.6, "UK Hip-Hop", "https://www.youtube.com/watch?v=MFXs2M35yLo")

]


# ===============================
# INITIALIZE GENIUS API
# ===============================
genius = lyricsgenius.Genius(GENIUS_TOKEN, timeout=15, retries=3)

# ===============================
# FUNCTION: Fetch Lyrics
# ===============================
def fetch_lyrics(song_title, artist_name, language):
    try:
        song = genius.search_song(song_title, artist_name)
        if song and song.lyrics:
            return song.lyrics   # keep original, no transliteration
        else:
            return None
    except Exception as e:
        print(f"Error fetching {song_title}: {e}")
        return None



# ===============================
# FUNCTION: Download Audio
# ===============================
def download_audio(youtube_url, filename):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"Audio downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {youtube_url}: {e}")

# ===============================
# PROCESS SONGS
# ===============================
data = []

for song_title, artist, language, emotion, valence, arousal, youtube_url in songs_info:
    song_id = f"{language.lower()}_{song_title.replace(' ', '_')}"

   # Fetch lyrics
    lyrics = fetch_lyrics(song_title, artist, language)
    if lyrics:
        lyrics_file = os.path.join(lyrics_folder, f"{song_id}.txt")
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(lyrics)
        print(f"Lyrics saved for {song_title}")
    else:
        print(f"Lyrics not found for {song_title}")



    # Download audio
    audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
    download_audio(youtube_url, audio_file)
    
    # Add to dataset CSV
    data.append([song_id, language, artist, emotion, valence, arousal])

# Save labels.csv
df = pd.DataFrame(data, columns=["song_id", "language", "singer", "valence", "arousal", "genre"])
# Save labels.csv (append but avoid duplicates)
if os.path.exists(labels_csv):
    old_df = pd.read_csv(labels_csv)
    combined_df = pd.concat([old_df, df], ignore_index=True)
    combined_df.drop_duplicates(subset=["song_id"], keep="first", inplace=True)
    combined_df.to_csv(labels_csv, index=False)
else:
    df.to_csv(labels_csv, index=False)

# df.to_csv(labels_csv, index=False)
print("Dataset creation complete! Lyrics, audio, and labels (including singer) are ready.")
