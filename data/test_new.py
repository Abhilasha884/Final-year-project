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

    # 💖 Emotional Ballads
    # ("Because You Loved Me", "Celine Dion", "English", 0.55, 0.5, "Pop Ballad", ["https://www.youtube.com/watch?v=2GP8JuBtcis", "https://soundcloud.com/liveconcerts/because-you-loved-me-celine-dion"]),
    
  
    
    # ("Un-break My Heart", "Toni Braxton", "English", 0.35, 0.45, "R&B Ballad",["https://www.youtube.com/watch?v=p2Rch6WvPJE", "https://soundcloud.com/tonibraxton/un-break-my-heart"]),
    
    # ("You’re Beautiful", "James Blunt", "English", 0.5, 0.4, "Pop Ballad",["https://www.youtube.com/watch?v=oofSnsGkops", "https://soundcloud.com/jamesblunt/youre-beautiful"]),
     # ("Don’t Stop Believin’", "Journey", "English", 0.9, 0.85, "Rock", ["https://www.youtube.com/watch?v=1k8craCGpgs", "https://soundcloud.com/journey/dont-stop-believin"]),
    # ("Viva La Vida", "Coldplay", "English", 0.8, 0.6, "Pop Rock", ["https://www.youtube.com/watch?v=dvgZkm1xWPE", "https://soundcloud.com/coldplay/viva-la-vida"]),

    # ("It’s My Life", "Bon Jovi", "English", 0.8, 0.85, "Pop Rock", ["https://www.youtube.com/watch?v=vx2u5uUu3DE", "https://soundcloud.com/bonjovi/its-my-life"]),
     # ("Viva La Vida", "Coldplay", "English", 0.80, 0.60, "Pop Rock", ["https://www.youtube.com/watch?v=dvgZkm1xWPE", "https://open.spotify.com/track/4Pz1rH5p0YRSe2UghTqK9T"]),
     # ("You’re Beautiful", "James Blunt", "English", 0.50, 0.40, "Pop Ballad", ["https://www.youtube.com/watch?v=oofSnsGkops", "https://open.spotify.com/track/3S5UaWOxbDnSdQy1FLG70U"]),
     # ("It’s My Life", "Bon Jovi", "English", 0.80, 0.85, "Pop Rock", ["https://www.youtube.com/watch?v=vx2u5uUu3DE", "https://soundcloud.com/bonjovi/its-my-life"]),

    # ("Complicated", "Avril Lavigne", "English", 0.75, 0.70, "Pop Rock", ["https://www.youtube.com/watch?v=5NPBIwQyPWE", "https://soundcloud.com/avrillavigne/complicated"]),



    # 💖 Emotional Ballads
    ("Because You Loved Me", "Céline Dion", "English", 0.55, 0.50, "Pop Ballad",
     ["https://www.youtube.com/watch?v=fpl4if07ics",
      "https://open.spotify.com/track/2zKcnzs0CYsTIh5Q5E9y6E"]),

    ("I Will Always Love You", "Whitney Houston", "English", 0.30, 0.40, "Soul Ballad",
     ["https://www.youtube.com/watch?v=3JWTaaS7LdU",
      "https://open.spotify.com/track/4eHbdreAnSOrDDsFfc4Fpm"]),

    ("Un-break My Heart", "Toni Braxton", "English", 0.35, 0.45, "R&B Ballad",
     ["https://www.youtube.com/watch?v=p2Rch6WvPJE",
      "https://open.spotify.com/track/4Jdoz7kYFKEcZlafhf5G45"]),

    # 🎉 Feel-good / Funky Pop
    ("Mambo No.5", "Lou Bega", "English", 0.95, 0.90, "Latin Pop",
     ["https://www.youtube.com/watch?v=EK_LN3XEcnw",
      "https://soundcloud.com/lou-bega/mambo-no-5"]),

    ("Livin’ la Vida Loca", "Ricky Martin", "English", 0.90, 0.90, "Pop",
     ["https://www.youtube.com/watch?v=p47fEXGabaY",
      "https://open.spotify.com/track/7nSnVutpVd9E2eF9zravG6"]),

    ("Can’t Get You Out of My Head", "Kylie Minogue", "English", 0.85, 0.80, "Dance Pop",
     ["https://www.youtube.com/watch?v=c18441Eh_WE&utm_source=chatgpt.com",
      "https://soundcloud.com/kylieminogue/cant-get-you-out-of-my-head"]),

    ("Hey Ya!", "Outkast", "English", 0.95, 0.85, "Funk Pop",
     ["https://www.youtube.com/watch?v=PWgvGjAhvIw",
      "https://soundcloud.com/outkast/hey-ya"]),

    ("Crazy in Love", "Beyoncé ft. Jay-Z", "English", 0.90, 0.90, "R&B Pop",
     ["https://www.youtube.com/watch?v=ViwtNLUqkMY",
      "https://open.spotify.com/track/3S5UaWOxbDnSdQy1FLG70U"]),

    # 🎸 Pop Rock / Feel-good Rock
    ("Smooth", "Santana ft. Rob Thomas", "English", 0.90, 0.85, "Latin Rock",
     ["https://www.youtube.com/watch?v=6Whgn_iE5uc",
      "https://soundcloud.com/santana-official/smooth"]),

    ("Drops of Jupiter", "Train", "English", 0.70, 0.60, "Pop Rock",
     ["https://www.youtube.com/watch?v=6vEme-BwLl0",
      "https://soundcloud.com/train/drops-of-jupiter"]),

    ("It’s My Life", "Bon Jovi", "English", 0.85, 0.75, "Rock",
     ["https://www.youtube.com/watch?v=vx2u5uUu3DE",
      "https://soundcloud.com/bonjovi/its-my-life"]),

    ("Complicated", "Avril Lavigne", "English", 0.70, 0.65, "Pop Rock",
     ["https://www.youtube.com/watch?v=5NPBIwQyPWE",
      "https://soundcloud.com/avrillavigne/complicated"]),

    # 💃 Dance / Disco Revival
    ("Hung Up", "Madonna", "English", 0.90, 0.85, "Dance Pop",
     ["https://www.youtube.com/watch?v=EDwb9jOVRtU",
      "https://soundcloud.com/madonna/hung-up"]),

    ("Don’t Stop Movin’", "S Club 7", "English", 0.90, 0.80, "Dance Pop",
     ["https://www.youtube.com/watch?v=5s9Xl5v7VnI",
      "https://soundcloud.com/sclub7/dont-stop-movin"]),

    ("Rock Your Body", "Justin Timberlake", "English", 0.85, 0.90, "Funk Pop",
     ["https://www.youtube.com/watch?v=TSVHoHyErBQ",
      "https://soundcloud.com/justintimberlake/rock-your-body"]),

    ("Can’t Get Enough of You Baby", "Smash Mouth", "English", 0.85, 0.75, "Pop Rock",
     ["https://www.youtube.com/watch?v=16gT5YeZ-jA&utm_source=chatgpt.com",
      "https://open.spotify.com/track/0UVIXwqfTjefBbDyY60MWB?utm_source=chatgpt.com"]),

    # 🎶 Classic Rock / Timeless
    ("Don’t Stop Believin’", "Journey", "English", 0.80, 0.70, "Classic Rock",
     ["https://www.youtube.com/watch?v=1k8craCGpgs",
      "https://soundcloud.com/journeyband/dont-stop-believin"]),

    ("Viva La Vida", "Coldplay", "English", 0.85, 0.80, "Pop Rock",
     ["https://www.youtube.com/watch?v=dvgZkm1xWPE",
      "https://soundcloud.com/coldplay/viva-la-vida"])


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
