import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL
import logging
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "cookies.txt"))
FFMPEG_PATH = r"C:\Users\HP\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"
#  r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
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


("Alright Now", "Freeway", "English", 0.66, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=3rhYvrssbJo"]),
("What We Do", "Freeway", "English", 0.63, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=1swlUtEkXZ0"]),
("Roc the Mic", "Freeway", "English", 0.68, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=XKr1mrpBwIM"]),
("Flashing Lights", "Kanye West", "English", 0.58, 0.74, "Hip-Hop", ["https://www.youtube.com/watch?v=ila-hAUXR5U"]),
("Good Morning", "Kanye West", "English", 0.82, 0.72, "Hip-Hop", ["https://www.youtube.com/watch?v=6CHs4x2uqcQ"]),
("Homecoming", "Kanye West", "English", 0.85, 0.7, "Hip-Hop", ["https://www.youtube.com/watch?v=LQ488QrqGE4"]),
("Champion", "Kanye West", "English", 0.9, 0.83, "Hip-Hop", ["https://www.youtube.com/watch?v=FoLE8FmooIM"]),
("Spaceship", "Kanye West", "English", 0.55, 0.68, "Hip-Hop", ["https://www.youtube.com/watch?v=wGM6N0qXeu4"]),
("Jesus Walks", "Kanye West", "English", 0.6, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=MYF7H_fpc-g"]),
("Diamonds from Sierra Leone", "Kanye West", "English", 0.52, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=92FCRmggNqQ"]),
("Drive Slow", "Kanye West", "English", 0.57, 0.6, "Hip-Hop", ["https://www.youtube.com/watch?v=xvAFTiP3vzk"]),
("Touch the Sky Remix", "Kanye West", "English", 0.88, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=YkwQbuAGLj4"]),
("Crack Music", "Kanye West", "English", 0.5, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=2tmPSK-w90o"]),
("We Major", "Kanye West", "English", 0.8, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=L-otjVPGEQg"]),
("Heard 'Em Say", "Kanye West", "English", 0.7, 0.64, "Hip-Hop", ["https://www.youtube.com/watch?v=elVF7oG0pQs"]),
("Late", "Kanye West", "English", 0.76, 0.69, "Hip-Hop", ["https://www.youtube.com/watch?v=V9QN2YFSKM4"]),
("Bring the Pain", "Method Man", "English", 0.58, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=T0BlXy3Roj4"]),
("Da Rockwilder", "Method Man", "English", 0.64, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=WCYy8jpp7R8"]),
("All I Need", "Method Man", "English", 0.82, 0.75, "Hip-Hop", ["https://www.youtube.com/watch?v=XW1HNWqdVbk"]),
("Judgement Day", "Method Man", "English", 0.54, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=5F4kTIFsg_M"]),
("Ice Cream", "Raekwon", "English", 0.84, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=jgh10of6DKA"]),
("Incarcerated Scarfaces", "Raekwon", "English", 0.48, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=1ZYau0hJHFk"]),
("Verbal Intercourse", "Raekwon", "English", 0.5, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=9ZX1pk7rL8k"]),
("Heaven & Hell", "Raekwon", "English", 0.46, 0.85, "Hip-Hop", ["https://www.youtube.com/watch?v=fF4frZmtJ1U"]),
("Daytona 500", "Ghostface Killah", "English", 0.55, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=5ZGi2lyQJQs"]),
("Apollo Kids", "Ghostface Killah", "English", 0.68, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=S0bYHTApml0"]),
("Cherchez LaGhost", "Ghostface Killah", "English", 0.78, 0.77, "Hip-Hop", ["https://www.youtube.com/watch?v=eIdPfbwNV3w"]),
("Run", "Ghostface Killah", "English", 0.6, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=eMSY3zfLxRA"]),
("Above the Clouds", "Gang Starr", "English", 0.58, 0.83, "Hip-Hop", ["https://www.youtube.com/watch?v=N_4O5iodyOA"]),
("Mass Appeal", "Gang Starr", "English", 0.7, 0.74, "Hip-Hop", ["https://www.youtube.com/watch?v=y9lNbNGbo24"]),
("Full Clip", "Gang Starr", "English", 0.66, 0.87, "Hip-Hop", ["https://www.youtube.com/watch?v=qmj1q67NDAk"]),
("DWYCK", "Gang Starr", "English", 0.72, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=TgelVkHEKdw"]),
("Work", "Gang Starr", "English", 0.68, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=GSszWXkDHa8"]),
("Above the Rim", "2Pac", "English", 0.6, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=Gcxwei78rTI"]),
("I Get Around", "2Pac", "English", 0.88, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=YqJAnQTwmJs"]),
("Hail Mary", "2Pac", "English", 0.44, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=vw-O1JdXcXk"]),
("Keep Ya Head Up", "2Pac", "English", 0.76, 0.69, "Hip-Hop", ["https://www.youtube.com/watch?v=SHVzWMFMH6Y"]),
("So Many Tears", "2Pac", "English", 0.42, 0.73, "Hip-Hop", ["https://www.youtube.com/watch?v=1Z52-lIZMbQ"]),
("Picture Me Rollin'", "2Pac", "English", 0.66, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=YcmnPvZjk_4"]),
("Life Goes On", "2Pac", "English", 0.74, 0.66, "Hip-Hop", ["https://www.youtube.com/watch?v=KrHmzJI8z3M"]),
("Brenda's Got a Baby", "2Pac", "English", 0.36, 0.58, "Hip-Hop", ["https://www.youtube.com/watch?v=NRWUs0KtB-I"]),
("Temptations", "2Pac", "English", 0.7, 0.79, "Hip-Hop", ["https://www.youtube.com/watch?v=skg0w8DpEe4"]),
("How Do U Want It", "2Pac", "English", 0.86, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=PJWpyKvHZAk"]),
("Me Against the World", "2Pac", "English", 0.5, 0.74, "Hip-Hop", ["https://www.youtube.com/watch?v=QlPYub-gTjE"]),
("Ambitionz Remix", "2Pac", "English", 0.52, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=6r4abhWnW-Y"]),
("I Ain't Mad at Cha", "2Pac", "English", 0.78, 0.69, "Hip-Hop", ["https://www.youtube.com/watch?v=oXkKURgzYVY"]),
("Pour Out a Little Liquor", "2Pac", "English", 0.58, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=C0EBY9_XnFE"]),
("Heartz of Men", "2Pac", "English", 0.46, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=8k2GywV1QC8"]),
("Troublesome '96", "2Pac", "English", 0.5, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=KUQmABnX964"]),
("Can't C Me", "2Pac", "English", 0.64, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=Pz5F0Ju97r0"]),
("Skandalouz", "2Pac", "English", 0.72, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=-2bGyHt7sg8"]),
("Shorty Wanna Ride", "Young Buck", "English", 0.7, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=GsyJVv-9ygM"]),
("Let Me In", "Young Buck", "English", 0.68, 0.85, "Hip-Hop", ["https://www.youtube.com/watch?v=4AFjbiv-kHA"]),
("Stunt 101", "G-Unit", "English", 0.84, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=c6qk1AFH9Y4"]),
("Wanna Get to Know You", "G-Unit", "English", 0.82, 0.75, "Hip-Hop", ["https://www.youtube.com/watch?v=YF7TvXu4lSY"]),
("Smile", "G-Unit", "English", 0.76, 0.7, "Hip-Hop", ["https://www.youtube.com/watch?v=1CHNkaCn_zY"]),
("Poppin Them Thangs", "G-Unit", "English", 0.78, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=lc0zKB88XPM"]),
("Outta Control", "50 Cent", "English", 0.8, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=Z3Oux1lN__4"]),
("Just a Lil Bit", "50 Cent", "English", 0.83, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=GllEDACUbNo"]),
("Window Shopper", "50 Cent", "English", 0.86, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=bFLow5StvvU"]),
("Hate It or Love It", "The Game", "English", 0.8, 0.75, "Hip-Hop", ["https://www.youtube.com/watch?v=BuMBmK5uksg"]),
("How We Do", "The Game", "English", 0.86, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=PH34kMOjmQk"]),
("Dreams", "The Game", "English", 0.68, 0.69, "Hip-Hop", ["https://www.youtube.com/watch?v=2K0q74jtV8s"]),
("Westside Story", "The Game", "English", 0.72, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=E4yqZdzDixk"]),
("Put You on the Game", "The Game", "English", 0.6, 0.85, "Hip-Hop", ["https://www.youtube.com/watch?v=3OYojGxshoI"]),
("Wouldn't Get Far", "The Game", "English", 0.78, 0.83, "Hip-Hop", ["https://www.youtube.com/watch?v=MlzrC-B6n-M"]),
("Let's Ride", "The Game", "English", 0.82, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=tDkmhZIQ9jo"]),
("Church for Thugs", "The Game", "English", 0.58, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=0p5CC8t7Eg0"]),
("One Blood", "The Game", "English", 0.64, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=q0ZUbFVgZpc"]),
("Ali Bomaye", "The Game", "English", 0.52, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=eU4ZvfkmOck"]),
("Red Nation", "The Game", "English", 0.66, 0.93, "Hip-Hop", ["https://www.youtube.com/watch?v=jSAwWrbdoEQ"]),
("El Chapo", "The Game", "English", 0.62, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=RviOwY0OKyE"]),
("Hate It or Love It Remix", "The Game", "English", 0.82, 0.79, "Hip-Hop", ["https://www.youtube.com/watch?v=BuMBmK5uksg"]),



]





# ===============================
# INITIALIZE GENIUS API
# ===============================
genius = lyricsgenius.Genius(GENIUS_TOKEN, timeout=15, retries=3)

# ===============================
# LYRICS MULTI-SOURCE ENGINE
# ===============================

def clean_lyrics(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


# -------------------------------
# SOURCE 1 — Genius
# -------------------------------
def fetch_from_genius(song_title, artist):

    search_queries = [
        f"{song_title} {artist}",
        f"{song_title} lyrics",
        f"{song_title} bandish",
        f"{song_title} hindustani",
        song_title
    ]

    for query in search_queries:
        try:
            print(f"Trying Genius search: {query}")
            song = genius.search_song(query)
            if song and song.lyrics:
                return clean_lyrics(song.lyrics)
        except:
            continue

    return None



# -------------------------------
# SOURCE 2 — HindiLyrics.net
# BEST for Hindi/classical/bhajan
# -------------------------------
def fetch_from_hindilyrics(song_title):
    try:
        query = song_title.replace(" ", "-").lower()
        url = f"https://hindilyrics.net/{query}/"

        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_div = soup.find("div", class_="lyrics")

        if lyrics_div:
            return clean_lyrics(lyrics_div.get_text())
    except:
        pass
    return None


# -------------------------------
# SOURCE 3 — LyricsMint
# -------------------------------
def fetch_from_lyricsmint(song_title):
    try:
        query = song_title.replace(" ", "-").lower()
        url = f"https://www.lyricsmint.com/{query}"

        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(page.text, "html.parser")
        div = soup.find("div", class_="entry-content")

        if div:
            return clean_lyrics(div.get_text())
    except:
        pass
    return None




# -------------------------------
# MASTER LYRICS FUNCTION
# -------------------------------
def fetch_lyrics(song_title, artist):

    print(f"🔎 Searching lyrics for {song_title}...")

    # 1️⃣ Genius
    
    lyrics = fetch_from_genius(song_title, artist)
    if lyrics:
        print("✅ Genius")
        return lyrics

    # 2️⃣ HindiLyrics
    lyrics = fetch_from_hindilyrics(song_title)
    if lyrics:
        print("✅ HindiLyrics")
        return lyrics

    # 3️⃣ LyricsMint
    lyrics = fetch_from_lyricsmint(song_title)
    if lyrics:
        print("✅ LyricsMint")
        return lyrics

    print("❌ Lyrics not found anywhere")
    return None


    # lyrics = fetch_from_azlyrics(song_title, artist_name)
    # if lyrics:
    #     print("✅ Found on AZLyrics")
    #     return lyrics

    # print("❌ Lyrics not found anywhere")
    # return None

# ===============================
# FUNCTION: Fetch Lyrics
# ===============================
# def fetch_lyrics(song_title, artist_name):
#     try:
#         song = genius.search_song(song_title, artist_name)
#         if song and song.lyrics:
#             return song.lyrics
#     except Exception as e:
#         print(f"❌ Error fetching lyrics for {song_title}: {e}")
#     return None

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
                'force_ipv4': True,
                'retries': 10,
                'fragment_retries': 10,

                # 🔥 Use Node runtime
                'js_runtimes': {'node': {}},

                # 🔥 Android client avoids signature/403 issues
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android']
                    }
                },

                'postprocessor_args': ['-ss', '0', '-t', str(duration)],
                # 🔥 Trim first 60 sec
                # 'download_sections': f"*0-{duration}",

                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }],

                'logger': logger,
            }

            if proxy:
                ydl_opts['proxy'] = proxy

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # locate generated mp3
            for f in os.listdir(folder):
                if f.startswith(os.path.splitext(os.path.basename(full_path))[0]) and f.endswith(".mp3"):
                    final_path = os.path.join(folder, f)
                    print(f"✅ Audio saved: {final_path}")
                    return final_path

        except Exception as e:
            print(f"⚠️ Failed for {url}: {e}")
            continue

    print(f"❌ Audio download failed for URLs: {url_list}")
    return None


# def download_audio(url_list, filepath, duration=60, proxy=None):
#     folder, base = os.path.split(filepath)
#     base = safe_filename(base)
#     full_path = os.path.join(folder, base)
#     if not full_path.lower().endswith(".mp3"):
#         full_path += ".mp3"

#     print(f"➡️ Downloading trimmed audio to: {full_path}")

#     logger = logging.getLogger("yt_dlp")
#     logger.setLevel(logging.ERROR)

#     for url in [u for u in url_list if "youtube.com" in u or "youtu.be" in u]:
#         try:

#             ydl_opts = {
#                 'format': 'bestaudio[ext=m4a]/bestaudio',
#                 'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
#                 'cookies': COOKIES_PATH,
#                 'ffmpeg_location': FFMPEG_PATH,
#                 'noplaylist'    : True,
#                 'geo_bypass': True,
#                 'force_ipv4': True,
#                 'retries': 10,
#                 'fragment_retries': 10,

#                 # 🔥 makes y  t-dlp use Node runtime
#                 'js_runtimes': ['node'],

#                 # 🔥 bypass YouTube signature blocking
#                 'extractor_args': {
#                     'youtube': {
#                         'player_client': ['android']
#                     }
#                 },

#                 # trim first 60 sec
#                 'download_sections': f"*0-{duration}",

#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': 'mp3',
#                     'preferredquality': '192'
#                 }],
#             }


#             # ydl_opts = {
#             #     'format': 'bestaudio/best',
#             #     'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
#             #     'cookies': COOKIES_PATH,
#             #     'ffmpeg_location': FFMPEG_PATH,
#             #     'noplaylist': True,
#             #     'geo_bypass': True,
#             #     'quiet': False,
#             #     'force_ipv4': True,  # ✅ Force IPv4 to avoid CDN issues
#             #     'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
#             #     'http_headers': {
#             #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
#             #         'Accept-Language': 'en-US,en;q=0.9',
#             #         'Accept': '*/*',
#             #     },
#             #     'prefer_ffmpeg': True,
#             #     'postprocessors': [{
#             #         'key': 'FFmpegExtractAudio',
#             #         'preferredcodec': 'mp3',
#             #         'preferredquality': '192',
#             #     }],
#             #     'postprocessor_args': ['-ss', '0', '-t', str(duration)],
#             #     'merge_output_format': 'mp3',
#             #     'logger': logger,
#             # }

#             if proxy:
#                 ydl_opts['proxy'] = proxy

#             with YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])

#             # Validate file creation
#             if os.path.exists(full_path):
#                 print(f"✅ 60-sec audio saved: {full_path}")
#                 return full_path
#             else:
#                 # yt-dlp may create with a slightly different name
#                 for f in os.listdir(folder):
#                     if f.startswith(os.path.splitext(os.path.basename(full_path))[0]) and f.endswith(".mp3"):
#                         new_path = os.path.join(folder, f)
#                         print(f"✅ 60-sec audio saved (renamed): {new_path}")
#                         return new_path

#         except Exception as e:
#             print(f"⚠️ Failed for {url}: {e}")
#             continue

#     print(f"❌ Audio download failed for URLs: {url_list}")
#     return None

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
