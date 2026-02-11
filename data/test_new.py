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

("Hate Me Now", "Nas", "English", 0.64, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=dKSJN3WWR3E"]),

("If I Ruled the World", "Nas", "English", 0.78, 0.74, "Hip-Hop",
 ["https://www.youtube.com/watch?v=mlp-IIG9ApU"]),

("The Message", "Grandmaster Flash", "English", 0.60, 0.70, "Hip-Hop",
 ["https://www.youtube.com/watch?v=PobrSpMwKk4"]),

("Rapper's Delight", "The Sugarhill Gang", "English", 0.85, 0.76, "Hip-Hop",
 ["https://www.youtube.com/watch?v=rKTUAESacQM"]),

("Juicy", "The Notorious B.I.G.", "English", 0.82, 0.73, "Hip-Hop",
 ["https://www.youtube.com/watch?v=_JZom_gVfuw"]),

("California Love", "2Pac", "English", 0.88, 0.89, "Hip-Hop",
 ["https://www.youtube.com/watch?v=5wBTdfAkqGU"]),

("Changes", "2Pac", "English", 0.67, 0.65, "Hip-Hop",
 ["https://www.youtube.com/watch?v=eXvBjCO19QY"]),

("Dear Mama", "2Pac", "English", 0.72, 0.61, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Mb1ZvUDvLDY"]),

("Ambitionz Az a Ridah", "2Pac", "English", 0.58, 0.87, "Hip-Hop",
 ["https://www.youtube.com/watch?v=77nB_9uIcN4"]),

("Still D.R.E.", "Dr. Dre", "English", 0.74, 0.83, "Hip-Hop",
 ["https://www.youtube.com/watch?v=_CL6n0FJZpk"]),

("The Next Episode", "Dr. Dre", "English", 0.79, 0.85, "Hip-Hop",
 ["https://www.youtube.com/watch?v=QZXc39hT8t4"]),

("In Da Club", "50 Cent", "English", 0.86, 0.84, "Hip-Hop",
 ["https://www.youtube.com/watch?v=5qm8PH4xAss"]),

("Candy Shop", "50 Cent", "English", 0.80, 0.78, "Hip-Hop",
 ["https://www.youtube.com/watch?v=SRcnnId15BA"]),

("Many Men", "50 Cent", "English", 0.52, 0.69, "Hip-Hop",
 ["https://www.youtube.com/watch?v=5D3crqpClPY"]),

("P.I.M.P.", "50 Cent", "English", 0.77, 0.73, "Hip-Hop",
 ["https://www.youtube.com/watch?v=UDApZhXTpH8"]),



 ("Protect Ya Neck", "Wu-Tang Clan", "English", 0.63, 0.88, "Hip-Hop",
 ["https://www.youtube.com/watch?v=R0IUR4gkPIE"]),

("Triumph", "Wu-Tang Clan", "English", 0.66, 0.91, "Hip-Hop",
 ["https://www.youtube.com/watch?v=cPRKsKwEdUQ"]),

("Grindin'", "Clipse", "English", 0.69, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=TjWAWcx4xdE"]),

("Drop It Like It's Hot", "Snoop Dogg", "English", 0.78, 0.77, "Hip-Hop",
 ["https://www.youtube.com/watch?v=GtUVQei3nX4"]),

("Gin and Juice", "Snoop Dogg", "English", 0.82, 0.73, "Hip-Hop",
 ["https://www.youtube.com/watch?v=fWCZse1iwE0"]),

("Beautiful", "Snoop Dogg", "English", 0.84, 0.69, "Hip-Hop",
 ["https://www.youtube.com/watch?v=_FE194VN6c4"]),

("Passin' Me By", "The Pharcyde", "English", 0.74, 0.66, "Hip-Hop",
 ["https://www.youtube.com/watch?v=a-mAK3uB2_0"]),

("Scenario", "A Tribe Called Quest", "English", 0.76, 0.85, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Q6TLWqn82J4"]),

("Electric Relaxation", "A Tribe Called Quest", "English", 0.71, 0.67, "Hip-Hop",
 ["https://www.youtube.com/watch?v=WHRnvjCkTsw"]),

("Award Tour", "A Tribe Called Quest", "English", 0.79, 0.78, "Hip-Hop",
 ["https://www.youtube.com/watch?v=P800UWoE9xs"]),

("Rosa Parks", "OutKast", "English", 0.77, 0.80, "Hip-Hop",
 ["https://www.youtube.com/watch?v=drsQLEU0N1Y"]),

("ATLiens", "OutKast", "English", 0.58, 0.74, "Hip-Hop",
 ["https://www.youtube.com/watch?v=N7rE7k1FQ1A"]),

("So Fresh, So Clean", "OutKast", "English", 0.83, 0.76, "Hip-Hop",
 ["https://www.youtube.com/watch?v=-JfEJq56IwI"]),

("International Players Anthem", "UGK", "English", 0.75, 0.72, "Hip-Hop",
 ["https://www.youtube.com/watch?v=awMIbA34MT8"]),

("Still Tippin'", "Mike Jones", "English", 0.73, 0.84, "Hip-Hop",
 ["https://www.youtube.com/watch?v=7ZAYzLjQAd4"]),



 ("Make Em Say Uhh!", "Master P", "English", 0.74, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=h1fBYUWxaKQ"]),

("Back That Azz Up", "Juvenile", "English", 0.81, 0.88, "Hip-Hop",
 ["https://www.youtube.com/watch?v=9GvB9ySUJ3A"]),

("Ha", "Juvenile", "English", 0.68, 0.83, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Ww9VlmXKYgs"]),

("Still Fly", "Big Tymers", "English", 0.84, 0.82, "Hip-Hop",
 ["https://www.youtube.com/watch?v=9iCd6UHR-3I"]),

("Get Your Roll On", "Big Tymers", "English", 0.79, 0.80, "Hip-Hop",
 ["https://www.youtube.com/watch?v=9fJ3sZ0M8xA"]),

("Bling Bling", "B.G.", "English", 0.77, 0.79, "Hip-Hop",
 ["https://www.youtube.com/watch?v=3GwjfUFyY6M"]),

("What Happened to That Boy", "Birdman", "English", 0.70, 0.84, "Hip-Hop",
 ["https://www.youtube.com/watch?v=7o8zN8lWn-0"]),

("Number One Stunna", "Big Tymers", "English", 0.76, 0.81, "Hip-Hop",
 ["https://www.youtube.com/watch?v=2X_2IdybTV0"]),

("Slow Motion", "Juvenile", "English", 0.72, 0.68, "Hip-Hop",
 ["https://www.youtube.com/watch?v=JWeJHN5P-E8"]),

("Ridin'", "Chamillionaire", "English", 0.66, 0.83, "Hip-Hop",
 ["https://www.youtube.com/watch?v=CtwJvgPJ9xw"]),

("Turn My Swag On", "Soulja Boy", "English", 0.85, 0.78, "Hip-Hop",
 ["https://www.youtube.com/watch?v=R7yfISlGLNU"]),

("Crank That (Soulja Boy)", "Soulja Boy", "English", 0.91, 0.90, "Hip-Hop",
 ["https://www.youtube.com/watch?v=8UFIYGkROII"]),

("Teach Me How to Dougie", "Cali Swag District", "English", 0.88, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=4Vys9GZq0mE"]),

("Rack City", "Tyga", "English", 0.74, 0.81, "Hip-Hop",
 ["https://www.youtube.com/watch?v=AE3yia1AJeQ"]),

("A Milli", "Lil Wayne", "English", 0.63, 0.89, "Hip-Hop",
 ["https://www.youtube.com/watch?v=1Vf4mMCpNY0"]),


 ("Go DJ", "Lil Wayne", "English", 0.72, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=7u1aZqLq4Jk"]),

("Fireman", "Lil Wayne", "English", 0.70, 0.88, "Hip-Hop",
 ["https://www.youtube.com/watch?v=7q2xZQ7Q3BY"]),

("Lollipop", "Lil Wayne", "English", 0.83, 0.82, "Hip-Hop",
 ["https://www.youtube.com/watch?v=2IH8tNQAzSs"]),

("6 Foot 7 Foot", "Lil Wayne", "English", 0.66, 0.90, "Hip-Hop",
 ["https://www.youtube.com/watch?v=c7tOAGY59uQ"]),

("How to Love", "Lil Wayne", "English", 0.64, 0.58, "Hip-Hop",
 ["https://www.youtube.com/watch?v=y8Gf4-eT3w0"]),

("Mrs. Officer", "Lil Wayne", "English", 0.78, 0.73, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Jf1Vh8G5FJQ"]),

("Every Girl", "Young Money", "English", 0.75, 0.84, "Hip-Hop",
 ["https://www.youtube.com/watch?v=1n7qfJ7bZQg"]),

("BedRock", "Young Money", "English", 0.80, 0.76, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Ha80ZaecGkQ"]),

("Right Above It", "Lil Wayne", "English", 0.79, 0.88, "Hip-Hop",
 ["https://www.youtube.com/watch?v=CHZtMNbrmWE"]),

("Look at Me Now", "Chris Brown", "English", 0.82, 0.91, "Hip-Hop",
 ["https://www.youtube.com/watch?v=8gyLR4NfMiI"]),

("No Hands", "Waka Flocka Flame", "English", 0.84, 0.89, "Hip-Hop",
 ["https://www.youtube.com/watch?v=skhxizRYxps"]),

("Hard in da Paint", "Waka Flocka Flame", "English", 0.61, 0.92, "Hip-Hop",
 ["https://www.youtube.com/watch?v=WkkC9cK8Hz0"]),

("Grove St. Party", "Waka Flocka Flame", "English", 0.86, 0.90, "Hip-Hop",
 ["https://www.youtube.com/watch?v=ZKXGkB5ZuKk"]),

("Round of Applause", "Waka Flocka Flame", "English", 0.80, 0.88, "Hip-Hop",
 ["https://www.youtube.com/watch?v=ALsHox5sYCk"]),

("O Let's Do It", "Waka Flocka Flame", "English", 0.74, 0.87, "Hip-Hop",
 ["https://www.youtube.com/watch?v=4gK8J1d2Z7A"]),


 ("U Don't Know", "Jay-Z", "English", 0.68, 0.86, "Hip-Hop",
 ["https://www.youtube.com/watch?v=HAjQf0zag_4"]),

("Public Service Announcement", "Jay-Z", "English", 0.72, 0.84, "Hip-Hop",
 ["https://www.youtube.com/watch?v=e2XE3bP8wIY"]),

("Izzo (H.O.V.A.)", "Jay-Z", "English", 0.83, 0.81, "Hip-Hop",
 ["https://www.youtube.com/watch?v=JrL6b9YFYb0"]),

("Dirt Off Your Shoulder", "Jay-Z", "English", 0.79, 0.78, "Hip-Hop",
 ["https://www.youtube.com/watch?v=Oz_-VaTHpc8"]),

("Show Me What You Got", "Jay-Z", "English", 0.80, 0.82, "Hip-Hop",
 ["https://www.youtube.com/watch?v=FS4U-HAHwps"]),

("Encore", "Jay-Z", "English", 0.85, 0.80, "Hip-Hop",
 ["https://www.youtube.com/watch?v=FQ3slUz7Jo8"]),

("Can I Kick It?", "A Tribe Called Quest", "English", 0.90, 0.74, "Hip-Hop",
 ["https://www.youtube.com/watch?v=O3pyCGnZzYA"]),

("Bonita Applebum", "A Tribe Called Quest", "English", 0.86, 0.69, "Hip-Hop",
 ["https://www.youtube.com/watch?v=KSCYjxdjmJg"]),

("Check the Rhime", "A Tribe Called Quest", "English", 0.88, 0.78, "Hip-Hop",
 ["https://www.youtube.com/watch?v=1QWEPdgS3As"]),

("Jazz (We've Got)", "A Tribe Called Quest", "English", 0.82, 0.73, "Hip-Hop",
 ["https://www.youtube.com/watch?v=cxN4nKk2cfk"]),

("Award Tour Remix", "A Tribe Called Quest", "English", 0.80, 0.79, "Hip-Hop",
 ["https://www.youtube.com/watch?v=krpC1q1J0nI"]),

("Breakadawn", "De La Soul", "English", 0.85, 0.72, "Hip-Hop",
 ["https://www.youtube.com/watch?v=V6a7hG6aB2Q"]),

("Me Myself and I", "De La Soul", "English", 0.88, 0.70, "Hip-Hop",
 ["https://www.youtube.com/watch?v=7z0lQF3p1f0"]),

("Ring Ring Ring", "De La Soul", "English", 0.84, 0.75, "Hip-Hop",
 ["https://www.youtube.com/watch?v=PXAlCazm3J0"]),

("Buddy", "De La Soul", "English", 0.83, 0.79, "Hip-Hop",
 ["https://www.youtube.com/watch?v=F3Wg4z0b9r0"])



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
