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

("Are You Experienced?", "Jimi Hendrix", "English", 0.82, 0.9, "Rock", ["https://www.youtube.com/watch?v=WUOb5v--vTY"]),
("Purple Haze", "Jimi Hendrix", "English", 0.8, 0.96, "Rock", ["https://www.youtube.com/watch?v=cJunCsrhJjg"]),
("Voodoo Child (Slight Return)", "Jimi Hendrix", "English", 0.84, 0.98, "Rock", ["https://www.youtube.com/watch?v=BUmMo0KgdXE"]),
("Hey Joe", "Jimi Hendrix", "English", 0.86, 0.82, "Rock", ["https://www.youtube.com/watch?v=rXwMrBb2x1Q"]),
("Little Wing", "Jimi Hendrix", "English", 0.88, 0.76, "Rock", ["https://www.youtube.com/watch?v=35luFxHO5E0"]),
("Black Magic Woman", "Santana", "English", 0.9, 0.84, "Rock", ["https://www.youtube.com/watch?v=9wT1s96JIb0"]),
("Smooth", "Santana", "English", 0.94, 0.88, "Rock", ["https://www.youtube.com/watch?v=6Whgn_iE5uc"]),
("Maria Maria", "Santana", "English", 0.92, 0.82, "Rock", ["https://www.youtube.com/watch?v=nPLV7lGbmT4"]),
("Oye Como Va", "Santana", "English", 0.96, 0.86, "Rock", ["https://www.youtube.com/watch?v=J7ATTjg7tpE"]),
("Europa", "Santana", "English", 0.88, 0.7, "Rock", ["https://www.youtube.com/watch?v=Ot6pSrKT1oc"]),
("Born to Be Wild", "Steppenwolf", "English", 0.92, 0.96, "Rock", ["https://www.youtube.com/watch?v=egMWlD3fLJ8"]),
("Magic Carpet Ride", "Steppenwolf", "English", 0.9, 0.88, "Rock", ["https://www.youtube.com/watch?v=HPE9a_epmWw"]),
("Rock Me", "Steppenwolf", "English", 0.88, 0.9, "Rock", ["https://www.youtube.com/watch?v=xa6xquyj5X0"]),
("Monster", "Steppenwolf", "English", 0.82, 0.92, "Rock", ["https://www.youtube.com/watch?v=Sk3sURDS4IA"]),
("The Pusher", "Steppenwolf", "English", 0.8, 0.84, "Rock", ["https://www.youtube.com/watch?v=LmdDMGp8IS8"]),
("We Built This City", "Starship", "English", 0.94, 0.86, "Rock", ["https://www.youtube.com/watch?v=K1b8AhIsSYQ"]),
("Nothing's Gonna Stop Us Now", "Starship", "English", 0.96, 0.82, "Rock", ["https://www.youtube.com/watch?v=3wxyN3z9PL4"]),
("Sara", "Starship", "English", 0.88, 0.74, "Rock", ["https://www.youtube.com/watch?v=32ScTb6_KHg"]),
("It's Not Enough", "Starship", "English", 0.86, 0.8, "Rock", ["https://www.youtube.com/watch?v=zSZnq5ZbP1w"]),
("Find Your Way Back", "Starship", "English", 0.9, 0.84, "Rock", ["https://www.youtube.com/watch?v=Dj4gqJzlIvc"]),
("Centerfold", "The J. Geils Band", "English", 0.94, 0.88, "Rock", ["https://www.youtube.com/watch?v=BqDjMZKf-wg"]),
("Freeze Frame", "The J. Geils Band", "English", 0.92, 0.9, "Rock", ["https://www.youtube.com/watch?v=wHo43B6nu60"]),
("Love Stinks", "The J. Geils Band", "English", 0.88, 0.86, "Rock", ["https://www.youtube.com/watch?v=E0LAs7X5ybE"]),
("Give It to Me", "The J. Geils Band", "English", 0.9, 0.88, "Rock", ["https://www.youtube.com/watch?v=UDU3evB5_K8"]),
("Musta Got Lost", "The J. Geils Band", "English", 0.86, 0.82, "Rock", ["https://www.youtube.com/watch?v=x4y1mKxYD6Q"]),
("The Boys Are Back in Town", "Thin Lizzy", "English", 0.92, 0.9, "Rock", ["https://www.youtube.com/watch?v=5_xqb416S7o"]),
("Jailbreak", "Thin Lizzy", "English", 0.88, 0.92, "Rock", ["https://www.youtube.com/watch?v=At_rPiCCnpY"]),
("Whiskey in the Jar", "Thin Lizzy", "English", 0.9, 0.84, "Rock", ["https://www.youtube.com/watch?v=b3UOe5LF_bw"]),
("Rosalie", "Thin Lizzy", "English", 0.86, 0.9, "Rock", ["https://www.youtube.com/watch?v=cSo9CC2wKVI"]),
("Still in Love With You", "Thin Lizzy", "English", 0.88, 0.76, "Rock", ["https://www.youtube.com/watch?v=0HQV9I0C_bM"]),
("Barracuda", "Heart", "English", 0.9, 0.94, "Rock", ["https://www.youtube.com/watch?v=PeMvMNpvB5M"]),
("Crazy on You", "Heart", "English", 0.92, 0.9, "Rock", ["https://www.youtube.com/watch?v=yZjEC4WhCvg"]),
("Alone", "Heart", "English", 0.88, 0.7, "Rock", ["https://www.youtube.com/watch?v=1Cw1ng75KP0"]),
("Magic Man", "Heart", "English", 0.86, 0.88, "Rock", ["https://www.youtube.com/watch?v=A99bkAcLIas"]),
("These Dreams", "Heart", "English", 0.9, 0.74, "Rock", ["https://www.youtube.com/watch?v=41P8UxneDJE"]),
("More Than Words", "Extreme", "English", 0.92, 0.68, "Rock", ["https://www.youtube.com/watch?v=UrIiLvg58SY"]),
("Hole Hearted", "Extreme", "English", 0.9, 0.82, "Rock", ["https://www.youtube.com/watch?v=I-h4A7bF8wQ"]),
("Get the Funk Out", "Extreme", "English", 0.88, 0.9, "Rock", ["https://www.youtube.com/watch?v=IqP76XWHQI0"]),
("Decadence Dance", "Extreme", "English", 0.86, 0.92, "Rock", ["https://www.youtube.com/watch?v=OoKJpcROgJk"]),
("Rest in Peace", "Extreme", "English", 0.84, 0.88, "Rock", ["https://www.youtube.com/watch?v=odz3c68JE1c"]),
("Plush (Acoustic)", "Stone Temple Pilots", "English", 0.84, 0.7, "Rock", ["https://www.youtube.com/watch?v=xnWGJoPvQ5U"]),
("Creep (Acoustic)", "Stone Temple Pilots", "English", 0.82, 0.66, "Rock", ["https://www.youtube.com/watch?v=Lml0wUI_v8g"]),
("Interstate Love Song (Acoustic)", "Stone Temple Pilots", "English", 0.86, 0.72, "Rock", ["https://www.youtube.com/watch?v=b-dZkDpSiHA"]),
("Big Bang Baby", "Stone Temple Pilots", "English", 0.88, 0.9, "Rock", ["https://www.youtube.com/watch?v=G0gAxuvo5rc"]),
("Sour Girl", "Stone Temple Pilots", "English", 0.9, 0.8, "Rock", ["https://www.youtube.com/watch?v=YxS4lqppZ6Y"]),
("Under the Bridge", "Red Hot Chili Peppers", "English", 0.9, 0.74, "Rock", ["https://www.youtube.com/watch?v=GLvohMXgcBo"]),
("Californication", "Red Hot Chili Peppers", "English", 0.88, 0.78, "Rock", ["https://www.youtube.com/watch?v=YlUKcNNmywk"]),
("Scar Tissue", "Red Hot Chili Peppers", "English", 0.92, 0.7, "Rock", ["https://www.youtube.com/watch?v=mzJj5-lubeM"]),
("Otherside", "Red Hot Chili Peppers", "English", 0.84, 0.76, "Rock", ["https://www.youtube.com/watch?v=rn_YodiJO6k"]),
("Dani California", "Red Hot Chili Peppers", "English", 0.9, 0.88, "Rock", ["https://www.youtube.com/watch?v=Sb5aq5HcS1A"]),
("Numb", "Linkin Park", "English", 0.86, 0.82, "Rock", ["https://www.youtube.com/watch?v=kXYiU_JCYtU"]),
("In the End", "Linkin Park", "English", 0.88, 0.9, "Rock", ["https://www.youtube.com/watch?v=eVTXPUF4Oz4"]),
("Crawling", "Linkin Park", "English", 0.78, 0.86, "Rock", ["https://www.youtube.com/watch?v=Gd9OhYroLN0"]),
("Somewhere I Belong", "Linkin Park", "English", 0.82, 0.88, "Rock", ["https://www.youtube.com/watch?v=zsCD5XCu6CM"]),
("Breaking the Habit", "Linkin Park", "English", 0.84, 0.8, "Rock", ["https://www.youtube.com/watch?v=v2H4l9RpkwM"]),
("Bring Me to Life (Acoustic)", "Evanescence", "English", 0.8, 0.72, "Rock", ["https://www.youtube.com/watch?v=i3MKTm-49uI"]),
("Lithium (Acoustic)", "Evanescence", "English", 0.78, 0.68, "Rock", ["https://www.youtube.com/watch?v=PJGpsL_XYQI"]),
("Tourniquet", "Evanescence", "English", 0.72, 0.86, "Rock", ["https://www.youtube.com/watch?v=5Wt1yEejRbM"]),
("Imaginary", "Evanescence", "English", 0.84, 0.78, "Rock", ["https://www.youtube.com/watch?v=dpjz5uRCXTg"]),
("Everybody's Fool", "Evanescence", "English", 0.8, 0.82, "Rock", ["https://www.youtube.com/watch?v=jhC1pI76Rqo"]),
("Take Me Out", "Franz Ferdinand", "English", 0.92, 0.88, "Rock", ["https://www.youtube.com/watch?v=Ijk4j-r7qPA"]),
("Do You Want To", "Franz Ferdinand", "English", 0.94, 0.9, "Rock", ["https://www.youtube.com/watch?v=1OJRRUnY--A"]),
("Walk Away", "Franz Ferdinand", "English", 0.88, 0.82, "Rock", ["https://www.youtube.com/watch?v=XZ_wzpmVzdE"]),
("No You Girls", "Franz Ferdinand", "English", 0.9, 0.86, "Rock", ["https://www.youtube.com/watch?v=25sBhhOR4lw"]),
("Love Illumination", "Franz Ferdinand", "English", 0.92, 0.88, "Rock", ["https://www.youtube.com/watch?v=Ooq23i-QGBM"]),
("Seven Nation Army (Acoustic)", "The White Stripes", "English", 0.86, 0.7, "Rock", ["https://www.youtube.com/watch?v=0J2QdDbelmY"]),
("Hotel Yorba", "The White Stripes", "English", 0.88, 0.82, "Rock", ["https://www.youtube.com/watch?v=DZPEUyiNcjA"]),
("We're Going to Be Friends", "The White Stripes", "English", 0.9, 0.68, "Rock", ["https://www.youtube.com/watch?v=PKfD8d3XJok"]),
("Dead Leaves and the Dirty Ground", "The White Stripes", "English", 0.82, 0.9, "Rock", ["https://www.youtube.com/watch?v=7OyytKqYjkE"]),
("The Denial Twist", "The White Stripes", "English", 0.84, 0.88, "Rock", ["https://www.youtube.com/watch?v=y6LuwU3LPLE"]),
("Howlin' for You", "The Black Keys", "English", 0.92, 0.88, "Rock", ["https://www.youtube.com/watch?v=TLSpj7q6_mM"]),
("Lonely Boy", "The Black Keys", "English", 0.94, 0.9, "Rock", ["https://www.youtube.com/watch?v=a_426RiwST8"]),
("Gold on the Ceiling", "The Black Keys", "English", 0.9, 0.92, "Rock", ["https://www.youtube.com/watch?v=6yCIDkFI7ew"]),
("Tighten Up", "The Black Keys", "English", 0.88, 0.84, "Rock", ["https://www.youtube.com/watch?v=mpaPBCBjSVc"]),
("Little Black Submarines", "The Black Keys", "English", 0.86, 0.8, "Rock", ["https://www.youtube.com/watch?v=6k8es2BNloE"]),
("Use Me", "Hozier", "English", 0.82, 0.76, "Rock", ["https://www.youtube.com/watch?v=PVjiKRfKpPI"]),
("Almost (Sweet Music)", "Hozier", "English", 0.88, 0.8, "Rock", ["https://www.youtube.com/watch?v=JJ9IX4zgyLs"]),
("From Eden", "Hozier", "English", 0.84, 0.78, "Rock", ["https://www.youtube.com/watch?v=phZ1QcFsPvA"]),
("Angel of Small Death", "Hozier", "English", 0.8, 0.82, "Rock", ["https://www.youtube.com/watch?v=txEI3PEOsPE"]),
("Cherry Wine", "Hozier", "English", 0.86, 0.66, "Rock", ["https://www.youtube.com/watch?v=SdSCCwtNEjA"]),
("Take Me to Church (Acoustic)", "Hozier", "English", 0.82, 0.7, "Rock", ["https://www.youtube.com/watch?v=MYSVMgRr6pw"]),
("Someone New (Acoustic)", "Hozier", "English", 0.84, 0.68, "Rock", ["https://www.youtube.com/watch?v=Ax3qCW319nk"]),
("Work Song", "Hozier", "English", 0.86, 0.72, "Rock", ["https://www.youtube.com/watch?v=nH7bjV0Q_44"]),
("Movement (Acoustic)", "Hozier", "English", 0.8, 0.66, "Rock", ["https://www.youtube.com/watch?v=OSye8OO5TkM"]),
("Dinner & Diatribes", "Hozier", "English", 0.88, 0.86, "Rock", ["https://www.youtube.com/watch?v=HlLx7oE7q3I"]),
("Sex on Fire (Acoustic)", "Kings of Leon", "English", 0.84, 0.7, "Rock", ["https://www.youtube.com/watch?v=kxmHkxpxoaE"]),
("Use Somebody (Live)", "Kings of Leon", "English", 0.88, 0.76, "Rock", ["https://www.youtube.com/watch?v=it-xrhxHcoQ"]),
("Temple (Live)", "Kings of Leon", "English", 0.82, 0.78, "Rock", ["https://www.youtube.com/watch?v=xHhK9KYtuRg"]),
("Pyro (Live)", "Kings of Leon", "English", 0.8, 0.74, "Rock", ["https://www.youtube.com/watch?v=1-_X56xhpGc"]),
("Closer (Live)", "Kings of Leon", "English", 0.82, 0.76, "Rock", ["https://www.youtube.com/watch?v=hYECLfqjdnw"]),

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
