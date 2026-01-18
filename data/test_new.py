import os
import re
import pandas as pd
import lyricsgenius
from yt_dlp import YoutubeDL
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "cookies.txt"))
FFMPEG_PATH = r"C:\Users\HP\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"


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

("Take It Easy", "Eagles", "English", 0.70, 0.65, "Roots Rock",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKHBCYtGxpJAIAF1K7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Take+It+Easy%22%2C+%22Eagles%22&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.biIUeUlGJLzSg6MMMZufbgHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D32Oc2d_3yEk&tit=Take+It+Easy+%282013+Remaster%29&pos=01&vid=47a676afc5584c0ce6d275128d2ce07a&sigr=8xHszs6qjwIv&sigt=d6g9cI1lo81w&sigi=FV5kyQrxu7hh"
]
),

("That'll Be the Day", "Buddy Holly", "English", 0.78, 0.75, "Rockabilly",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKDnHntGxpBAIAP_K7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22That%27ll+Be+the+Day%22%2C+%22Buddy+Holly%22&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.riUUWCtrKeeWtkJ8hNEUhwFoFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DV46r035VntQ&tit=That%27ll+Be+The+Day&pos=01&vid=762dc12cf9516a5dc76bfb9ad21191f5&sigr=HVMZoYNzSJDx&sigt=7YikbHnPJZ5h&sigi=DIcgS7fMIwWP"
]
),

("Summertime Blues", "Eddie Cochran", "English", 0.80, 0.78, "Rockabilly",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKDzvRtGxpKQIAfxe7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Summertime+Blues%22%2C+%22Eddie+Cochran%22&fr=mcafee&turl=https%3A%2F%2Ftse3.mm.bing.net%2Fth%2Fid%2FOVP.ehCj_WBxP7db3mJpAxSzagFoFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D1C38Zevwyx4&tit=Summertime+Blues&pos=01&vid=f56a512ea2185c90842f84c987a96e41&sigr=YZ4yPt6QhTLA&sigt=h.19RsUcw9ZV&sigi=FwcEExrwoyeK"
]
),

("Great Balls of Fire", "Jerry Lee Lewis", "English", 0.82, 0.80, "Rockabilly",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKDzv.tGxpKQIA7y67HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Great+Balls+of+Fire%22%2C+%22Jerry+Lee+Lewis%22&fr=mcafee&turl=https%3A%2F%2Ftse3.mm.bing.net%2Fth%2Fid%2FOVP.ktDpU8Shu-tjvK1hmH_1xQHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DMMwq_D-DWsU&tit=Jerry+Lee+Lewis+-+Great+Balls+of+Fire+%28Lyric+Video%29&pos=01&vid=5651003c289e199c39a930a26074e545&sigr=y1zvvL27BB05&sigt=4j_gXcYpnDZs&sigi=olEy6tw7VOYk"
]
),

("Where the Stars and Stripes and the Eagle Fly", "Aaron Tippin", "English", 0.65, 0.60, "Patriotic Country",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKDnEStWxpJgIAZ5i7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Where+the+Stars+and+Stripes+and+the+Eagle+Fly%22%2C+%22Aaron+Tippin%22&fr=mcafee&turl=https%3A%2F%2Ftse2.mm.bing.net%2Fth%2Fid%2FOVP.CsCd3Lcl-m780i5KqLIkcAG4Fo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DTTKmjhJ1__o&tit=Aaron+Tippin+-+Where+The+Stars+%26+Stripes+%26+The+Eagle+Fly&pos=01&vid=dbc5c1ee8ab35bf463b14d831bbaf01c&sigr=BoMdKY4UIFwY&sigt=6abf7_JUVDrQ&sigi=kBhETIWBzQrz"
]
),

("Arlington", "Trace Adkins", "English", 0.48, 0.42, "Patriotic Country",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKAg4ntWxpcwIApOK7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Arlington%22%2C+%22Trace+Adkins%22&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.eMOgCmAeocf_kRFSb-21QgHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DrJO7lJIxG10&tit=Trace+Adkins+-+Arlington&pos=01&vid=bc3f4ef049ead3dca61eb0378a4acb80&sigr=RiwFJSgxYnmW&sigt=KtN6nPnXzt8E&sigi=1n0VO09pp9Um"
]
),

("Some Gave All", "Billy Ray Cyrus", "English", 0.50, 0.45, "Patriotic Country",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=Awr1Sa07tWxpEQIAz9q7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Some+Gave+All%22%2C+%22Billy+Ray+Cyrus%22&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.GJpmUFj0GJ2nmMw_x2zlnQHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DydWhRObVxrM&tit=Billy+Ray+Cyrus+-+Some+Gave+All+%28Official+Music+Video%29&pos=01&vid=883e6488296f5fb029e6a56e4249c004&sigr=OJJurFYU424x&sigt=j7Yl0GFmdFy0&sigi=.3anbwkMb7G0"
]
),

("Ae Mere Watan Ke Logon", "Lata Mangeshkar", "Hindi", 0.40, 0.35, "Indian Patriotic Song",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrKAg5OtWxphAIAl5C7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Ae+Mere+Watan+Ke+Logon%22%2C+%22Lata+Mangeshkar%22&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.6xatA7ZH2o4P_oidKNy6YAHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DSz0ew6pxa6Y&tit=Ae+Mere+Watan+Ke+Logon+Full+Song+%7C+%E0%A4%90+%E0%A4%AE%E0%A5%87%E0%A4%B0%E0%A5%87+%E0%A4%B5%E0%A4%A4%E0%A4%A8+%E0%A4%95%E0%A5%87+%E0%A4%B2%E0%A5%8B%E0%A4%97%E0%A5%8B+%7C+Lata+Mangeshkar+%7C+Independence+Day+Song&pos=01&vid=9dbb268d6616134956f01ce4599cf74b&sigr=w9_mV5r7db2u&sigt=ObIQALgljFAV&sigi=qGUF92FMOkL7"
]
),

("Sandese Aate Hain", "Sonu Nigam", "Hindi", 0.45, 0.40, "Indian Patriotic Song",
[
    "https://in.video.search.yahoo.com/search/video;_ylt=AwrPrjhitWxpBQIAF8.7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=%22Sandese+Aate+Hain%22%2C+%22Sonu+Nigam%22&fr=mcafee&turl=https%3A%2F%2Ftse3.mm.bing.net%2Fth%2Fid%2FOVP.IUWUtxmOD5U8ZWWoCxJsoQEsDh%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DUovOpJXNhi8&tit=Border+-+Sandese+Aate+Hain+%7C+Sonu+Nigam%2C+Roop+Kumar+Rathod+%7C+Sunny+Deol+%7C+Hindi+Patriotic+Song&pos=01&vid=e5b45b12aba99565c3e2ac20f4298b1e&sigr=CbeDu1P3lWGk&sigt=5UIKotkOKKnd&sigi=BTD6W1Tm_BnK"
]
),

("Tera Rang Aisa Chad Gaya", "Mahendra Kapoor", "Hindi", 0.55, 0.50, "Indian Patriotic Song",
[
    "https://www.youtube.com/watch?v=bP7jukiJHSw&pp=ygUtdGVyYSByYW5nIGFpc2EgY2hhZCBnYXlhIHNvbmcgbWFoZW5kcmEga2Fwb29y"
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

    for url in [u for u in url_list if "youtube.com" in u or "youtu.be" in u]:
        try:

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
                'cookies': COOKIES_PATH,
                'ffmpeg_location': FFMPEG_PATH,
                'noplaylist': True,
                'geo_bypass': True,
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
