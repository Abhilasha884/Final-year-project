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

    # ("Sweet But Psycho", "Ava Max", "English", 0.8, 0.85, "Pop", "https://www.youtube.com/watch?v=WXBHCQYxwr0"),
    # ("Old Town Road", "Lil Nas X ft. Billy Ray Cyrus", "English", 0.8, 0.8, "Country Rap", "https://www.youtube.com/watch?v=r7qovpFAGrQ"),
    # ("Bad Guy", "Billie Eilish", "English", 0.6, 0.7, "Pop", "https://www.youtube.com/watch?v=DyDfgMOUjCI"),
    # ("Dance Monkey", "Tones and I", "English", 0.85, 0.8, "Electropop", "https://www.youtube.com/watch?v=q0hyYWKXF0Q"),
    # ("Senorita", "Shawn Mendes, Camila Cabello", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=Pkh8UtuejGw"),
    # ("Circles", "Post Malone", "English", 0.6, 0.5, "Pop", "https://www.youtube.com/watch?v=GrAchTdepsU"),
    # ("Sunflower", "Post Malone, Swae Lee", "English", 0.7, 0.6, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=ApXoWvfEYVU"),
    # ("Mood", "24kGoldn ft. Iann Dior", "English", 0.8, 0.8, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=GrAchTdepsU"),
    # ("Levitating", "Dua Lipa", "English", 0.8, 0.8, "Pop/Disco", "https://www.youtube.com/watch?v=TUVcZfQe-Kw"),
    # ("Good 4 U", "Olivia Rodrigo", "English", 0.9, 0.9, "Pop/Rock", "https://www.youtube.com/watch?v=gNi_6U5Pm_o"),
    # ("Dynamite", "BTS", "English", 0.9, 0.8, "K-Pop", "https://www.youtube.com/watch?v=gdZLi9oWNZg"),
    #  ("Butter", "BTS", "English", 0.9, 0.85, "K-Pop", "https://www.youtube.com/watch?v=WMweEpGlu_U"),
    # ("drivers license", "Olivia Rodrigo", "English", 0.2, 0.3, "Pop", "https://www.youtube.com/watch?v=ZmDBbnmKpqQ"),
    # ("Tum Se Hi", "Mohit Chauhan", "Hindi", 0.8, 0.5, "Romantic", "https://www.youtube.com/watch?v=tQ0yjYUFKAE"),
    # ("Shivers", "Ed Sheeran", "English", 0.85, 0.8, "Pop", "https://www.youtube.com/watch?v=Il-an3K9pjg"),
    # ("Peaches", "Justin Bieber ft. Daniel Caesar, Giveon", "English", 0.7, 0.5, "R&B", "https://www.youtube.com/watch?v=tQ0yjYUFKAE"),

    # ("Adore You", "Harry Styles", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=VF-r5TtlT9w"),
    # ("Watermelon Sugar", "Harry Styles", "English", 0.9, 0.7, "Pop", "https://www.youtube.com/watch?v=E07s5ZYygMg"),
    # ("Stay", "The Kid LAROI, Justin Bieber", "English", 0.8, 0.9, "Pop", "https://www.youtube.com/watch?v=kTJczUoc26U"),
    # ("Kiss Me More", "Doja Cat ft. SZA", "English", 0.9, 0.8, "Pop", "https://www.youtube.com/watch?v=0EVVKs6DQLo"),
    # ("Leave The Door Open", "Silk Sonic", "English", 0.9, 0.8, "R&B/Soul", "https://www.youtube.com/watch?v=adLGHcj_fmA"),
    # ("Cold Heart (PNAU Remix)", "Elton John & Dua Lipa", "English", 0.85, 0.75, "Pop/Dance", "https://www.youtube.com/watch?v=qod03PVTLqk"),
    # ("Heat Waves", "Glass Animals", "English", 0.6, 0.5, "Indie Pop", "https://www.youtube.com/watch?v=mRD0-GxqHVo"),
    # ("Happier Than Ever", "Billie Eilish", "English", 0.3, 0.4, "Alternative Pop", "https://www.youtube.com/watch?v=5GJWxDKyk3A"),
    # ("Montero (Call Me By Your Name)", "Lil Nas X", "English", 0.7, 0.8, "Pop/Hip-Hop", "https://www.youtube.com/watch?v=6swmTBVI83k"),
    # ("Permission to Dance", "BTS", "English", 0.95, 0.85, "K-Pop", "https://www.youtube.com/watch?v=CuklIb9d3fI"),
    # ("Bad Habits", "Ed Sheeran", "English", 0.7, 0.8, "Pop", "https://www.youtube.com/watch?v=orJSJGHjBLI"),
    # ("Industry Baby", "Lil Nas X, Jack Harlow", "English", 0.8, 0.9, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=UTHLKHL_whs"),
    # ("You Right", "Doja Cat, The Weeknd", "English", 0.6, 0.7, "R&B", "https://www.youtube.com/watch?v=JXgV1rXUoME"),


    # ("Woman", "Doja Cat", "English", 0.8, 0.7, "Pop/R&B", "https://www.youtube.com/watch?v=yxW5yuzVi8w"),  
    # ("Fancy Like", "Walker Hayes", "English", 0.8, 0.7, "Country", "https://www.youtube.com/watch?v=G_zuB-ogIBw"),
    # ("Billie Eilish.", "Armani White", "English", 0.85, 0.9, "Hip-Hop", "https://www.youtube.com/watch?v=4vYOwhll1fs"),  
    # ("As It Was", "Harry Styles", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=H5v3kku4y6Q"),
    # ("Rich Flex", "Drake & 21 Savage", "English", 0.6, 0.8, "Hip-Hop", "https://www.youtube.com/watch?v=sllmR3TJd_w"),
    # ("Kill Bill", "SZA", "English", 0.5, 0.6, "R&B", "https://www.youtube.com/watch?v=MSRcC626prw"),
    # ("Calm Down", "Rema, Selena Gomez", "English", 0.9, 0.75, "Afrobeats", "https://www.youtube.com/watch?v=WcIcVapfqXw"),
    # ("I'm Good (Blue)", "David Guetta, Bebe Rexha", "English", 0.95, 0.9, "Dance", "https://www.youtube.com/watch?v=90RLzVUuXe4"),
    # ("Celestial", "Ed Sheeran", "English", 0.8, 0.65, "Pop", "https://www.youtube.com/watch?v=23g5HBOg3Ic"),
    # ("Forget Me", "Lewis Capaldi", "English", 0.25, 0.35, "Ballad", "https://www.youtube.com/watch?v=nBZlrbrBO1I"),
    # ("Flowers", "Miley Cyrus", "English", 0.8, 0.6, "Pop", "https://www.youtube.com/watch?v=G7KNmW9a75Y"),  
    # ("Made You Look", "Meghan Trainor", "English", 0.9, 0.8, "Pop", "https://www.youtube.com/watch?v=gPCCYMeXin0"),
    # ("Under The Influence", "Chris Brown", "English", 0.7, 0.75, "R&B", "https://www.youtube.com/watch?v=pfxyk1glEq4"),
    # ("Lag Jaa Gale", "Lata Mangeshkar", "Hindi", 0.3, 0.4, "Classical", "https://www.youtube.com/watch?v=Umqb9KENgmk"),  # Works for download
    # ("Gallan Goodiyaan", "Yashita Sharma, Manish Kumar Tipu, Farhan Akhtar, Shankar Mahadevan, Sukhwinder Singh", "Hindi", 0.9, 0.8, "Dance/Pop", "https://www.youtube.com/watch?v=HZ7PAyCDwEg")  # Excel Movies working link
   
    # ("Victoria’s Secret", "JAX", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=F9K5IS-inHs&utm_source=chatgpt.com"),  # Alternate official upload (works globally)
    # ("Vampire", "Olivia Rodrigo", "English", 0.9, 0.9, "Pop-Punk", "https://www.youtube.com/watch?v=RlPNh_PBZb4&utm_source=chatgpt.com"),  # Alternate VEVO link
    # ("Chaiyya Chaiyya", "Sukhwinder Singh, Sapna Awasthi", "Hindi", 0.8, 0.7, "Folk/Pop", "https://www.youtube.com/watch?v=APo73rlxWaE&utm_source=chatgpt.com"),  # Works globally
    # ("Aap Jaisa Koi", "Nazia Hassan", "Hindi", 0.75, 0.65, "Pop", "https://www.youtube.com/watch?v=1jf5kuvScJc&utm_source=chatgpt.com"),  # Stable remastered upload
    # ("Sandese Aate Hain", "Sonu Nigam, Roop Kumar Rathod", "Hindi", 0.5, 0.6, "Patriotic", "https://www.youtube.com/watch?v=UovOpJXNhi8&utm_source=chatgpt.com"),  # Zee Music working link


    
    # ("Kalank Title Track", "Arijit Singh", "Hindi", 0.7, 0.6, "Emotional", "https://www.youtube.com/watch?v=b4hrUSBP4nc&utm_source=chatgpt.com"),
    # ("Kesariya", "Arijit Singh", "Hindi", 0.9, 0.6, "Romantic/Folk", "https://www.youtube.com/watch?v=BddP6PYo2gs&utm_source=chatgpt.com"),
    # ("Deva Deva", "Arijit Singh, Jonita Gandhi", "Hindi", 0.8, 0.7, "Spiritual/Fusion", "https://www.youtube.com/watch?v=WjAPDofGg28&utm_source=chatgpt.com"),
    # ("Manike", "Jubin Nautiyal, Yohani", "Hindi", 0.75, 0.8, "Pop/Romantic", "https://www.youtube.com/watch?v=zqHUMF9syFA&utm_source=chatgpt.com"),
    # ("Pasoori", "Shae Gill, Ali Sethi", "Hindi", 0.6, 0.5, "Folk/Fusion", "https://www.youtube.com/watch?v=5Eqb_-j3FDA&utm_source=chatgpt.com"),
    # ("Oo Antava Oo Oo Antava", "Indravathi Chauhan", "Hindi", 0.85, 0.9, "Dance/Item", "https://www.youtube.com/watch?v=u_wB6byrl5k&utm_source=chatgpt.com"),
    # ("Naatu Naatu", "Rahul Sipligunj, Kaala Bhairava", "Hindi", 0.9, 0.95, "Dance/Folk", "https://www.youtube.com/watch?v=4_eEgJhsBMo&utm_source=chatgpt.com"),
    # ("Jhoome Jo Pathaan", "Arijit Singh, Sukriti Kakar", "Hindi", 0.85, 0.8, "Pop/Dance", "https://www.youtube.com/watch?v=YxWlaYCA8MU&utm_source=chatgpt.com"),


    # ("Maan Meri Jaan", "King", "Hindi", 0.7, 0.6, "Pop/Romantic", "https://www.youtube.com/watch?v=VuG7ge_8I2Y&utm_source=chatgpt.com"),
    # ("Chaleya", "Arijit Singh, Shilpa Rao", "Hindi", 0.8, 0.5, "Romantic/Pop", "https://www.youtube.com/watch?v=VAdGW7QDJiU&utm_source=chatgpt.com"),
    # ("Param Sundari", "Shreya Ghoshal", "Hindi", 0.9, 0.8, "Folk", "https://www.youtube.com/watch?v=w4ClQO0FFQg&utm_source=chatgpt.com"),
    # ("Raataan Lambiyan", "Jubin Nautiyal, Asees Kaur", "Hindi", 0.8, 0.5, "Romantic", "https://www.youtube.com/watch?v=gvyUuxdRdR4&utm_source=chatgpt.com"),
    # ("Nadiyon Paar (Let the Music Play Again)", "Shamur, Rashmeet Kaur, IP Singh, Sachin-Jigar", "Hindi", 0.9, 0.9, "Dance", "https://www.youtube.com/watch?v=DKj5m9cSMZs&utm_source=chatgpt.com"),
    # ("Ranjha", "B Praak, Jasleen Royal", "Hindi", 0.5, 0.4, "Romantic", "https://www.youtube.com/watch?v=V7LwfY5U5WI&utm_source=chatgpt.com"),
    # ("Meri Jaan", "Neeti Mohan", "Hindi", 0.75, 0.5, "Romantic", "https://www.youtube.com/watch?v=ldWFpeu36K0&utm_source=chatgpt.com"),
    # ("Dhokha", "Arijit Singh", "Hindi", 0.25, 0.5, "Sad", "https://www.youtube.com/watch?v=2JBYnvUlAEc&utm_source=chatgpt.com"),

    # ("Jab Saiyaan", "Shreya Ghoshal", "Hindi", 0.8, 0.6, "Romantic", "https://www.youtube.com/watch?v=0uR45roIGA4&utm_source=chatgpt.com"),
    # ("Bhool Bhulaiyaa 2 Title Track", "Neeraj Shridhar", "Hindi", 0.85, 0.9, "Pop", "https://www.youtube.com/watch?v=J1rOfVst-EQ&utm_source=chatgpt.com"),
    # ("Dil Bechara", "A.R. Rahman", "Hindi", 0.9, 0.9, "Romantic", "https://in.video.search.yahoo.com/search/video;_ylt=AwrKFukJw.ZohAIA8i67HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=dil+bechara&fr=mcafee&turl=https%3A%2F%2Ftse3.mm.bing.net%2Fth%2Fid%2FOVP.nZWEMJW7hoBkY0z2ijRtKQEsDh%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DzX9decJVplU&tit=Dil+Bechara+-+Lyrical+Video+%7C+Sushant+Singh+Rajput+%7C+Sanjana+Sanghi+%7C+A.+R.+Rahman+Hit+Song&pos=21&vid=4174ee5a6dfcfca24576f313e1d7cb42&sigr=bnALHJP_Tn6q&sigt=.GIMstHfIo5k&sigi=.wnImuwJcs33"),
    # ("Duniyaa", "Akhil, Dhvani Bhanushali", "Hindi", 0.8, 0.8, "Romantic", "https://www.youtube.com/watch?v=rIczU1zBw28&utm_source=chatgpt.com"),
    # ("Hawayein", "Arijit Singh", "Hindi", 0.9, 0.9, "Romantic", "https://www.youtube.com/watch?v=lBACSadCKaQ&utm_source=chatgpt.com"),
    # ("O Saki Saki", "Neha Kakkar, Tulsi Kumar", "Hindi", 0.8, 0.8, "Dance", "https://www.youtube.com/watch?v=_uUdJalMaF8&utm_source=chatgpt.com")

    # ("Ordinary", "Alex Warren", "English", 0.75, 0.7, "Pop", "https://www.youtube.com/watch?v=u2ah9tWTkmk")
    # ("Abracadabra", "Lady Gaga", "English", 0.85, 0.9, "Dance-Pop", "https://www.youtube.com/watch?v=vBynw9Isr28"),
    # ("Something Beautiful", "Miley Cyrus", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=y2nu8zpVBmY"),
    # ("Luther", "Kendrick Lamar ft. SZA", "English", 0.7, 0.65, "Rap/R&B", "https://www.rollingstone.com/music/music-news/kendrick-lamar-sza-luther-music-video-1235315583/"),
    # ("Born Again", "Lisa ft. Doja Cat and Raye", "English", 0.88, 0.85, "Pop/Electronic", "https://www.rollingstone.com/music/music-news/lisa-doja-cat-raye-born-again-song-1235259355/"),
    # ("Twilight Zone", "Ariana Grande", "English", 0.75, 0.8, "Synthpop", "https://mykissradio.com/2025/08/06/ariana-grande-puts-out-twilight-zone-video-tying-into-supernatural-story/"),
    # ("Soft Girl Era", "Ari Lennox", "English", 0.7, 0.6, "R&B", "https://thisisrnb.com/2025/04/softness-in-motion-ari-lennox-dazzles-in-dreamy-new-soft-girl-era-visual/"),
    # ("Physical", "Dua Lipa ft. Troye Sivan", "English", 0.85, 0.9, "Pop", "https://www.youtube.com/watch?v=1NZex2G3hA8"),
    # ("APT.", "Rosé & Bruno Mars", "English", 0.8, 0.85, "Pop-Punk", "https://en.wikipedia.org/wiki/Apt._(song)")

 
    
    
    # ("Kings & Queens", "Ava Max", "English", 0.85, 0.8, "Pop Rock", "https://www.youtube.com/watch?v=jH1RNk8954Q"),
    ("Don’t Go Yet", "Camila Cabello", "English", 0.75, 0.85, "Latin Pop", "https://in.video.search.yahoo.com/search/video;_ylt=AwrKC.793PdoNQIAjz67HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Nj?type=E211IN1357G0&p=Don%E2%80%99t+Go+Yet+%E2%80%93+Camila+Cabello&fr=mcafee&turl=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOVP.fsGILyGEOt56NgEJ0SNOwQHgFo%3Fpid%3DApi%26w%3D296%26h%3D156%26c%3D7%26p%3D0&rurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D4cujO3AL_6w&tit=Camila+Cabello+-+Don%27t+Go+Yet+%28Lyrics%29&pos=01&vid=539a22bf814f7824ce8eee6fd9f8648c&sigr=9DoFbcbuRN_Z&sigt=dFR6Q3Uz6c7c&sigi=ZxuGFtwrftxz")


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
