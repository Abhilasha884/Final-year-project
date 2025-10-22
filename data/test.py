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

    # ("Sweet But Psycho", "Ava Max", "English", 0.8, 0.85, "Pop", "https://www.youtube.com/watch?v=WXBHCQYxwr0")

    ("Physical", "Dua Lipa ft. Troye Sivan", "English", 0.85, 0.9, "Pop", "https://www.youtube.com/watch?v=1NZex2G3hA8"),

    ("Sweet But Psycho", "Ava Max", "English", 0.8, 0.85, "Pop", "https://www.youtube.com/watch?v=ZuxcZr8t0S0"),
    ("Old Town Road", "Lil Nas X ft. Billy Ray Cyrus", "English", 0.8, 0.8, "Country Rap", "https://www.youtube.com/watch?v=r7qovpFAGrQ"),
    ("Bad Guy", "Billie Eilish", "English", 0.6, 0.7, "Pop", "https://www.youtube.com/watch?v=DyDfgMOUjCI"),
    ("Dance Monkey", "Tones and I", "English", 0.85, 0.8, "Electropop", "https://www.youtube.com/watch?v=q0hyYWKXF0Q"),
    ("Senorita", "Shawn Mendes, Camila Cabello", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=Pkh8UtuejGw"),
    ("Circles", "Post Malone", "English", 0.6, 0.5, "Pop", "https://www.youtube.com/watch?v=wXhTHyIgQ_U"),
    ("Sunflower", "Post Malone, Swae Lee", "English", 0.7, 0.6, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=ApXoWvfEYVU"),
    ("Mood", "24kGoldn ft. Iann Dior", "English", 0.8, 0.8, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=GrAchTdepsU"),
    ("Adore You", "Harry Styles", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=R-BS87NTV5I"),
    ("Watermelon Sugar", "Harry Styles", "English", 0.9, 0.7, "Pop", "https://www.youtube.com/watch?v=E07s5ZYygMg")



    # ("Levitating", "Dua Lipa", "English", 0.8, 0.8, "Pop/Disco", "https://www.youtube.com/watch?v=TUVcZfQe-Kw"),
    # ("Good 4 U", "Olivia Rodrigo", "English", 0.9, 0.9, "Pop/Rock", "https://www.youtube.com/watch?v=gNi_6U5m_7o"),
    # ("Stay", "The Kid LAROI, Justin Bieber", "English", 0.8, 0.9, "Pop", "https://www.youtube.com/watch?v=kTJFM_g9KAY"),
    # ("Kiss Me More", "Doja Cat ft. SZA", "English", 0.9, 0.8, "Pop", "https://www.youtube.com/watch?v=FjIuS4bgN8g"),
    # ("Leave The Door Open", "Silk Sonic", "English", 0.9, 0.8, "R&B/Soul", "https://www.youtube.com/watch?v=adLGHcj_RQk"),
    # ("Cold Heart (PNAU Remix)", "Elton John & Dua Lipa", "English", 0.85, 0.75, "Pop/Dance", "https://www.youtube.com/watch?v=5D7p7y8R0lM"),
    # ("Heat Waves", "Glass Animals", "English", 0.6, 0.5, "Indie Pop", "https://www.youtube.com/watch?v=l_Qy0P06l5Y"),
    # ("Dynamite", "BTS", "English", 0.9, 0.8, "K-Pop", "https://www.youtube.com/watch?v=gdZLi9oWNZg"),
    # ("Butter", "BTS", "English", 0.9, 0.85, "K-Pop", "https://www.youtube.com/watch?v=WMweEpGlu_U"),
    # ("Happier Than Ever", "Billie Eilish", "English", 0.3, 0.4, "Alternative Pop", "https://www.youtube.com/watch?v=N_H4yKjP0q4"),
    # ("drivers license", "Olivia Rodrigo", "English", 0.2, 0.3, "Pop", "https://www.youtube.com/watch?v=ZmDBbnmKpqQ"),
    # ("Montero (Call Me By Your Name)", "Lil Nas X", "English", 0.7, 0.8, "Pop/Hip-Hop", "https://www.youtube.com/watch?v=6th-W72S_eI"),
    # ("Peaches", "Justin Bieber ft. Daniel Caesar, Giveon", "English", 0.7, 0.5, "R&B", "https://www.youtube.com/watch?v=tQ0yjYUFKAE"),
    # ("Permission to Dance", "BTS", "English", 0.95, 0.85, "K-Pop", "https://www.youtube.com/watch?v=C7M8P0hR-tM"),
    # ("Shivers", "Ed Sheeran", "English", 0.85, 0.8, "Pop", "https://www.youtube.com/watch?v=Il-an3K9pjg"),
    # ("Bad Habits", "Ed Sheeran", "English", 0.7, 0.8, "Pop", "https://www.youtube.com/watch?v=8tg2_jE7gO8"),
    # ("Industry Baby", "Lil Nas X, Jack Harlow", "English", 0.8, 0.9, "Hip-Hop/Pop", "https://www.youtube.com/watch?v=UT9B2aA-Fm4"),
    # ("Woman", "Doja Cat", "English", 0.8, 0.7, "Pop/R&B", "https://www.youtube.com/watch?v=ZOyW9J0xYGM"),
    # ("Fancy Like", "Walker Hayes", "English", 0.8, 0.7, "Country", "https://www.youtube.com/watch?v=aG3sQ137w7U"),
    # ("My Universe", "Coldplay X BTS", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=3YqPKLZF_Es"),
    # ("You Right", "Doja Cat, The Weeknd", "English", 0.6, 0.7, "R&B", "https://www.youtube.com/watch?v=wX-r-mGgVdM"),
    # ("Billie Eilish.", "Armani White", "English", 0.85, 0.9, "Hip-Hop", "https://www.youtube.com/watch?v=G_Hl4VqA8tA"),
    # ("As It Was", "Harry Styles", "English", 0.8, 0.7, "Pop", "https://www.youtube.com/watch?v=H5v3kku4CgY"),
    # ("Rich Flex", "Drake & 21 Savage", "English", 0.6, 0.8, "Hip-Hop", "https://www.youtube.com/watch?v=I4Q117P58Sg"),
    # ("Kill Bill", "SZA", "English", 0.5, 0.6, "R&B", "https://www.youtube.com/watch?v=W0-F-B-oJ4Y"),
    # ("Calm Down", "Rema, Selena Gomez", "English", 0.9, 0.75, "Afrobeats", "https://www.youtube.com/watch?v=N8TadE1Q2gQ"),
    # ("I'm Good (Blue)", "David Guetta, Bebe Rexha", "English", 0.95, 0.9, "Dance", "https://www.youtube.com/watch?v=d_Hl4VqA8tA"),
    # ("Flowers", "Miley Cyrus", "English", 0.8, 0.6, "Pop", "https://www.youtube.com/watch?v=G7KNm9TZsrg"),
    # ("Made You Look", "Meghan Trainor", "English", 0.9, 0.8, "Pop", "https://www.youtube.com/watch?v=YwL34WJ1f-I"),
    # ("Under The Influence", "Chris Brown", "English", 0.7, 0.75, "R&B", "https://www.youtube.com/watch?v=1F3gGvB6pYI"),
    # ("Celestial", "Ed Sheeran", "English", 0.8, 0.65, "Pop", "https://www.youtube.com/watch?v=S2fFtyy-8m8"),
    # ("Forget Me", "Lewis Capaldi", "English", 0.25, 0.35, "Ballad", "https://www.youtube.com/watch?v=wz-m8T3Lq8I"),
    # ("Victoria’s Secret", "JAX", "English", 0.7, 0.6, "Pop", "https://www.youtube.com/watch?v=j1aVp4E-EAE"),
    # ("Vampire", "Olivia Rodrigo", "English", 0.9, 0.9, "Pop-Punk", "https://www.youtube.com/watch?v=hB0Pq03zJkE"),
    # ("Dil Deewana (Maine Pyar Kiya)", "S. P. Balasubrahmanyam, Lata Mangeshkar", "Hindi", 0.8, 0.6, "Romantic", "https://www.youtube.com/watch?v=H43hX2I2I8o"),
    # ("Pehla Nasha (Jo Jeeta Wohi Sikandar)", "Udit Narayan, Sadhana Sargam", "Hindi", 0.8, 0.7, "Romantic", "https://www.youtube.com/watch?v=r_O_gL9hQpQ"),
    # ("Tujhe Dekha Toh (Dilwale Dulhania Le Jayenge)", "Lata Mangeshkar, Kumar Sanu", "Hindi", 0.9, 0.6, "Romantic", "https://www.youtube.com/watch?v=C25KkSgJz4c"),
    # ("Chura Ke Dil Mera (Main Khiladi Tu Anari)", "Kumar Sanu, Alka Yagnik", "Hindi", 0.85, 0.75, "Dance/Romantic", "https://www.youtube.com/watch?v=L2Gj3oP-GqQ"),
    # ("Dil Tera Deewana (Anari)", "Udit Narayan, Alka Yagnik", "Hindi", 0.7, 0.5, "Romantic", "https://www.youtube.com/watch?v=M5yBwJ-V69o"),
    # ("Humma Humma (Bombay)", "Remo Fernandes, Kavita Krishnamurthy", "Hindi", 0.7, 0.8, "Pop/Dance", "https://www.youtube.com/watch?v=1F3gGvB6pYI"),
    # ("Tu Hi Re (Bombay)", "Hariharan, Kavita Krishnamurthy", "Hindi", 0.5, 0.4, "Romantic/Emotional", "https://www.youtube.com/watch?v=x-w26_x-g00"),
    # ("Jaanam Samjha Karo (Jaanam Samjha Karo)", "Anu Malik, Hema Sardesai", "Hindi", 0.8, 0.7, "Pop/Romantic", "https://www.youtube.com/watch?v=Xh0mGz-N8_k"),
    # ("Dheere Dheere Se Meri Zindagi Mein Aana (Aashiqui)", "Kumar Sanu, Anuradha Paudwal", "Hindi", 0.7, 0.5, "Romantic", "https://www.youtube.com/watch?v=L013Vj7_z_A"),
    # ("Kuch Kuch Hota Hai (Kuch Kuch Hota Hai)", "Udit Narayan, Alka Yagnik", "Hindi", 0.85, 0.65, "Romantic", "https://www.youtube.com/watch?v=K37C4W9m4bM"),
    # ("Dil To Pagal Hai (Dil To Pagal Hai)", "Lata Mangeshkar, Udit Narayan", "Hindi", 0.7, 0.6, "Romantic", "https://www.youtube.com/watch?v=07-5eF6j0j0"),
    # ("Chand Sifarish (Fanaa)", "Shaan, Kailash Kher", "Hindi", 0.7, 0.6, "Romantic/Pop", "https://www.youtube.com/watch?v=X0yKk0d-Xj0"),
    # ("Kabhi Alvida Naa Kehna (Kabhi Alvida Naa Kehna)", "Sonu Nigam, Alka Yagnik", "Hindi", 0.4, 0.3, "Emotional/Romantic", "https://www.youtube.com/watch?v=Vz2Xw3-x-g0"),
    # ("Kaho Naa Pyaar Hai (Kaho Naa Pyaar Hai)", "Lucky Ali, Asha Bhosle", "Hindi", 0.85, 0.7, "Pop/Romantic", "https://www.youtube.com/watch?v=ZfM6k0f5kGk"),
    # ("Ek Pal Ka Jeena (Kaho Naa Pyaar Hai)", "Lucky Ali", "Hindi", 0.8, 0.8, "Pop/Dance", "https://www.youtube.com/watch?v=E_Pq5r9fR0Q"),
    # ("Woh Ladki Hai Kahan (Dil Chahta Hai)", "Shankar Mahadevan, Kavita Krishnamurthy", "Hindi", 0.9, 0.8, "Dance/Pop", "https://www.youtube.com/watch?v=p1t3Q7kS3fA"),
    # ("Bole Chudiyan (Kabhi Khushi Kabhie Gham...)", "Amitabh Bachchan, Sonu Nigam, Alka Yagnik, Kavita Krishnamurthy, Udit Narayan", "Hindi", 0.95, 0.85, "Dance/Wedding", "https://www.youtube.com/watch?v=U0-F-B-oJ4Y"),
    # ("Gori Gori (Main Hoon Na)", "KK, Shreya Ghoshal, Sunidhi Chauhan, Sukhwinder Singh", "Hindi", 0.9, 0.75, "Pop/Dance", "https://www.youtube.com/watch?v=d_Hl4VqA8tA"),
    # ("Main Hoon Na (Main Hoon Na)", "Sonu Nigam, Shreya Ghoshal", "Hindi", 0.8, 0.6, "Patriotic/Pop", "https://www.youtube.com/watch?v=G7KNm9TZsrg"),
    # ("Aankhon Mein Tera Hi Chehra (Album - Aryans)", "Aryans", "Hindi", 0.8, 0.6, "Pop/Romantic", "https://www.youtube.com/watch?v=s0K6PqJq1yU"),
    # ("Dooba Dooba (Album - Boondein)", "Silk Route", "Hindi", 0.7, 0.5, "Indie Pop", "https://www.youtube.com/watch?v=mYlS2kQ7R7g"),
    # ("O Sanam (Album - Sunoh)", "Lucky Ali", "Hindi", 0.6, 0.4, "Indie Pop", "https://www.youtube.com/watch?v=yYyB-m9xS_A"),
    # ("Pal Pal Har Pal (Lage Raho Munna Bhai)", "Sonu Nigam, Shreya Ghoshal", "Hindi", 0.8, 0.5, "Romantic", "https://www.youtube.com/watch?v=g8fK84b7B_c"),
    # ("Jashn-E-Baharaa (Jodhaa Akbar)", "Javed Ali", "Hindi", 0.6, 0.4, "Romantic/Classical", "https://www.youtube.com/watch?v=H74Yx-oF2R0"),
    # ("Tere Naina (Chandni Chowk To China)", "Shaan, Shreya Ghoshal", "Hindi", 0.7, 0.5, "Romantic", "https://www.youtube.com/watch?v=M5yBwJ-V69o"),
    # ("Tu Jaane Na (Ajab Prem Ki Ghazab Kahani)", "Atif Aslam", "Hindi", 0.4, 0.3, "Romantic/Sad", "https://www.youtube.com/watch?v=H4uW0QoRkmc"),
    # ("Jogi Mahi (Bachna Ae Haseeno)", "Sukhwinder Singh, Udit Narayan, Alka Yagnik, Sonu Nigam", "Hindi", 0.85, 0.7, "Dance/Wedding", "https://www.youtube.com/watch?v=Xz29g41Ld0w"),
    # ("Chalte Chalte (Chalte Chalte)", "Sonu Nigam, Alka Yagnik", "Hindi", 0.7, 0.5, "Romantic", "https://www.youtube.com/watch?v=tQ0yjYUFKAE"),
    # ("Maahi Ve (Kaante)", "Hariharan, Sadhana Sargam, Sukhwinder Singh, Madhushree", "Hindi", 0.8, 0.6, "Wedding/Emotional", "https://www.youtube.com/watch?v=t_u_q_Nq4-0"),
    # ("Tere Naam (Tere Naam)", "Udit Narayan", "Hindi", 0.4, 0.4, "Romantic/Emotional", "https://www.youtube.com/watch?v=tP4-o-dD8C8"),
    # ("Aankhein Khuli (Mohabbatein)", "Lata Mangeshkar, Udit Narayan, Kumar Sanu, Sonu Nigam, Jaspinder Narula", "Hindi", 0.7, 0.6, "Romantic/Pop", "https://www.youtube.com/watch?v=e_g72C9a2L0"),
    # ("Suno Na Suno Na (Chalte Chalte)", "Abhijeet Bhattacharya", "Hindi", 0.75, 0.5, "Romantic", "https://www.youtube.com/watch?v=q0D84G1x-1g"),
    # ("Maa Da Laadla (Dostana)", "Saleem Shahzada", "Hindi", 0.8, 0.7, "Pop/Dance", "https://www.youtube.com/watch?v=Q8lV3r1Q8tY"),
    # ("Mere Khwabon Mein (Dilwale Dulhania Le Jayenge)", "Lata Mangeshkar", "Hindi", 0.7, 0.5, "Romantic/Classic", "https://www.youtube.com/watch?v=jW0D2M3mXhE"),
    # ("Dil Laga Liya (Dil Hai Tumhaara)", "Alka Yagnik, Udit Narayan", "Hindi", 0.85, 0.6, "Romantic", "https://www.youtube.com/watch?v=fXvj51jB6nU"),
    # ("Tumse Milke Dil Ka (Main Hoon Na)", "Sonu Nigam, Aftab Sabri, Hashim Sabri", "Hindi", 0.8, 0.7, "Pop/Dance", "https://www.youtube.com/watch?v=d_Hl4VqA8tA"),
    # ("Oh My Darling (Mujhse Dosti Karoge!)", "Alisha Chinai, Sonu Nigam", "Hindi", 0.7, 0.6, "Romantic/Pop", "https://www.youtube.com/watch?v=C25KkSgJz4c"),
    # ("Taal Se Taal Mila (Taal)", "Alka Yagnik, Udit Narayan", "Hindi", 0.75, 0.65, "Classical/Romantic", "https://www.youtube.com/watch?v=1F3gGvB6pYI"),
    # ("Dil Ne Yeh Kaha Hai Dil Se (Dhadkan)", "Alka Yagnik, Kumar Sanu, Udit Narayan", "Hindi", 0.7, 0.5, "Romantic", "https://www.youtube.com/watch?v=L2Gj3oP-GqQ")


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
