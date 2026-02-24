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
FFMPEG_PATH = r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
#  r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"
# FFMPEG_PATH = r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"


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


# ("Sibelius – Symphony No.1 – I. Andante", "Jean Sibelius", "English", 0.6, 0.68, "Classical", ["https://www.youtube.com/watch?v=dCIw_4oJ4Gg"]),
# ("Sibelius – Symphony No.5 – I. Tempo molto", "Jean Sibelius", "English", 0.72, 0.74, "Classical", ["https://www.youtube.com/watch?v=FMQRsWN-uGc"]),
# ("Sibelius – Finlandia", "Jean Sibelius", "English", 0.78, 0.8, "Classical", ["https://www.youtube.com/watch?v=fE0RbPsC9uE"]),
# ("Sibelius – Violin Concerto – I. Allegro", "Jean Sibelius", "English", 0.66, 0.72, "Classical", ["https://www.youtube.com/watch?v=3u-unvYedx8"]),
# ("Sibelius – Valse Triste", "Jean Sibelius", "English", 0.54, 0.5, "Classical", ["https://www.youtube.com/watch?v=5Ls8-pk4IS4"]),
# ("Stravinsky – The Firebird Suite", "Igor Stravinsky", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=kd1xYKGnOEw"]),
# ("Stravinsky – Petrushka", "Igor Stravinsky", "English", 0.7, 0.84, "Classical", ["https://www.youtube.com/watch?v=esD90diWZds"]),
# ("Stravinsky – The Rite of Spring", "Igor Stravinsky", "English", 0.52, 0.92, "Classical", ["https://www.youtube.com/watch?v=EkwqPJZe8ms"]),
# ("Stravinsky – Pulcinella Suite", "Igor Stravinsky", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=J_E7w9P8x5o"]),
# ("Stravinsky – Symphony of Psalms", "Igor Stravinsky", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=DqWZGUO_eoc"]),
# ("Ravel – Pavane for a Dead Princess", "Maurice Ravel", "English", 0.62, 0.4, "Classical", ["https://www.youtube.com/watch?v=DVtNt-6OTM8"]),
# ("Ravel – Daphnis et Chloé Suite No.2", "Maurice Ravel", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=14OM6Ysnk6M"]),
# ("Ravel – Piano Concerto in G – II. Adagio", "Maurice Ravel", "English", 0.66, 0.48, "Classical", ["https://www.youtube.com/watch?v=NRTWLQ4nI6Q"]),
# ("Ravel – Le Tombeau de Couperin", "Maurice Ravel", "English", 0.72, 0.58, "Classical", ["https://www.youtube.com/watch?v=Wj4HliLWVxQ"]),
# ("Ravel – La Valse", "Maurice Ravel", "English", 0.64, 0.76, "Classical", ["https://www.youtube.com/watch?v=TMSgWhIENSk"]),
# ("Prokofiev – Romeo and Juliet Suite", "Sergei Prokofiev", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=7qqrIusxVAI"]),
# ("Prokofiev – Symphony No.1 Classical – I. Allegro", "Sergei Prokofiev", "English", 0.82, 0.72, "Classical", ["https://www.youtube.com/watch?v=jIQEvI9bqEE"]),
# ("Prokofiev – Piano Concerto No.3 – I. Andante", "Sergei Prokofiev", "English", 0.74, 0.82, "Classical", ["https://www.youtube.com/watch?v=BS0SwRoYAW0"]),
# ("Prokofiev – Lieutenant Kijé Suite", "Sergei Prokofiev", "English", 0.76, 0.68, "Classical", ["https://www.youtube.com/watch?v=DkoKGA-30cY"]),
# ("Prokofiev – Alexander Nevsky", "Sergei Prokofiev", "English", 0.64, 0.76, "Classical", ["https://www.youtube.com/watch?v=xyDKezDLGTM"]),
# ("Shostakovich – Symphony No.5 – I. Moderato", "Dmitri Shostakovich", "English", 0.58, 0.82, "Classical", ["https://www.youtube.com/watch?v=cg0M4LzEITQ"]),
# ("Shostakovich – Symphony No.7 Leningrad – I. Allegretto", "Dmitri Shostakovich", "English", 0.62, 0.84, "Classical", ["https://www.youtube.com/watch?v=GB3zR_X25UU"]),
# ("Shostakovich – String Quartet No.8", "Dmitri Shostakovich", "English", 0.54, 0.7, "Classical", ["https://www.youtube.com/watch?v=-0nKJoZY64A"]),
# ("Shostakovich – Jazz Suite No.2 Waltz", "Dmitri Shostakovich", "English", 0.74, 0.66, "Classical", ["https://www.youtube.com/watch?v=O4gQEslOKjI"]),
# ("Shostakovich – Piano Concerto No.2 – II. Andante", "Dmitri Shostakovich", "English", 0.68, 0.5, "Classical", ["https://www.youtube.com/watch?v=1DFqVqApms8"]),
# ("Bartók – Concerto for Orchestra – I. Introduzione", "Béla Bartók", "English", 0.72, 0.82, "Classical", ["https://www.youtube.com/watch?v=04M2pQ0jOeE"]),
# ("Bartók – Romanian Folk Dances", "Béla Bartók", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=Z50Ooqv1GFg"]),
# ("Bartók – Music for Strings, Percussion and Celesta", "Béla Bartók", "English", 0.6, 0.78, "Classical", ["https://www.youtube.com/watch?v=2EsNGS9vYe8"]),
# ("Bartók – Bluebeard's Castle", "Béla Bartók", "English", 0.56, 0.72, "Classical", ["https://www.youtube.com/watch?v=GoImjQOEp-Q"]),
# ("Bartók – Piano Concerto No.2 – I. Allegro", "Béla Bartók", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=Qr_JUGdoVSY"]),
# ("Elgar – Enigma Variations", "Edward Elgar", "English", 0.74, 0.68, "Classical", ["https://www.youtube.com/watch?v=vLNLvcBmoqo"]),
# ("Elgar – Pomp and Circumstance No.1", "Edward Elgar", "English", 0.88, 0.82, "Classical", ["https://www.youtube.com/watch?v=Q0PHWKRFgZ0"]),
# ("Elgar – Cello Concerto – I. Adagio", "Edward Elgar", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=8V41D0Uczf0"]),
# ("Elgar – Violin Concerto", "Edward Elgar", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=JCT3NQKstxw"]),
# ("Elgar – Serenade for Strings", "Edward Elgar", "English", 0.78, 0.56, "Classical", ["https://www.youtube.com/watch?v=f4XK0oF88hc"]),
# ("Vaughan Williams – The Lark Ascending", "Ralph Vaughan Williams", "English", 0.88, 0.46, "Classical", ["https://www.youtube.com/watch?v=ZR2JlDnT2l8"]),
# ("Vaughan Williams – Fantasia on Greensleeves", "Ralph Vaughan Williams", "English", 0.82, 0.54, "Classical", ["https://www.youtube.com/watch?v=oWz-Hfw4fnk"]),
# ("Vaughan Williams – Norfolk Rhapsody", "Ralph Vaughan Williams", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=5DeT3DkyXc8"]),
# ("Vaughan Williams – Symphony No.5 – I. Preludio", "Ralph Vaughan Williams", "English", 0.7, 0.6, "Classical", ["https://www.youtube.com/watch?v=oJQ60haeZ_4"]),
# ("Vaughan Williams – Sea Symphony", "Ralph Vaughan Williams", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=2kR3FzJ1Hh4"]),
# ("Britten – Young Person's Guide to the Orchestra", "Benjamin Britten", "English", 0.86, 0.74, "Classical", ["https://www.youtube.com/watch?v=4vbvhU22uAM"]),
# ("Britten – War Requiem", "Benjamin Britten", "English", 0.52, 0.72, "Classical", ["https://www.youtube.com/watch?v=625WOYzdvFw"]),
# ("Britten – Simple Symphony", "Benjamin Britten", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=cSvSOWbNNMI"]),
# ("Britten – Serenade for Tenor, Horn and Strings", "Benjamin Britten", "English", 0.64, 0.56, "Classical", ["https://www.youtube.com/watch?v=gcnw-jBCpPw"]),
# ("Britten – Violin Concerto", "Benjamin Britten", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=dDTIae06t6Y"]),
# ("Copland – Appalachian Spring", "Aaron Copland", "English", 0.84, 0.6, "Classical", ["https://www.youtube.com/watch?v=TXV8yO1FucA"]),
# ("Copland – Fanfare for the Common Man", "Aaron Copland", "English", 0.88, 0.76, "Classical", ["https://www.youtube.com/watch?v=ZdqjcMmjeaA"]),
# ("Copland – Rodeo", "Aaron Copland", "English", 0.86, 0.72, "Classical", ["https://www.youtube.com/watch?v=du4DrdGp9vM"]),
# ("Copland – Billy the Kid Suite", "Aaron Copland", "English", 0.8, 0.7, "Classical", ["https://www.youtube.com/watch?v=wdhE0R4qGX0"]),
# ("Copland – El Salón México", "Aaron Copland", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=WoILPBDsfvI"]),
# ("Gershwin – Rhapsody in Blue", "George Gershwin", "English", 0.86, 0.78, "Classical", ["https://www.youtube.com/watch?v=cH2PH0auTUU"]),
# ("Gershwin – An American in Paris", "George Gershwin", "English", 0.84, 0.74, "Classical", ["https://www.youtube.com/watch?v=K4I2OzMltM4"]),
# ("Gershwin – Piano Concerto in F", "George Gershwin", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=MDxKtkkbE7w"]),
# ("Gershwin – Cuban Overture", "George Gershwin", "English", 0.82, 0.78, "Classical", ["https://www.youtube.com/watch?v=dsGbFKszo7I"]),
# ("Gershwin – Variations on 'I Got Rhythm'", "George Gershwin", "English", 0.88, 0.82, "Classical", ["https://www.youtube.com/watch?v=HV62aaWu590"]),
# ("Handel – Water Music Suite", "George Frideric Handel", "English", 0.84, 0.68, "Classical", ["https://www.youtube.com/watch?v=EVAB2z1RPu4"]),
# ("Handel – Music for the Royal Fireworks", "George Frideric Handel", "English", 0.86, 0.72, "Classical", ["https://www.youtube.com/watch?v=EkttBYzD-jY"]),
# ("Handel – Messiah – Hallelujah Chorus", "George Frideric Handel", "English", 0.92, 0.74, "Classical", ["https://www.youtube.com/watch?v=IUZEtVbJT5c"]),
# ("Handel – Concerto Grosso Op.6 No.5", "George Frideric Handel", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=4ISLEvMU9jY"]),
# ("Handel – Sarabande in D Minor", "George Frideric Handel", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=JSAd3NpDi6Q"]),
# ("Haydn – Symphony No.94 Surprise – I. Adagio", "Joseph Haydn", "English", 0.86, 0.68, "Classical", ["https://www.youtube.com/watch?v=8Su-ACFnUCs"]),
# ("Haydn – Symphony No.101 Clock – I. Adagio", "Joseph Haydn", "English", 0.88, 0.7, "Classical", ["https://www.youtube.com/watch?v=qPx4m5FJRaE"]),
# ("Haydn – Symphony No.104 London – I. Adagio", "Joseph Haydn", "English", 0.84, 0.72, "Classical", ["https://www.youtube.com/watch?v=OitPLIowJ70"]),
# ("Haydn – The Creation", "Joseph Haydn", "English", 0.82, 0.6, "Classical", ["https://www.youtube.com/watch?v=EuIs7R2BpvQ"]),
# ("Haydn – Trumpet Concerto – I. Allegro", "Joseph Haydn", "English", 0.9, 0.74, "Classical", ["https://www.youtube.com/watch?v=NHjgSiTBddM"]),
# ("Fauré – Requiem – Pie Jesu", "Gabriel Fauré", "English", 0.62, 0.4, "Classical", ["https://www.youtube.com/watch?v=o9al6HNOgSo"]),
# ("Fauré – Cantique de Jean Racine", "Gabriel Fauré", "English", 0.7, 0.44, "Classical", ["https://www.youtube.com/watch?v=g16zSj6Ynko"]),
# ("Fauré – Pavane Op.50", "Gabriel Fauré", "English", 0.68, 0.46, "Classical", ["https://www.youtube.com/watch?v=mpgyTl8yqbw"]),
# ("Fauré – Nocturne Op.33 No.1", "Gabriel Fauré", "English", 0.66, 0.42, "Classical", ["https://www.youtube.com/watch?v=P1CHImcrSzU"]),
# ("Fauré – Requiem – Libera Me", "Gabriel Fauré", "English", 0.6, 0.38, "Classical", ["https://www.youtube.com/watch?v=VMdoq7uE74A"]),
# ("Paganini – Caprice No.24", "Niccolò Paganini", "English", 0.82, 0.9, "Classical", ["https://www.youtube.com/watch?v=WMLoBXgPil8"]),
# ("Paganini – Violin Concerto No.1 – I. Allegro", "Niccolò Paganini", "English", 0.8, 0.86, "Classical", ["https://www.youtube.com/watch?v=vadJp7vokL8"]),
# ("Paganini – Moto Perpetuo", "Niccolò Paganini", "English", 0.88, 0.92, "Classical", ["https://www.youtube.com/watch?v=D-TAO7U6rtg"]),
# ("Paganini – La Campanella", "Niccolò Paganini", "English", 0.84, 0.88, "Classical", ["https://www.youtube.com/watch?v=6ruHDWSNvB8"]),
# ("Paganini – Variations on 'God Save the King'", "Niccolò Paganini", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=GflrUhvIWU4"]),
# ("Albinoni – Adagio in G Minor", "Tomaso Albinoni", "English", 0.6, 0.46, "Classical", ["https://www.youtube.com/watch?v=XMbvcp480Y4"]),
# ("Albinoni – Oboe Concerto in D Minor – I. Allegro", "Tomaso Albinoni", "English", 0.74, 0.66, "Classical", ["https://www.youtube.com/watch?v=dLxJLrjvl8A"]),
# ("Albinoni – Sonata in G Minor", "Tomaso Albinoni", "English", 0.68, 0.54, "Classical", ["https://www.youtube.com/watch?v=2DL5YpEobO0"]),
# ("Albinoni – Sinfonia in C Major", "Tomaso Albinoni", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=2DL5YpEobO0"]),
# ("Albinoni – Concerto Op.9 No.2", "Tomaso Albinoni", "English", 0.72, 0.64, "Classical", ["https://www.youtube.com/watch?v=CRw9dS05gmw"]),
# ("Rachmaninoff – Piano Concerto No.1 – I. Vivace", "Sergei Rachmaninoff", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=y6EX3t2Mdnw"]),
# ("Rachmaninoff – Piano Concerto No.2 – I. Moderato", "Sergei Rachmaninoff", "English", 0.72, 0.82, "Classical", ["https://www.youtube.com/watch?v=r-SZ_e5GWMc"]),
# ("Rachmaninoff – Piano Concerto No.3 – I. Allegro", "Sergei Rachmaninoff", "English", 0.74, 0.86, "Classical", ["https://www.youtube.com/watch?v=iU55znLN4_U"]),
# ("Rachmaninoff – Rhapsody on a Theme of Paganini", "Sergei Rachmaninoff", "English", 0.68, 0.76, "Classical", ["https://www.youtube.com/watch?v=ppJ5uITLECE"]),
# ("Rachmaninoff – Vocalise", "Sergei Rachmaninoff", "English", 0.66, 0.44, "Classical", ["https://www.youtube.com/watch?v=895pPv4l_9Q"]),
# ("Borodin – Symphony No.1 – I. Allegro", "Alexander Borodin", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=PIil9TONfu0"]),
# ("Borodin – Polovtsian Dances", "Alexander Borodin", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=kqKclPhsK0o"]),
# ("Borodin – String Quartet No.2", "Alexander Borodin", "English", 0.72, 0.54, "Classical", ["https://www.youtube.com/watch?v=2YAzUC6LzNk"]),
# ("Borodin – Prince Igor Overture", "Alexander Borodin", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=cDJkzgOpdb0"]),
# ("Borodin – In the Steppes of Central Asia", "Alexander Borodin", "English", 0.7, 0.52, "Classical", ["https://www.youtube.com/watch?v=g4tlQxaHetI"]),
# ("Rimsky-Korsakov – Scheherazade", "Nikolai Rimsky-Korsakov", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=zY4w4_W30aQ"]),
# ("Rimsky-Korsakov – Capriccio Espagnol", "Nikolai Rimsky-Korsakov", "English", 0.84, 0.78, "Classical", ["https://www.youtube.com/watch?v=RY2Bt0TppKw"]),
# ("Rimsky-Korsakov – Russian Easter Overture", "Nikolai Rimsky-Korsakov", "English", 0.82, 0.8, "Classical", ["https://www.youtube.com/watch?v=z4e8CvxV4Ho"]),
# ("Rimsky-Korsakov – Antar Symphony", "Nikolai Rimsky-Korsakov", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=NiTInQcjy4o"]),
# ("Rimsky-Korsakov – Flight of the Bumblebee", "Nikolai Rimsky-Korsakov", "English", 0.86, 0.94, "Classical", ["https://www.youtube.com/watch?v=59QXMCsx_5E"]),
# ("Villa-Lobos – Bachianas Brasileiras No.5", "Heitor Villa-Lobos", "English", 0.84, 0.6, "Classical", ["https://www.youtube.com/watch?v=pUCuEd1tjCg"]),
# ("Villa-Lobos – Chôros No.10", "Heitor Villa-Lobos", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=pXR7C1p1Sbk"]),
# ("Villa-Lobos – Uirapuru", "Heitor Villa-Lobos", "English", 0.74, 0.7, "Classical", ["https://www.youtube.com/watch?v=Zs7VSsI3O9Q"]),
# ("Villa-Lobos – Symphony No.1 – I. Allegro", "Heitor Villa-Lobos", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=4QIy8dnkHdw"]),
# ("Villa-Lobos – Floresta do Amazonas", "Heitor Villa-Lobos", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=eaZuti6fpTA"]),
# ("Khachaturian – Gayane Suite", "Aram Khachaturian", "English", 0.84, 0.82, "Classical", ["https://www.youtube.com/watch?v=8ob0nRhSKAw"]),
# ("Khachaturian – Masquerade Suite", "Aram Khachaturian", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=Y7JxLgf3wLM"]),
# ("Khachaturian – Violin Concerto – I. Allegro", "Aram Khachaturian", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=rixdcNMGwWQ"]),
# ("Khachaturian – Symphony No.2 – I. Allegro", "Aram Khachaturian", "English", 0.7, 0.8, "Classical", ["https://www.youtube.com/watch?v=RylaqvH2bgQ"]),
# ("Khachaturian – Adagio of Spartacus and Phrygia", "Aram Khachaturian", "English", 0.68, 0.5, "Classical", ["https://www.youtube.com/watch?v=fnFeXa0nGao"]),
# ("Glazunov – Symphony No.5 – I. Moderato", "Alexander Glazunov", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=7bp7cAsXevI"]),
# ("Glazunov – Violin Concerto", "Alexander Glazunov", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=ZbQ1f4NNWkU"]),
# ("Glazunov – Raymonda Suite", "Alexander Glazunov", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=wp0zVYk-2Qo"]),
# ("Glazunov – The Seasons Ballet", "Alexander Glazunov", "English", 0.8, 0.68, "Classical", ["https://www.youtube.com/watch?v=MquuAy4041c"]),
# ("Glazunov – Symphony No.1 – I. Allegro", "Alexander Glazunov", "English", 0.74, 0.7, "Classical", ["https://www.youtube.com/watch?v=sdoV1N71kTQ"]),
# ("Myaskovsky – Symphony No.6 – I. Allegro", "Nikolai Myaskovsky", "English", 0.58, 0.7, "Classical", ["https://www.youtube.com/watch?v=MEughaH6Zww"]),
# ("Myaskovsky – Symphony No.21", "Nikolai Myaskovsky", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=c0hU4wQCCQM"]),
# ("Myaskovsky – Cello Concerto", "Nikolai Myaskovsky", "English", 0.62, 0.64, "Classical", ["https://www.youtube.com/watch?v=pqNWSM7fcCE"]),
# ("Myaskovsky – Sinfonietta", "Nikolai Myaskovsky", "English", 0.7, 0.6, "Classical", ["https://www.youtube.com/watch?v=4Qcmip_2buU"]),

# ("Clementi – Toccata in C Major", "Muzio Clementi", "English", 0.78, 0.76, "Classical", ["https://www.youtube.com/watch?v=3OZAOFDB3Qs"]),
# ("Hummel – Trumpet Concerto in E", "Johann Nepomuk Hummel", "English", 0.88, 0.74, "Classical", ["https://www.youtube.com/watch?v=902St2UAmfA"]),
# ("Hummel – Piano Concerto in A Minor", "Johann Nepomuk Hummel", "English", 0.8, 0.72, "Classical", ["https://www.youtube.com/watch?v=qH9nBvCzozc"]),
# ("Hummel – Septet in D Minor", "Johann Nepomuk Hummel", "English", 0.72, 0.66, "Classical", ["https://www.youtube.com/watch?v=3rHkyAzcPO8"]),
# ("Hummel – Mandolin Concerto", "Johann Nepomuk Hummel", "English", 0.84, 0.7, "Classical", ["https://www.youtube.com/watch?v=lV-pdJK8iXE"]),
# ("Hummel – Fantasie in E-flat", "Johann Nepomuk Hummel", "English", 0.78, 0.68, "Classical", ["https://www.youtube.com/watch?v=Nzrz5LyWMY4"]),
# ("Spohr – Violin Concerto No.8", "Louis Spohr", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=muCq10WC2G8"]),
# ("Spohr – Clarinet Concerto No.1", "Louis Spohr", "English", 0.74, 0.68, "Classical", ["https://www.youtube.com/watch?v=xxAzo2LC1Ck"]),
# ("Spohr – Symphony No.5", "Louis Spohr", "English", 0.7, 0.66, "Classical", ["https://www.youtube.com/watch?v=4hP88S1QenY"]),
# ("Spohr – Nonet Op.31", "Louis Spohr", "English", 0.72, 0.64, "Classical", ["https://www.youtube.com/watch?v=StIxuL01mj0"]),
# ("Spohr – Faust Overture", "Louis Spohr", "English", 0.68, 0.72, "Classical", ["https://www.youtube.com/watch?v=X6bPXTNRx6s"]),
# ("Mendelssohn – Violin Concerto in E Minor", "Felix Mendelssohn", "English", 0.8, 0.74, "Classical", ["https://www.youtube.com/watch?v=I03Hs6dwj7E"]),
# ("Mendelssohn – Italian Symphony – I. Allegro", "Felix Mendelssohn", "English", 0.86, 0.76, "Classical", ["https://www.youtube.com/watch?v=_HX_jF1_Tgc"]),
# ("Mendelssohn – Hebrides Overture", "Felix Mendelssohn", "English", 0.74, 0.64, "Classical", ["https://www.youtube.com/watch?v=MdQyN7MYSN8"]),
# ("Mendelssohn – Songs Without Words", "Felix Mendelssohn", "English", 0.78, 0.52, "Classical", ["https://www.youtube.com/watch?v=mGaruN5VZPA"]),
# ("Mendelssohn – Octet in E-flat Major", "Felix Mendelssohn", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=Vw1kcQ-QbZw"]),
# ("Schumann – Piano Concerto in A Minor", "Robert Schumann", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=fWDrJT0s1s8"]),
# ("Schumann – Kinderszenen", "Robert Schumann", "English", 0.74, 0.48, "Classical", ["https://www.youtube.com/watch?v=jnB51JbW2VQ"]),
# ("Schumann – Carnaval", "Robert Schumann", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=LNo2aiKV-a0"]),
# ("Schumann – Symphony No.3 Rhenish", "Robert Schumann", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=3lRdCGIp-rg"]),
# ("Schumann – Dichterliebe", "Robert Schumann", "English", 0.64, 0.5, "Classical", ["https://www.youtube.com/watch?v=0p0f_g4PMwc"]),
# ("Bizet – Carmen Suite No.1", "Georges Bizet", "English", 0.84, 0.78, "Classical", ["https://www.youtube.com/watch?v=qvit0kufwZ4"]),
# ("Bizet – Carmen Habanera", "Georges Bizet", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=lvxEN_hCefU"]),
# ("Bizet – L'Arlésienne Suite", "Georges Bizet", "English", 0.8, 0.74, "Classical", ["https://www.youtube.com/watch?v=3cgNtNt8I0Y"]),
# ("Bizet – Symphony in C", "Georges Bizet", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=STvVanu1fQU"]),
# ("Bizet – Jeux d'enfants", "Georges Bizet", "English", 0.76, 0.64, "Classical", ["https://www.youtube.com/watch?v=AABsvKMCaQU"]),
# ("Delibes – Lakmé Flower Duet", "Léo Delibes", "English", 0.88, 0.52, "Classical", ["https://www.youtube.com/watch?v=C1ZL5AxmK_A"]),
# ("Delibes – Coppélia Waltz", "Léo Delibes", "English", 0.86, 0.66, "Classical", ["https://www.youtube.com/watch?v=dixdXcZouZ8"]),
# ("Delibes – Sylvia Pizzicato", "Léo Delibes", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=8RhQkxkXkP0"]),
# ("Delibes – Le Roi s'amuse", "Léo Delibes", "English", 0.74, 0.62, "Classical", ["https://www.youtube.com/watch?v=QYun5LRGce4"]),
# ("Delibes – Lakmé Bell Song", "Léo Delibes", "English", 0.84, 0.72, "Classical", ["https://www.youtube.com/watch?v=gMO0KFL3E58"]),




("Speedball", "John Zorn", "English", 3.08, 5.87, "Jazz", ["https://www.youtube.com/watch?v=TNi5F9JqhuI"]),
("They All Laughed", "Tony Bennett", "English", 6.33, 5.14, "Jazz", ["https://www.youtube.com/watch?v=MaoxQxT2bw4"]),
("Parabola", "Jawbreaker", "English", 4.56, 4.67, "Jazz", ["https://www.youtube.com/watch?v=Bd36WAodbgc"]),
("Fly Me to the Moon", "Nat King Cole", "English", 4.37, 4.72, "Jazz", ["https://www.youtube.com/watch?v=KLvLo6-XOnI"]),
("Toto Dies", "Nellie McKay", "English", 5.17, 4.45, "Jazz", ["https://www.youtube.com/watch?v=4httaMdvYrY"]),
("Night and Day (Drunk)", "Kimza International", "English", 5.75, 4.66, "Jazz", [""]),
("Vanetihu (LP Version)", "Nina Simone", "English", 6.49, 5.17, "Jazz", ["https://www.youtube.com/watch?v=Bak81pz-Gjg"]),
("ライラニア", "志方あきこ", "English", 5.2, 4.11, "Jazz", ["https://www.youtube.com/watch?v=yRt9DHpvmL4"]),
("ナゾ？", "志方あきこ", "English", 5.11, 4.59, "Jazz", ["https://www.youtube.com/watch?v=EQzg0AFvZPI"]),
("sorriso～Memoria", "志方あきこ", "English", 5.55, 4.8, "Jazz", ["https://www.youtube.com/watch?v=lsILBPbz834"]),
("LAYLANIA", "志方あきこ", "English", 4.41, 5.26, "Jazz", ["https://www.youtube.com/watch?v=yRt9DHpvmL4"]),
("Farfalle", "志方あきこ", "English", 4.46, 4.66, "Jazz", ["https://www.youtube.com/watch?v=uOrrPmPifzc"]),
("The Trial Of Your Betrayal", "Shadow Keep", "English", 5.4, 4.53, "Jazz", ["https://www.youtube.com/watch?v=3X9sxYZl8U0"]),
("Mark Of The Usurper", "Shadow Keep", "English", 5.4, 4.53, "Jazz", ["https://www.youtube.com/watch?v=EJ5wdEbaPi8"]),
("Corruption Within", "Shadow Keep", "English", 5.4, 4.53, "Jazz", ["https://www.youtube.com/watch?v=vVOpoAvj5iQ"]),
("Arena", "志方あきこ", "English", 6.23, 4.56, "Jazz", ["https://www.youtube.com/watch?v=myP5iRYXBO8"]),
("Meta-morale", "Shadow Keep", "English", 5.31, 4.7, "Jazz", ["https://www.youtube.com/watch?v=zhIm0NwQ5no"]),
("As one", "Shikata", "English", 5.67, 4.91, "Jazz", ["https://www.youtube.com/watch?v=A4K-OKL6oSM"]),
("Incêndio", "Stupid", "English", 5.81, 4.88, "Jazz", ["https://www.youtube.com/watch?v=D-k_oSk3-6E"]),
("Farfalle〜異邦人〜", "志方あきこ", "English", 5.15, 4.75, "Jazz", ["https://www.youtube.com/watch?v=8V6uUXiCCTk"]),
("It's Karma Bitch", "Anjulie", "English", 4.95, 4.61, "Jazz", ["https://www.youtube.com/watch?v=qdHQ3hGPqug"]),
("Stupid", "Stupid", "English", 5.67, 4.91, "Jazz", ["https://www.youtube.com/watch?v=ipbfrPZd-x4"]),
("Calamity", "Marjolaine", "English", 5.81, 4.88, "Jazz", ["https://www.youtube.com/watch?v=LD0bsTjX1GQ"]),
("PHANTASMAGORIA", "Noriko Mitose, Akiko Shikata, Haruka Shimotsuki", "English", 5.66, 4.88, "Jazz", ["https://www.youtube.com/watch?v=eRkINJAoY6E"]),
("Constante Transição", "Stupid", "English", 5.81, 4.88, "Jazz", ["https://www.youtube.com/watch?v=rj3l7QXl-3U"]),
("Japanese Beer", "Stupid", "English", 5.83, 4.9, "Jazz", ["https://www.youtube.com/watch?v=v6WyOv3OP4A"]),
("Desce o céu", "Stupid", "English", 5.59, 4.88, "Jazz", ["https://www.youtube.com/watch?v=yGiEBzMWmFY"]),
("It Had Better Be Tonight [Meglio Stasera]", "Michael Bublé", "English", 4.39, 5.57, "Jazz", ["https://www.youtube.com/watch?v=uLy2Q8wZQYc"]),
("Always (Violin Interlude)", "Kalliope", "English", 5.55, 4.75, "Jazz", ["https://www.youtube.com/watch?v=lAk0MI93yMs"]),
("OPIUM II", "Ofasia", "English", 5.75, 4.87, "Jazz", ["https://www.youtube.com/watch?v=5I9csWWQpA4"]),
("Show Me Love", "Kalliope", "English", 5.75, 4.87, "Jazz", ["https://www.youtube.com/watch?v=esCIIk0OfSo"]),
("Orient Express", "Joe Zawinul", "English", 4.14, 6.62, "Jazz", ["https://www.youtube.com/watch?v=922LumI2ilo"]),
("Africa Con India", "Trilok Gurtu", "English", 4.14, 6.62, "Jazz", ["https://www.youtube.com/watch?v=czjOPlaiVZ8"]),
("And Still We Move", "Crissi Cochrane", "English", 6.49, 5.24, "Jazz", ["https://www.youtube.com/watch?v=uGDepxv7iX8"]),
("Moist", "Nastassja Kinski", "English", 6.35, 4.92, "Jazz", ["https://www.youtube.com/watch?v=OoLZfMogO88"]),
("Would You", "Touch and Go", "English", 5.82, 6.43, "Jazz", ["https://www.youtube.com/watch?v=Hn-KmLIt-AQ"]),
("Mi niña Lola", "Buika", "English", 3.87, 4.92, "Jazz", ["https://www.youtube.com/watch?v=Gu9Qm4Uvibc"]),
("History Repeating", "Shirley Bassey", "English", 5.9, 4.6, "Jazz", ["https://www.youtube.com/watch?v=nC2pgcagyRk"]),
("Be Careful", "Peter Cincotti", "English", 3.92, 5.52, "Jazz", ["https://www.youtube.com/watch?v=9FghVc5JoI8"]),
("Salty Papa Blues", "Dinah Washington", "English", 3.71, 4.66, "Jazz", ["https://www.youtube.com/watch?v=b4yOIyJE_2A"]),
("If You Don't, I Know Who Will", "Bessie Smith", "English", 3.71, 4.66, "Jazz", ["https://www.youtube.com/watch?v=l5i-bYcEhy0"]),
("Try  a Little Tenderness", "Michael Bublé", "English", 4.0, 5.7, "Jazz", ["https://www.youtube.com/watch?v=7mG8gcsVeH0"]),
("Freakmachine", "Tied & Tickled Trio", "English", 5.46, 4.62, "Jazz", ["https://www.youtube.com/watch?v=OwQlBwR1iHU"]),
("To Composer John Cage", "Anthony Braxton", "English", 3.44, 5.63, "Jazz", ["https://www.youtube.com/watch?v=ceeUhorJ39I"]),
("Kpaligoth", "Ruins", "English", 3.06, 5.31, "Jazz", ["https://www.youtube.com/watch?v=8ZQjwgZhDXo"]),
("To My Friend Kenny McKenny", "Anthony Braxton", "English", 3.44, 5.63, "Jazz", ["https://www.youtube.com/watch?v=vjDEt2XezHQ"]),
("Cage of Love (Jazz version)", "Daija", "English", 5.02, 4.93, "Jazz", ["https://www.youtube.com/watch?v=TqZGSWg7jJ0"]),
("The Boss", "Joseph Gershwin & The Universal-International Orchestra", "English", 4.54, 6.12, "Jazz", [""]),
("St. Louis Blues", "Louis Prima", "English", 4.27, 4.57, "Jazz", ["https://www.youtube.com/watch?v=ZsTmOptC0w4"]),
("Opus de Bebop", "Stan Getz", "English", 5.11, 5.52, "Jazz", ["https://www.youtube.com/watch?v=bnFPVyaeONA"]),
("After An Asteroid Crashes Into The Ocean", "The 5th Plateau", "English", 5.11, 5.52, "Jazz", ["https://www.youtube.com/watch?v=MfsugkikLJI"]),
("Riding With Gabriel Greenberg", "Badly Drawn Boy", "English", 5.11, 5.52, "Jazz", [""]),
("Flashing Lights", "BADBADNOTGOOD", "English", 4.4, 5.0, "Jazz", ["https://www.youtube.com/watch?v=bl6EomECg2Q"]),
("Track C-Group Dancers", "Charles Mingus", "English", 4.68, 5.8, "Jazz", ["https://www.youtube.com/watch?v=vv3DnIvCifI"]),
("Powerhouse", "Raymond Scott", "English", 5.56, 4.46, "Jazz", ["https://www.youtube.com/watch?v=r3FLN0iQ9SQ"]),
("Paranoid Android (Live)", "Brad Mehldau", "English", 2.1, 2.27, "Jazz", ["https://www.youtube.com/watch?v=SHxuIKjZEiQ"]),
("No. 4", "Peter Brötzmann", "English", 4.19, 4.55, "Jazz", ["https://www.youtube.com/watch?v=lTQX_70VHI0"]),
("Them That Got [Live]", "Ben Folds", "English", 5.04, 3.96, "Jazz", ["https://www.youtube.com/watch?v=mh-MZhxD4jw"]),
("Perry Mason Theme", "Buddy Morrow", "English", 2.81, 3.55, "Jazz", ["https://www.youtube.com/watch?v=xiWUEUY5N5Y"]),
("Shoot To Kill", "Quincy Jones & His Orchestra", "English", 2.71, 3.97, "Jazz", ["https://www.youtube.com/watch?v=xKZ-zNSovDg"]),
("French Quarter", "Alex North & His Orchestra", "English", 2.84, 3.04, "Jazz", ["https://www.youtube.com/watch?v=EXkfI33OuXI"]),
("Taunting Scene", "Stan Kenton & His Orchestra", "English", 2.93, 3.91, "Jazz", ["https://www.youtube.com/watch?v=XE2r4AXh3IM"]),
("Where I Live: The Apartment, Cleaning Up for Jenny, The Polish Landlady", "Stan Getz", "English", 2.81, 3.55, "Jazz", ["https://www.youtube.com/watch?v=QVeFR4InP_M"]),
("The End", "PENTAGONIC", "English", 5.01, 3.8, "Jazz", ["https://www.youtube.com/watch?v=V14BYcZcQQA"]),
("Experiment In Terror", "Henry Mancini", "English", 2.93, 4.42, "Jazz", ["https://www.youtube.com/watch?v=6UJ0SSHlarc"]),
("On The Corner (Take 4)", "Miles Davis", "English", 3.95, 5.57, "Jazz", ["https://www.youtube.com/watch?v=FryzQG8FG5Q"]),
("Oclupaca", "Duke Ellington", "English", 4.92, 4.58, "Jazz", ["https://www.youtube.com/watch?v=7gj7uFy76UU"]),
("A Distant Voice", "Sonny Simmons", "English", 2.45, 3.71, "Jazz", ["https://www.youtube.com/watch?v=5nmAC7mq144"]),
("Lonely Woman", "Desert Island Dicks", "English", 4.24, 4.49, "Jazz", ["https://www.youtube.com/watch?v=YDLfZExOZ1g"]),
("Rastro 5 (For Emir Kusturica)", "Tripleplay", "English", 3.95, 5.57, "Jazz", ["https://www.youtube.com/watch?v=Qfs5YZSgi1c"]),
("Peace", "Desert Island Dicks", "English", 3.63, 4.17, "Jazz", ["https://www.youtube.com/watch?v=lRKi_KBSggI"]),
("Focus on Sanity", "Desert Island Dicks", "English", 3.63, 4.17, "Jazz", ["https://www.youtube.com/watch?v=utlYg2PciCk"]),
("Racine de deux", "Grumpf Quartet", "English", 3.95, 5.57, "Jazz", ["https://www.youtube.com/watch?v=p_3BEJMiR_8"]),
("Till The Cows Come Home", "Lucille Bogan", "English", 5.5, 5.49, "Jazz", ["https://www.youtube.com/watch?v=U5DTaPYSys0"]),
("Nice Guys", "Art Ensemble Of Chicago", "English", 3.26, 2.44, "Jazz", ["https://www.youtube.com/watch?v=hukv33Hoynw"]),
("Take Five (Live)", "The Dave Brubeck Quartet", "English", 5.05, 3.76, "Jazz", ["https://www.youtube.com/watch?v=JXi3rHqnAIU"]),
("鳳凰-Phoenix", "Jazztronik", "English", 4.85, 3.63, "Jazz", ["https://www.youtube.com/watch?v=luPrDGVh2ss"]),
("Phoenix", "Jazztronik", "English", 1.85, 2.5, "Jazz", ["https://www.youtube.com/watch?v=_mDO4n5xO30"]),
("Night Waltz", "Keiko Matsui", "English", 4.53, 3.82, "Jazz", ["https://www.youtube.com/watch?v=vR-5f-hMsiE"]),
("Splitter", "greenwood, johnny", "English", 1.45, 2.54, "Jazz", ["https://www.youtube.com/watch?v=Q2sHF7yezjI"]),
("Manteca (Remastered 1998)", "Dizzy Gillespie & His Orchestra", "English", 3.45, 5.35, "Jazz", ["https://www.youtube.com/watch?v=YRCMmhijz1s"]),
("22nd Century", "Nina Simone", "English", 1.24, 1.5, "Jazz", ["https://www.youtube.com/watch?v=PYcgCiWAv8c"]),
("3", "Bohren & der Club of Gore", "English", 2.2, 3.0, "Jazz", ["https://www.youtube.com/watch?v=RJBLUYcqwwY"]),
("Eclipse", "Charles Mingus", "English", 2.58, 1.62, "Jazz", ["https://www.youtube.com/watch?v=7D2pa81NPfs"]),
("The Night", "Morphine", "English", 4.96, 4.03, "Jazz", ["https://www.youtube.com/watch?v=wI5hxEguomo"]),
("Lobby", "The Kilimanjaro Darkjazz Ensemble", "English", 4.88, 3.79, "Jazz", ["https://www.youtube.com/watch?v=Ueokv2qQYDs"]),
("On Demon Wings", "Bohren & der Club of Gore", "English", 5.14, 3.34, "Jazz", ["https://www.youtube.com/watch?v=vl4x5pchBXI"]),
("Maximum Black", "Bohren & der Club of Gore", "English", 5.25, 3.74, "Jazz", ["https://www.youtube.com/watch?v=v15-0atv0us"]),
("Pirate Jenny", "Nina Simone", "English", 4.52, 3.75, "Jazz", ["https://www.youtube.com/watch?v=V7awW5nrDHk"]),
("Confessions (feat. Leland Whitty)", "BADBADNOTGOOD", "English", 5.55, 3.37, "Jazz", ["https://www.youtube.com/watch?v=237Nq3O4XXE"]),
("All Is Loneliness", "Moondog", "English", 5.32, 4.71, "Jazz", ["https://www.youtube.com/watch?v=2PGYZ0KP1cQ"]),
("The Antidote", "King Kooba", "English", 3.99, 4.8, "Jazz", ["https://www.youtube.com/watch?v=k1lKdJrTaGo"]),
("Dropjes", "Brad Mehldau", "English", 4.28, 4.41, "Jazz", ["https://www.youtube.com/watch?v=9fxjn7foM2Y"]),
("Kryptonite Smokes the Red Line", "Isotope 217", "English", 3.4, 5.57, "Jazz", ["https://www.youtube.com/watch?v=rvHDgN-DgPE"]),
("Weird Nightmare", "Charles Mingus", "English", 3.97, 4.46, "Jazz", ["https://www.youtube.com/watch?v=5euJvrvgfqU"]),
("Juba Lee Brown", "His Name Is Alive", "English", 3.94, 3.4, "Jazz", ["https://www.youtube.com/watch?v=hRinrdWplNw"]),
("Under Wraps", "Barry Adamson", "English", 4.24, 4.83, "Jazz", ["https://www.youtube.com/watch?v=_iUrCzxT3Cc"]),
("DJ Krust - Re-Arrange", "The Cinematic Orchestra", "English", 6.0, 3.86, "Jazz", ["https://www.youtube.com/watch?v=nzXtIxXPw28"]),
("On The Wrong Side Of Relaxation", "Barry Adamson", "English", 3.4, 5.57, "Jazz", ["https://www.youtube.com/watch?v=gU-BlV_U_Ew"]),
("An Evening With Vincent Van Ritz", "Eberhard Weber", "English", 5.14, 3.62, "Jazz", ["https://www.youtube.com/watch?v=cUPDJWtYyvE"]),
("Headless Horseman", "Kay Starr", "English", 2.1, 3.07, "Jazz", ["https://www.youtube.com/watch?v=UUm6duVb_Wg"]),
("Sounds From The Big House", "Barry Adamson", "English", 5.25, 5.14, "Jazz", ["https://www.youtube.com/watch?v=7riAw65i4M8"]),
("Rocket Number Nine", "Sun Ra", "English", 4.0, 2.68, "Jazz", ["https://www.youtube.com/watch?v=mLcAA9RYChQ"]),
("Disused Fairground", "James Kealy & His Regal Toes", "English", 4.74, 4.3, "Jazz", [""]),
("My Sound", "Squarepusher", "English", 5.58, 3.9, "Jazz", ["https://www.youtube.com/watch?v=OzoFHatryJQ"]),
("Gloomy Sunday", "Sinéad O'Connor", "English", 4.98, 4.31, "Jazz", ["https://www.youtube.com/watch?v=i8M1HXlOuCk"]),
("Komm zurück zu mir", "Bohren & der Club of Gore", "English", 4.31, 4.3, "Jazz", ["https://www.youtube.com/watch?v=9IeL_jh7XM0"]),
("Something Glue", "Glen Porter", "English", 3.15, 3.32, "Jazz", ["https://www.youtube.com/watch?v=BcWf3ohp-eI"]),
("Sand", "Allan Holdsworth", "English", 3.15, 3.32, "Jazz", ["https://www.youtube.com/watch?v=N1L7Bn74MjU"]),
("Motherland", "Mark de Clive-Lowe", "English", 3.15, 3.32, "Jazz", ["https://www.youtube.com/watch?v=3Ingh1qZmU8"]),
("Don't Forget The Fourth Track", "Kid Loco", "English", 4.02, 3.57, "Jazz", ["https://www.youtube.com/watch?v=FRjOSmc01-M"]),
("Duration Part 1", "Sixtoo", "English", 3.15, 3.32, "Jazz", ["https://www.youtube.com/watch?v=uV_M404ixHA"]),
("Witchcraft", "Frank Sinatra", "English", 1.98, 1.81, "Jazz", ["https://www.youtube.com/watch?v=_RCtDnfYPGo"]),
("House Of Mirrors", "David McCallum", "English", 4.86, 2.52, "Jazz", ["https://www.youtube.com/watch?v=OZLo1O1rGhM"]),
("We Have Died Already", "St. Germain", "English", 3.67, 2.72, "Jazz", ["https://www.youtube.com/watch?v=gD_4oKFss14"]),
("Blue Light, Red Light (Someone's There)", "Harry Connick, Jr.", "English", 4.0, 3.71, "Jazz", ["https://www.youtube.com/watch?v=y3SZ3tacC9I"]),
("Rosemary's Baby Main Theme Vocal", "Krzysztof Komeda", "English", 3.23, 4.62, "Jazz", ["https://www.youtube.com/watch?v=7wRNX94fU94"]),
("Retribution", "Abbey Lincoln", "English", 2.9, 5.09, "Jazz", ["https://www.youtube.com/watch?v=gwBd9oeZjXw"]),
("Dialect", "Christian Scott", "English", 2.9, 5.09, "Jazz", ["https://www.youtube.com/watch?v=2KdDi5RZZdE"]),
("Root Canal", "Uri Caine", "English", 2.67, 3.62, "Jazz", ["https://www.youtube.com/watch?v=-9e60ylNcno"]),
("Wonderful World", "Nine Horses", "English", 5.09, 3.88, "Jazz", ["https://www.youtube.com/watch?v=Ga4IcaRsHic"]),
("Dragon", "Paolo Conte", "English", 3.0, 5.35, "Jazz", ["https://www.youtube.com/watch?v=59D5pvbyFS8"]),
("さよなら、パーフェクトワールド。", "ミドリ", "English", 3.0, 5.35, "Jazz", ["https://www.youtube.com/watch?v=9EUU9jNl-ro"]),
("Exit music", "The Cinematic Orchestra", "English", 4.42, 3.38, "Jazz", ["https://www.youtube.com/watch?v=Jrz_z0ExoXM"]),
("It's All Forgotten Now", "Al Bowlly", "English", 3.93, 5.78, "Jazz", ["https://www.youtube.com/watch?v=mUgfQLJa0ZQ"]),
("Flute Band In Gauteng", "Jhelisa", "English", 4.97, 4.29, "Jazz", ["https://www.youtube.com/watch?v=4pZ6HI6FklI"]),
("Spooky", "The Puppini Sisters", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=c2v3Ej7z46E"]),
("Thrash Altenessen", "Bohren & der Club of Gore", "English", 4.12, 5.82, "Jazz", ["https://www.youtube.com/watch?v=FuseFynx2z0"]),
("Frozen", "Nils Petter Molvær", "English", 4.92, 4.45, "Jazz", ["https://www.youtube.com/watch?v=ff66lk8JdVs"]),
("Tabula Rasa", "Nils Petter Molvær", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=2UzkpoIJ5VI"]),
("Cloudy, Since You Went Away", "The Caretaker", "English", 4.21, 5.52, "Jazz", ["https://www.youtube.com/watch?v=xSc1R6-5sJo"]),
("Hayaletler", "DANdadaDAN", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=MVo8Q0KO5Wo"]),
("Orbits", "Brand X", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=gTXw9pK1giA"]),
("Anthem (Antediluvian Adaptation)", "Christian Scott", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=R_f-hXljvWI"]),
("Kakonita (Deathprod Mix)", "Nils Petter Molvær", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=8HhS3SSSInI"]),
("Mahlah", "John Zorn", "English", 4.78, 4.62, "Jazz", ["https://www.youtube.com/watch?v=NAEiQRTHu6U"]),
("Spooky", "Tok Tok Tok", "English", 5.47, 5.52, "Jazz", ["https://www.youtube.com/watch?v=2TQxL7AGs4s"]),
("Above & Below", "Allan Holdsworth", "English", 4.4, 6.0, "Jazz", ["https://www.youtube.com/watch?v=JWd6WL8Xh04"]),
("Won't U Please B Nice", "Nellie McKay", "English", 5.82, 2.98, "Jazz", ["https://www.youtube.com/watch?v=HfOZpzBWagI"]),
("Who Killed Big Bird?", "Barry Adamson", "English", 3.56, 2.91, "Jazz", ["https://www.youtube.com/watch?v=HDeYv44ZAvA"]),
("My Sweet Lord", "Nina Simone", "English", 6.1, 5.02, "Jazz", ["https://www.youtube.com/watch?v=L2m6IsVvj5c"]),
("the chase", "Iordache", "English", 3.48, 5.51, "Jazz", ["https://www.youtube.com/watch?v=YeeUud04z8k"]),
("Lonely Women", "Laura Nyro", "English", 5.16, 3.73, "Jazz", ["https://www.youtube.com/watch?v=2yRwKQZDCMQ"]),
("Richard Pryor Addresses A Tearful Nation", "Joe Henry", "English", 4.12, 5.11, "Jazz", ["https://www.youtube.com/watch?v=TfyatPXKu5I"]),
("Restored, Returned", "Tord Gustavsen Ensemble", "English", 3.63, 4.64, "Jazz", ["https://www.youtube.com/watch?v=FFBdi3jveew"]),
("And Her Tears Flowed Like Wine", "Stan Kenton", "English", 3.63, 4.64, "Jazz", ["https://www.youtube.com/watch?v=0xFHfCaBY_4"]),
("Bölcsidal", "Emil.Rulez!", "English", 6.0, 5.48, "Jazz", ["https://www.youtube.com/watch?v=MUHBuKcXUXg"]),
("The Confidence Man", "Patti Austin", "English", 3.63, 4.64, "Jazz", ["https://www.youtube.com/watch?v=1ntw5iI_urU"]),
("I'm Down In the Dumps", "Magda Piskorczyk", "English", 5.18, 4.17, "Jazz", ["https://www.youtube.com/watch?v=H8dDwOSyP5o"]),
("De Drums", "Jarek Śmietana & Wojtek Karolak", "English", 3.53, 4.12, "Jazz", ["https://www.youtube.com/watch?v=9kvqcLD51Ts"]),
("St. James Infirmary (Live Version)", "Louis Armstrong", "English", 2.87, 4.07, "Jazz", ["https://www.youtube.com/watch?v=QzcpUdBw7gs"]),
("Crye Me A River", "Dinah Washington", "English", 3.63, 4.64, "Jazz", ["https://www.youtube.com/watch?v=tx6MQ7tmgSc"]),
("For All We Know", "Nina Simone", "English", 3.3, 3.82, "Jazz", ["https://www.youtube.com/watch?v=0t4VUElwLAs"]),
("Harlem Fuss", "Fats Waller And His Buddies", "English", 5.75, 5.16, "Jazz", ["https://www.youtube.com/watch?v=hzQSbAybdMY"]),
("Ain't Misbehavin' - Original", "Fats Waller", "English", 5.75, 5.16, "Jazz", ["https://www.youtube.com/watch?v=PSNPpssruFY"]),
("Zat You Santa Claus?", "Louis Armstrong", "English", 4.32, 3.55, "Jazz", ["https://www.youtube.com/watch?v=cqIpxPvhb_8"]),
("Vilderness (The Cinematic Orchestra remix)", "Nils Petter Molvær", "English", 5.42, 3.68, "Jazz", ["https://www.youtube.com/watch?v=Bj7Ve-avnqQ"]),
("Presence", "Nils Petter Molvær", "English", 4.32, 3.55, "Jazz", ["https://www.youtube.com/watch?v=vdGx3L9uNzI"]),
("Cold, Cold Heart", "Bill Frisell", "English", 3.88, 3.58, "Jazz", ["https://www.youtube.com/watch?v=Xxk1UOQz37I"]),
("Christmastime Is Here (Alternate Vocal Take 5)", "Vince Guaraldi Trio", "English", 5.85, 4.18, "Jazz", ["https://www.youtube.com/watch?v=otlpG-OtJkA"]),
("Wolf Extract", "Eivind Aarset", "English", 5.23, 3.66, "Jazz", ["https://www.youtube.com/watch?v=XknXuMMrAsU"]),
("Roma", "Pizzicato Five", "English", 5.09, 3.66, "Jazz", ["https://www.youtube.com/watch?v=FyddnW7dV0s"]),
("Structural Functions Of Prezens", "David Torn", "English", 5.36, 4.26, "Jazz", ["https://www.youtube.com/watch?v=hQXSJ7Tr-Ws"]),
("Zahouly", "ZAILM Atoyanh", "English", 5.2, 3.95, "Jazz", ["https://www.youtube.com/watch?v=Dxgmikenaww"]),
("Rising Thermal 14° 16' N; 32° 28' E", "Jon Hassell", "English", 5.14, 3.38, "Jazz", ["https://www.youtube.com/watch?v=o4b5wC68ELI"]),
("Transmit Regardless", "David Torn", "English", 4.81, 3.63, "Jazz", ["https://www.youtube.com/watch?v=L1LHSqbDaTg"]),
("Will", "Terje Rypdal", "English", 4.32, 3.55, "Jazz", ["https://www.youtube.com/watch?v=KbRn6UZQQdA"]),
("Krush", "ZAILM Atoyanh", "English", 5.74, 4.21, "Jazz", ["https://www.youtube.com/watch?v=p-Z3YrHJ1sU"]),
("Part 505", "ZAILM Atoyanh", "English", 5.47, 4.01, "Jazz", ["https://www.youtube.com/watch?v=1EiYudNKuNQ"]),
("Inside The Seventh Colour", "ZAILM Atoyanh", "English", 5.48, 4.37, "Jazz", ["https://www.youtube.com/watch?v=fguoI9RyWDo"]),
("Skeletonised", "Parrket", "English", 4.32, 3.55, "Jazz", ["https://www.youtube.com/watch?v=mZQ9vpKEgfI"]),
("Emm", "Vladimir Lisanovitch", "English", 5.32, 4.34, "Jazz", [""]),
("Dubzz in Moscow", "The Removers", "English", 5.05, 3.92, "Jazz", ["https://www.youtube.com/watch?v=xLYiIBCN9ec"]),
("Passt dich nicht an", "Ursull Manchill", "English", 5.1, 4.31, "Jazz", ["https://www.youtube.com/watch?v=4AIFUBytmLo"]),
("Umgekehrt", "Ursull Manchill", "English", 5.43, 4.33, "Jazz", ["https://www.youtube.com/watch?v=vcuQpbs0yT0"]),
("TounH Siah", "Zailm Atoyanh Group", "English", 5.02, 4.33, "Jazz", ["https://www.youtube.com/watch?v=FOfkZbYw3mU"]),
("Number 11", "The Removers", "English", 5.47, 4.15, "Jazz", ["https://www.youtube.com/watch?v=LEK4v1PT5po"]),
("Re-Arrange (The Cinematic Orchestra remix)", "Krust", "English", 4.32, 3.55, "Jazz", ["https://www.youtube.com/watch?v=nzXtIxXPw28"]),
("For Care", "The Removers", "English", 5.81, 4.77, "Jazz", ["https://www.youtube.com/watch?v=rSn3781hg4A"]),
("Fragrancies", "Vladimir Lisanovitch", "English", 5.17, 4.41, "Jazz", ["https://www.youtube.com/watch?v=pM68_oA-bSY"]),
("Lullaby In Black", "Horace Tapscott", "English", 2.41, 4.48, "Jazz", ["https://www.youtube.com/watch?v=e1r1jmMVzZY"]),
("Drinking Again", "Frank Sinatra", "English", 2.52, 3.63, "Jazz", ["https://www.youtube.com/watch?v=gnN_AmvYDGY"]),
("You've Got to Learn", "Nina Simone", "English", 5.38, 4.43, "Jazz", ["https://www.youtube.com/watch?v=_xvWScduT_g"]),
("Where or When", "Frank Sinatra", "English", 2.38, 3.93, "Jazz", ["https://www.youtube.com/watch?v=KKxRor4k8hM"]),
("Fingertips", "Poe", "English", 6.05, 4.83, "Jazz", ["https://www.youtube.com/watch?v=N67n9XszJ7s"]),
("When the Swallows Come Back to Capistrano", "The Ink Spots", "English", 4.8, 4.62, "Jazz", ["https://www.youtube.com/watch?v=TmT8S3qZnNk"]),
("Don't Get Around Much Anymore", "The Ink Spots", "English", 5.85, 4.08, "Jazz", ["https://www.youtube.com/watch?v=2emGPh-PR-M"]),
("Only the Lonely", "Frank Sinatra", "English", 1.67, 2.82, "Jazz", ["https://www.youtube.com/watch?v=lL5ZCKof5To"]),
("Road To The West", "菅野よう子", "English", 2.67, 4.37, "Jazz", ["https://www.youtube.com/watch?v=GZl7N8sXyEo"]),
("Mystery Man", "Terje Rypdal", "English", 5.59, 4.08, "Jazz", ["https://www.youtube.com/watch?v=LAmQbjmOE_g"]),
("We Three (My Echo, My Shadow and Me)", "The Ink Spots", "English", 3.93, 3.5, "Jazz", ["https://www.youtube.com/watch?v=TOhtZdINkY0"]),
("In My Secret Life", "Till Brönner", "English", 4.95, 3.93, "Jazz", ["https://www.youtube.com/watch?v=Ln8K5SmFLyM"]),
("The Bard Returns", "Brad Mehldau", "English", 3.42, 3.1, "Jazz", ["https://www.youtube.com/watch?v=uBEMNtcs9Lw"]),
("Samba Przed Rozstaniem", "Hanna Banaszak", "English", 5.33, 4.87, "Jazz", ["https://www.youtube.com/watch?v=RmpkiVh7_Hc"]),
("Hurt So Bad", "Alicia Keys", "English", 2.67, 4.37, "Jazz", ["https://www.youtube.com/watch?v=O74ElHK4Q-E"]),
("This Obsession", "Victoria Hart", "English", 2.67, 4.37, "Jazz", ["https://www.youtube.com/watch?v=Wr_MCKHCmVk"]),
("Lonely House", "Adi Braun", "English", 2.67, 4.37, "Jazz", ["https://www.youtube.com/watch?v=iLhm6ib3V8c"]),
("Maybe Winter", "Belle Plaine", "English", 4.82, 4.4, "Jazz", ["https://www.youtube.com/watch?v=jabxwMo3jQE"]),
("Lone", "FOWL", "English", 3.82, 3.56, "Jazz", ["https://www.youtube.com/watch?v=tNhYiE-Qnd4"]),
("Eggs and Sausage (In a Cadillac With Susan Michelson)", "Tom Waits", "English", 4.48, 3.81, "Jazz", ["https://www.youtube.com/watch?v=l2GfMu0JJs8"]),
("I'm in the Mood for Love", "Bryan Ferry", "English", 5.89, 3.65, "Jazz", ["https://www.youtube.com/watch?v=rO3_kyvGJig"]),
("Nighthawk Postcards (From Easy Street)", "Tom Waits", "English", 4.46, 3.92, "Jazz", ["https://www.youtube.com/watch?v=PlPN2vtn-0s"]),
("Mr. Groove", "Euge Groove", "English", 5.29, 3.48, "Jazz", ["https://www.youtube.com/watch?v=ehg2gwHwAuI"]),
("Deep in a Dream", "Chet Baker", "English", 5.43, 3.91, "Jazz", ["https://www.youtube.com/watch?v=OkXsKQvLQws"]),
("It Don't Mean a Thing (If It Ain't Got That Swing)", "Nina Simone", "English", 5.29, 3.48, "Jazz", ["https://www.youtube.com/watch?v=myRc-3oF1d0"]),
("You Can't Go Home Again", "Chet Baker", "English", 5.29, 3.48, "Jazz", ["https://www.youtube.com/watch?v=ItQhMIh8Fsw"]),
("Diagnostic", "Ibrahim Maalouf", "English", 5.5, 3.79, "Jazz", ["https://www.youtube.com/watch?v=le-e37ZMck8"]),
("Flash in Dreamland", "Bill Evans", "English", 4.25, 2.43, "Jazz", ["https://www.youtube.com/watch?v=MZ-nVJHH5eA"]),
("Shukuru", "Pharoah Sanders", "English", 5.29, 3.48, "Jazz", ["https://www.youtube.com/watch?v=RdDmBDBR3cc"]),
("Peaceful", "Miles Davis", "English", 6.05, 3.46, "Jazz", ["https://www.youtube.com/watch?v=VlIyqiIJ98w"]),
("Gentle Rain", "Art Farmer", "English", 5.99, 3.15, "Jazz", ["https://www.youtube.com/watch?v=IPdN1Hufx_o"]),
("From a late night train", "Blue Nile", "English", 3.25, 2.76, "Jazz", ["https://www.youtube.com/watch?v=cVNfv_4HAxk"]),
("Fuses", "Stereolab", "English", 6.42, 4.29, "Jazz", ["https://www.youtube.com/watch?v=HsGAvGIyosg"]),
("The House", "Katie Melua", "English", 4.73, 3.3, "Jazz", ["https://www.youtube.com/watch?v=GRYP2E395f8"]),
("Comment Allez Vous", "Blossom Dearie", "English", 6.59, 4.16, "Jazz", ["https://www.youtube.com/watch?v=rxPeQyExPP8"]),
("Cheater's Armoury", "Hanne Hukkelberg", "English", 7.05, 4.57, "Jazz", ["https://www.youtube.com/watch?v=AmNtuGGyAus"]),
("N.Y.C.'s No Lark", "Bill Evans", "English", 6.38, 3.09, "Jazz", ["https://www.youtube.com/watch?v=FQra1yLoolU"]),
("If Loving You Is Wrong", "Cassandra Wilson", "English", 6.62, 3.19, "Jazz", ["https://www.youtube.com/watch?v=theMDrkhLN0"]),
("And I Love Him", "Diana Krall", "English", 6.99, 3.32, "Jazz", ["https://www.youtube.com/watch?v=DDvootNWuLQ"]),
("Xibaba", "Donald Byrd", "English", 6.38, 3.09, "Jazz", ["https://www.youtube.com/watch?v=Zck1urC4-Uw"]),
("Friend", "Susanna and the Magical Orchestra", "English", 6.38, 3.09, "Jazz", ["https://www.youtube.com/watch?v=-qE48UrEwp4"]),
("Blame It On My Youth", "Chet Baker", "English", 6.42, 2.52, "Jazz", ["https://www.youtube.com/watch?v=swP4GyfmrnQ"]),
("World Peace Now", "Build An Ark", "English", 6.38, 3.09, "Jazz", ["https://www.youtube.com/watch?v=9nPFvEQabR8"]),
("Out Here. In There.", "Sidsel Endresen & Bugge Wesseltoft", "English", 5.98, 3.83, "Jazz", ["https://www.youtube.com/watch?v=UBRVI5uaIM4"]),
("The Kyper Belt", "Roscoe Mitchell", "English", 6.38, 3.09, "Jazz", ["https://www.youtube.com/watch?v=zP1BWL1SBeo"]),
("Blue Moon", "Connee Boswell", "English", 7.32, 4.31, "Jazz", ["https://www.youtube.com/watch?v=SWddzmJgKT8"]),
("Time and Space", "The Cinematic Orchestra", "English", 6.59, 4.01, "Jazz", ["https://www.youtube.com/watch?v=AmRYyQuQSUw"]),
("Dream", "Michael Bublé", "English", 6.72, 3.36, "Jazz", ["https://www.youtube.com/watch?v=S4qWcWiSx0w"]),
("If I Didn't Care", "The Ink Spots", "English", 7.18, 4.69, "Jazz", ["https://www.youtube.com/watch?v=zyLZ_KScE9s"]),
("One of Those Summer Days", "Rhye", "English", 3.94, 2.26, "Jazz", ["https://www.youtube.com/watch?v=Rl0IEfIr0XY"]),
("Moment Of Hesitation", "Flying Lotus", "English", 4.0, 1.57, "Jazz", ["https://www.youtube.com/watch?v=65UxkX5t2mI"]),
("Space Lion", "The Seatbelts", "English", 6.53, 3.51, "Jazz", ["https://www.youtube.com/watch?v=vu_YGgZQ9DE"]),
("Hoist Anchor", "Hanne Hukkelberg", "English", 7.05, 2.99, "Jazz", ["https://www.youtube.com/watch?v=JslQGP8ukUg"]),
("Afro-Harping", "Dorothy Ashby", "English", 7.67, 4.58, "Jazz", ["https://www.youtube.com/watch?v=GWqzoAEt9lM"]),
("That's How It Goes", "Michael Bublé", "English", 6.78, 3.6, "Jazz", ["https://www.youtube.com/watch?v=HETInzYq9Wo"]),
("Displaced", "Hanne Hukkelberg", "English", 5.95, 3.58, "Jazz", ["https://www.youtube.com/watch?v=2Yq9jbAekIU"]),
("Listen", "Room Eleven", "English", 7.24, 2.54, "Jazz", ["https://www.youtube.com/watch?v=50CfRvWwOso"]),
("Who Knows Where The Time Goes", "Nina Simone", "English", 4.79, 3.46, "Jazz", ["https://www.youtube.com/watch?v=xlJfrpuhlPI"]),
("Black Women", "St. Germain", "English", 5.9, 3.3, "Jazz", ["https://www.youtube.com/watch?v=GwmgPkBJlDU"]),
("Cherish", "Pat Metheny", "English", 7.42, 3.17, "Jazz", ["https://www.youtube.com/watch?v=HbW4FmlN3M8"]),
("Minha saudade", "João Donato", "English", 6.58, 2.79, "Jazz", ["https://www.youtube.com/watch?v=HrWjuk5QGSk"]),
("(You Don't Know) How Glad I Am", "Nancy Wilson", "English", 7.21, 4.41, "Jazz", ["https://www.youtube.com/watch?v=Wj05EY2aP7I"]),
("Your Request", "Nneka", "English", 7.29, 4.08, "Jazz", ["https://www.youtube.com/watch?v=7lOqnvoBSwM"]),
("Samotność", "O.S.T.R.", "English", 5.95, 3.89, "Jazz", ["https://www.youtube.com/watch?v=bkDF4i4Ovns"]),
("Secret Agent Man", "Mel Tormé", "English", 7.1, 2.84, "Jazz", ["https://www.youtube.com/watch?v=iUMWdYg7fpI"]),
("Day By Day", "Najee", "English", 6.81, 2.89, "Jazz", ["https://www.youtube.com/watch?v=xkxpQ9lmB8U"]),
("Somalia", "Al Di Meola", "English", 7.42, 3.17, "Jazz", ["https://www.youtube.com/watch?v=G7ojfXf4hdY"]),
("While We're Young", "Wes Montgomery", "English", 7.36, 3.7, "Jazz", ["https://www.youtube.com/watch?v=5nOXV0UHtQA"]),
("Cinny's Waltz", "Holly Cole", "English", 7.42, 3.17, "Jazz", ["https://www.youtube.com/watch?v=bRQEO1syBFc"]),
("Love Has No Pride", "Jane Monheit", "English", 7.39, 4.16, "Jazz", ["https://www.youtube.com/watch?v=yEuE1gpYJ8w"]),
("Azure", "Duke Ellington", "English", 7.05, 3.77, "Jazz", ["https://www.youtube.com/watch?v=wK-MTq4tUcA"]),
("Positive Thinking", "Acoustic Alchemy", "English", 6.89, 2.73, "Jazz", ["https://www.youtube.com/watch?v=LcT2d7L3QQM"]),
("Main Tune", "Contemporary Noise Quartet", "English", 5.58, 3.65, "Jazz", ["https://www.youtube.com/watch?v=ZvfRr14emRk"]),
("When I'm Alone", "Peter White", "English", 3.71, 1.58, "Jazz", ["https://www.youtube.com/watch?v=IvCyz-rnhm4"]),
("Dust My Broom", "Cassandra Wilson", "English", 5.42, 3.39, "Jazz", ["https://www.youtube.com/watch?v=M3mio2QIjGQ"]),
("The Wind Cries Mary", "Jamie Cullum", "English", 6.68, 3.41, "Jazz", ["https://www.youtube.com/watch?v=m0VNsv-3HTw"]),
("Nuages", "Allan Holdsworth", "English", 6.32, 3.87, "Jazz", ["https://www.youtube.com/watch?v=tR5PsfsZRFs"]),
("Bliss 2 This", "Candy Dulfer", "English", 7.41, 4.76, "Jazz", ["https://www.youtube.com/watch?v=efjGLRaxNNY"]),
("Kazuko", "Pharoah Sanders", "English", 6.22, 4.41, "Jazz", ["https://www.youtube.com/watch?v=Y7EGQzn8e1k"]),
("Christmas Time Is Here", "Anita Baker", "English", 7.13, 3.25, "Jazz", ["https://www.youtube.com/watch?v=Ro3oxKPau1Y"]),
("Say Goodnight", "Peter White", "English", 6.29, 3.21, "Jazz", ["https://www.youtube.com/watch?v=j6vr_inpJ-k"]),
("Call Me", "Kimbra", "English", 6.85, 5.27, "Jazz", ["https://www.youtube.com/watch?v=ShVeKvn_bI4"]),
("Words", "Bebel Gilberto", "English", 6.83, 5.37, "Jazz", ["https://www.youtube.com/watch?v=Ro4z8o1UBgM"]),
("Anyone to Love", "Michael Bublé", "English", 7.42, 5.57, "Jazz", ["https://www.youtube.com/watch?v=QrHN9tJLaQ0"]),
("Love At First Sight", "Michael Bublé", "English", 7.08, 4.63, "Jazz", ["https://www.youtube.com/watch?v=7FmfZTUaCMA"]),
("Inny Smak", "Bisquit", "English", 2.11, 1.58, "Jazz", ["https://www.youtube.com/watch?v=HPJ450D5Rng"]),
("Airport Sadness", "Brad Mehldau", "English", 7.0, 4.12, "Jazz", ["https://www.youtube.com/watch?v=f4gfv55So-o"]),
("About You feat. Thief", "Clara Hill", "English", 7.35, 6.54, "Jazz", ["https://www.youtube.com/watch?v=uxOfy1GcG7w"]),
("My Mind", "Jimi Tenor", "English", 7.11, 4.24, "Jazz", ["https://www.youtube.com/watch?v=4pgWVU0IJHM"]),
("Closer Still", "Brian Simpson", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=MbvG92QXbFU"]),
("Is It You", "Krzysztof Kiljański", "English", 3.93, 3.35, "Jazz", ["https://www.youtube.com/watch?v=X40H0eoMD1s"]),
("More To Love", "Amy Lee", "English", 4.0, 4.4, "Jazz", ["https://www.youtube.com/watch?v=lB8ZsjMhy5M"]),
("I Don't Stand a Ghost of a Chance With You", "Wes Montgomery", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=x-nnVnu7uto"]),
("When I Look In Your Eyes", "Irene Kral", "English", 6.72, 3.32, "Jazz", ["https://www.youtube.com/watch?v=Ihxd-1ccPyI"]),
("River", "Madeleine Peyroux Feat. K.D. Lang", "English", 5.88, 4.32, "Jazz", ["https://www.youtube.com/watch?v=QE2d7GCVNQw"]),
("Coney Island Girl", "The Rosebud Orchestra", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=JhLG1Bbe4ks"]),
("Intimate Strangers", "Rippingtons", "English", 6.82, 4.39, "Jazz", ["https://www.youtube.com/watch?v=kYUwjfXh_x8"]),
("When we fell in love", "Camiel", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=Vb2hfiYwOnw"]),
("Paula", "Torcuato Mariano", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=BULP6u7EaVc"]),
("Wind feat. Yukimi Nagano", "Sleep Walker", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=0BZJCdnm-gA"]),
("The Vital Thing", "Swing Out Sister", "English", 7.15, 4.52, "Jazz", ["https://www.youtube.com/watch?v=tOxiisSVc7I"]),
("INTO THE SUN feat. Bembe Segue", "Sleep Walker", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=IToXckMqGmI"]),
("Porgy (Live Version - Newport Jazz Festival, June 30, 1960)", "Nina Simone", "English", 4.55, 4.25, "Jazz", ["https://www.youtube.com/watch?v=2it_04q_5_k"]),
("Now I Know Why (They Call It Falling)", "Michael Franks", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=1EctAz1eSbE"]),
("Listen Up", "Doc Powell", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=qqC_f0HQOw8"]),
("Morning Scapes", "Bembe Segue", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=-MFiZBAAiJ8"]),
("'Round Midnight (Live Version)", "Dexter Gordon", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=0Q8ZV6Ppw-c"]),
("Time (Feat. Cleveland Watkiss) (Kyoto Jazz Massive Remix)", "Makoto", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=ZXT1pmeRDLQ"]),
("West Hartford (Album Version)", "Brad Mehldau", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=-DoJTLJ325w"]),
("Coney Island Girl", "Aleksandra Kwasniewska and The Belgian Sweets", "English", 7.22, 6.02, "Jazz", ["https://www.youtube.com/watch?v=4YdTXSL_yhk"]),
("Something Hairy", "Ambiguity", "English", 5.4, 3.04, "Jazz", ["https://www.youtube.com/watch?v=H5v3kku4y6Q"]),
("Tentations", "Mélanie Renaud", "English", 7.32, 6.41, "Jazz", ["https://www.youtube.com/watch?v=hgumt8UwTQE"]),
("Bali Hai", "Stacey Kent", "English", 7.3, 5.74, "Jazz", ["https://www.youtube.com/watch?v=u7IuoJwyRV0"]),
("Black Is the Color of My True Love's Hair (Jaffa Remix)", "Nina Simone & Jaffa", "English", 6.59, 5.23, "Jazz", ["https://www.youtube.com/watch?v=2d7puV62xQQ"]),
("Reel Life", "The Cinematic Orchestra", "English", 4.86, 2.38, "Jazz", ["https://www.youtube.com/watch?v=REkD2js-V6E"]),
("Hindsight", "Beady Belle", "English", 4.76, 2.05, "Jazz", ["https://www.youtube.com/watch?v=y5MS8suJNqw"]),
("Greeting To Saud (Brother McCoy Tyner)", "Pharoah Sanders", "English", 5.88, 3.54, "Jazz", ["https://www.youtube.com/watch?v=tbyQbLuVDDM"]),
("Southwest Passage", "Dave Grusin", "English", 4.33, 2.52, "Jazz", ["https://www.youtube.com/watch?v=7YSAUE85tXs"]),
("Supertaps", "Wevie Stonder", "English", 4.14, 1.91, "Jazz", ["https://www.youtube.com/watch?v=WjuAY9qWOcA"]),
("Suyafhu Skin... Snapping The Hollow Reed", "David Torn", "English", 5.24, 4.5, "Jazz", ["https://www.youtube.com/watch?v=j-frS4y00Ew"]),
("Gremlin", "PENTAGONIC", "English", 5.42, 3.85, "Jazz", ["https://www.youtube.com/watch?v=Bvf9v4EHovY"]),
("Too Late", "Janita", "English", 2.95, 2.74, "Jazz", ["https://www.youtube.com/watch?v=LuQ46cMJWh4"]),
("My Cat Arnold Is Dead", "Karen Mantler", "English", 2.11, 1.93, "Jazz", ["https://www.youtube.com/watch?v=2xYY4o_1G6Q"]),
("Dub", "Ernesto Schnack", "English", 1.87, 2.06, "Jazz", ["https://www.youtube.com/watch?v=53BizHIWNXQ"]),
("La Javanese", "Madeleine Peyroux", "English", 5.49, 3.49, "Jazz", ["https://www.youtube.com/watch?v=NDzp960aoIs"]),
("If Dogs Run Free", "Bob Dylan", "English", 5.29, 3.98, "Jazz", ["https://www.youtube.com/watch?v=UH6V1XvQC3c"]),
("No Mountain", "The Cat Empire", "English", 6.02, 2.74, "Jazz", ["https://www.youtube.com/watch?v=nxIHsdE0c-Y"]),
("All That's Left Is To Say Goodbye", "Astrud Gilberto", "English", 4.11, 3.15, "Jazz", ["https://www.youtube.com/watch?v=OT5pdFtC5ls"]),
("Spitalfields", "Red Snapper", "English", 4.11, 3.15, "Jazz", ["https://www.youtube.com/watch?v=xcGS_aePQfc"]),
("Waltz for Koop (DJ Patife remix)", "Koop", "English", 4.83, 3.55, "Jazz", ["https://www.youtube.com/watch?v=yVOCrUtKSyk"]),
("The Laziest Gal In Town", "Nina Simone", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=Cud-O1ZvcIo"]),
("Lazy", "Marilyn Monroe", "English", 6.15, 4.92, "Jazz", ["https://www.youtube.com/watch?v=3x28sT--W4Q"]),
("Soul Salsa Soul (short version)", "St. Germain", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=9v_NvxzX0gk"]),
("Nuthinduan Waltz", "Andrew Bird's Bowl of Fire", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=OuctmGmhfL0"]),
("Quiet Please", "Galactic", "English", 4.56, 3.53, "Jazz", ["https://www.youtube.com/watch?v=ox5DByUrWus"]),
("Yummy, Yummy, Yummy", "Julie London", "English", 5.25, 4.47, "Jazz", ["https://www.youtube.com/watch?v=d_2wH3q26EM"]),
("It's The Talk Of The Town", "Dizzy Gillespie", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=uJQVS-wKgxw"]),
("Gnossienne No 2", "Jacques Loussier Trio", "English", 4.07, 3.57, "Jazz", ["https://www.youtube.com/watch?v=MExWN-Nk_5Y"]),
("For All We Know", "Dexter Gordon", "English", 5.78, 3.66, "Jazz", ["https://www.youtube.com/watch?v=F5LBQhjQvvQ"]),
("Red Wagon", "Count Basie", "English", 4.98, 4.24, "Jazz", ["https://www.youtube.com/watch?v=yUvWGAICn58"]),
("Lazy Afternoon", "Jackie Allen", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=isWDkuKTAUo"]),
("I'll B Seeing You", "Frank Sinatra", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=dcdBmZImv30"]),
("Ev'rytime We Say Goodbye", "Nina Simone", "English", 3.24, 3.33, "Jazz", ["https://www.youtube.com/watch?v=zhPY3fjhoEA"]),
("The Laziest Girl in Town", "Nina Simone", "English", 4.87, 2.93, "Jazz", ["https://www.youtube.com/watch?v=Cud-O1ZvcIo"]),
("Worryin' the Life out of Me", "Chet Baker", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=1PluX6pqKHs"]),
("Why Should I Lend You Mine (When You've Broken Yours Off Already)", "Brand X", "English", 4.87, 2.93, "Jazz", ["https://www.youtube.com/watch?v=GPmN2l_U9So"]),
("Kadia Blues", "Orchestre de la Paillote", "English", 3.3, 3.42, "Jazz", ["https://www.youtube.com/watch?v=mBz9QKICoL4"]),
("I Wish To Weep", "Dadafon", "English", 5.43, 4.01, "Jazz", ["https://www.youtube.com/watch?v=uOojAUFJgj8"]),
("Lazy", "Toby Lightman", "English", 5.24, 4.92, "Jazz", ["https://www.youtube.com/watch?v=Qq7SGKm3GPI"]),
("Ain't Misbehavin' (I'm Savin' My Love for You)", "Leon Redbone", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=F6d1-k2p1Ck"]),
("Lou's Blues", "Lou Donaldson", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=0QWwFl7gLA0"]),
("Diagonal S's at the Motel 6", "Lee Feldman", "English", 4.87, 2.93, "Jazz", ["https://www.youtube.com/watch?v=Xzd0s1HfujE"]),
("I Think About Your Body (Brixton Bounce)", "Us3", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=-DqXHVmOBhU"]),
("Your Good Thing (Is About To End)", "Boz Scaggs", "English", 3.24, 3.33, "Jazz", ["https://www.youtube.com/watch?v=6n0CEpPWVpA"]),
("Songs We Sing", "Assembly of Dust", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=Wgr1e3Y4gUM"]),
("Sookie Sookie (Live)", "Grant Green", "English", 3.24, 3.33, "Jazz", ["https://www.youtube.com/watch?v=Ev2o_KSd45Y"]),
("Summertime", "Morcheeba & Hubert Laws", "English", 1.52, 1.52, "Jazz", ["https://www.youtube.com/watch?v=_x8p2iUYArU"]),
("Pacific Haze", "Steve Howe's Remedy", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=YXKqJwtqYK8"]),
("Blue in Green", "Desert Island Dicks", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=5TLtcYuaOSk"]),
("Mississippi", "June Tabor & The Oyster Band", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=HivRMq_XQVk"]),
("How Long, How Long Blues (LP Version)", "Milt Jackson", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=loKscILy_9I"]),
("Lipcurl", "Scorched Earth", "English", 4.46, 3.05, "Jazz", ["https://www.youtube.com/watch?v=gbxvD6ggT8k"]),
("Adelaide Blues", "Southern Jazz Group", "English", 3.24, 3.33, "Jazz", ["https://www.youtube.com/watch?v=TrGqfDMB408"]),
("I Get Along Without You Very W", "Chet Baker", "English", 3.41, 2.7, "Jazz", ["https://www.youtube.com/watch?v=IgbPHTBiAVQ"]),
("Love Hangover", "JetTricKs feat. AdeFunKe", "English", 5.73, 4.67, "Jazz", ["https://www.youtube.com/watch?v=mTpz5aoxV3I"]),
("Félenko Féfé", "Momo Wandel Soumah", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=YqMuix2ji50"]),
("Is There Anybody Here That Love My Jesus", "Medeski Martin and Wood", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=4OQ2E2awvfY"]),
("(a1) park bench people", "Jose James", "English", 3.05, 3.05, "Jazz", ["https://www.youtube.com/watch?v=n5GCd9NytAE"]),
("02. Winobranie", "Muzykoterapia", "English", 4.74, 3.46, "Jazz", ["https://www.youtube.com/watch?v=vEZJcp9CiVs"]),
("Discipline", "Sun Ra", "English", 5.17, 2.83, "Jazz", ["https://www.youtube.com/watch?v=WgEbpWr2fVQ"]),
("The Three Faces of Balal", "Yusef Lateef", "English", 4.03, 3.48, "Jazz", ["https://www.youtube.com/watch?v=iGkKc52Cvxo"]),
("Hare Krishna", "Alice Coltrane", "English", 1.31, 2.93, "Jazz", ["https://www.youtube.com/watch?v=Hn36bzFZeg4"]),
("Fujiyama", "The Dave Brubeck Quartet", "English", 5.67, 2.85, "Jazz", ["https://www.youtube.com/watch?v=O1X8CvYAK-E"]),
("The Sun", "Alice Coltrane", "English", 2.91, 2.41, "Jazz", ["https://www.youtube.com/watch?v=IYjOJyAhFXo"]),
("Hear Me", "Koru", "English", 5.96, 3.04, "Jazz", ["https://www.youtube.com/watch?v=0Cuwhc7F8Vw"]),
("A Love Supreme", "Alice Coltrane", "English", 3.69, 2.94, "Jazz", ["https://www.youtube.com/watch?v=NsHt1wA8ul0"]),
("Flimmer", "Bugge Wesseltoft", "English", 5.62, 4.13, "Jazz", ["https://www.youtube.com/watch?v=XjbfK7RlQVE"]),
("The Healing Smoke", "Jan Garbarek", "English", 1.5, 1.65, "Jazz", ["https://www.youtube.com/watch?v=01MSnC-WO40"]),
("Sweet Earth Flying", "His Name Is Alive", "English", 0.68, 0.49, "Jazz", ["https://www.youtube.com/watch?v=bvzl5yv8wj4"]),
("Fly", "Ilhan Ersahin", "English", 5.59, 3.15, "Jazz", ["https://www.youtube.com/watch?v=5mnQzQp6sDk"]),
("Like Armstrong + Laika", "Tied & Tickled Trio", "English", 3.86, 3.02, "Jazz", ["https://www.youtube.com/watch?v=Z5QUzX3OrMg"]),
("Pinky", "Sarah Vaughan", "English", 3.47, 3.47, "Jazz", ["https://www.youtube.com/watch?v=My3Un7l6QnQ"]),
("Lovers' Infiniteness", "Ketil Bjørnstad", "English", 5.27, 3.26, "Jazz", ["https://www.youtube.com/watch?v=vi9f7utQXbU"]),
("Breathe", "Trygve Seim", "English", 2.57, 1.68, "Jazz", ["https://www.youtube.com/watch?v=aua_aY9B-WQ"]),
("Green Tea Farm", "Hiromi Uehara", "English", 3.88, 2.07, "Jazz", ["https://www.youtube.com/watch?v=5sXxTJGGaj0"]),
("Without You (Set You Free Album Version)", "Tammy Trent", "English", 5.01, 3.47, "Jazz", ["https://www.youtube.com/watch?v=kvBON4gfBv0"]),
("Reel Life (Evolution II)", "The Cinematic Orchestra", "English", 6.13, 3.59, "Jazz", ["https://www.youtube.com/watch?v=REkD2js-V6E"]),
("Lying Together", "FKJ", "English", 6.28, 2.83, "Jazz", ["https://www.youtube.com/watch?v=Em7bXENLp5o"]),
("Salvation", "Koop", "English", 6.84, 4.57, "Jazz", ["https://www.youtube.com/watch?v=kXRSP1gYvcw"]),
("Eyesore", "Maria Mena", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=vBS0lNL2IqQ"]),
("Blue Monk", "Abbey Lincoln", "English", 4.8, 3.6, "Jazz", ["https://www.youtube.com/watch?v=eOk6G_htqtM"]),
("Flavour", "Room Eleven", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=DwE0AkSVP48"]),
("Comes Love", "Billie Holiday", "English", 5.94, 4.19, "Jazz", ["https://www.youtube.com/watch?v=6FrwC0TsIvE"]),
("Don't Go to Strangers", "Etta Jones", "English", 7.46, 5.11, "Jazz", ["https://www.youtube.com/watch?v=VBLaJtXbpRg"]),
("All Things To All Men (feat. Roots Manuva)", "The Cinematic Orchestra", "English", 5.75, 3.14, "Jazz", ["https://www.youtube.com/watch?v=1EMVF8sCNFQ"]),
("Walkin' Shoes", "Gerry Mulligan", "English", 6.7, 2.67, "Jazz", ["https://www.youtube.com/watch?v=PXBq_BnEmS4"]),
("Tenderly (Mocky Remix)", "Anita O'Day", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=d4at_gUbY4I"]),
("Round Midnight", "Amy Winehouse", "English", 4.74, 3.23, "Jazz", ["https://www.youtube.com/watch?v=STDBSB-tdFA"]),
("Two Sleepy People", "Julie London", "English", 6.86, 4.37, "Jazz", ["https://www.youtube.com/watch?v=QumYfDXDFMY"]),
("Wise One", "John Coltrane", "English", 6.98, 2.87, "Jazz", ["https://www.youtube.com/watch?v=yrqb0373cVs"]),
("The Writer's Audience Is Fiction", "You.May.Die.In.The.Desert", "English", 7.74, 4.38, "Jazz", ["https://www.youtube.com/watch?v=bL-jYlukphA"]),
("Don't Be Sad", "Brad Mehldau", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=3gJz0w4PkW4"]),
("Sasa", "Medeski, Martin and Wood", "English", 7.04, 2.96, "Jazz", ["https://www.youtube.com/watch?v=KzxsUpuphYo"]),
("Night In Tunisia", "Metropolitan Jazz Affair", "English", 6.94, 3.88, "Jazz", ["https://www.youtube.com/watch?v=_cNc8uw01pw"]),
("Let's Waste Some Time", "Molly Johnson", "English", 6.48, 4.02, "Jazz", ["https://www.youtube.com/watch?v=LtqQaWzJKhM"]),
("As Time Goes By", "Julie London", "English", 7.62, 4.32, "Jazz", ["https://www.youtube.com/watch?v=zasZgmY3y1k"]),
("A Tribute to Don Johnson", "Pink Freud", "English", 7.04, 2.96, "Jazz", ["https://www.youtube.com/watch?v=W5EsUHGljYA"]),
("Yesterdays", "Helen Merrill", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=fAvT6JDJfP8"]),
("Do It The Hard Way", "The Jazz Renegades", "English", 6.86, 2.93, "Jazz", ["https://www.youtube.com/watch?v=tnX6w0Wc_0Y"]),
("It Never Entered My Mind", "Till Brönner", "English", 7.29, 3.13, "Jazz", ["https://www.youtube.com/watch?v=CgUcrStVWrQ"]),
("The Blues Are Brewin'", "Billie Holiday", "English", 6.96, 2.65, "Jazz", ["https://www.youtube.com/watch?v=bWtUzdI5hlE"]),
("'Round Midnight", "Sun Ra", "English", 7.67, 3.84, "Jazz", ["https://www.youtube.com/watch?v=IBFEc76kB18"]),
("Tarde", "Till Brönner", "English", 7.21, 3.65, "Jazz", ["https://www.youtube.com/watch?v=gFjR8aIgEbE"]),
("La Sirena", "Banyan", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=6CskFHAFh_Y"]),
("Thanks for Coming", "Touch and Go", "English", 6.83, 2.62, "Jazz", ["https://www.youtube.com/watch?v=5fW10OPeeTY"]),
("Stairway to the Stars", "Johnny Hartman", "English", 7.75, 4.4, "Jazz", ["https://www.youtube.com/watch?v=sRmXDGYlbdA"]),
("That Certain Feeling", "George Gershwin", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=_p8xKT0aZ80"]),
("Summer Nights", "Lonnie Liston Smith", "English", 7.0, 2.6, "Jazz", ["https://www.youtube.com/watch?v=76YH1VWA3MY"]),
("Acoustic Time", "Norman Brown", "English", 6.91, 2.68, "Jazz", ["https://www.youtube.com/watch?v=nnIZN2S8ne4"]),
("People Will Say We're in Love", "Frank Sinatra", "English", 7.62, 3.92, "Jazz", ["https://www.youtube.com/watch?v=Odb_36tsEnI"]),
("Once In Your Life", "Betty Carter", "English", 7.22, 3.54, "Jazz", ["https://www.youtube.com/watch?v=wC3cHfrckrE"]),
("Whispering", "Frank Sinatra", "English", 7.43, 3.81, "Jazz", ["https://www.youtube.com/watch?v=OB54zSvZ-RU"]),
("Rituals", "Nicola Conte", "English", 7.35, 3.38, "Jazz", ["https://www.youtube.com/watch?v=sqyXWw1B1Ds"]),
("Bliss", "Rosie Brown", "English", 6.96, 3.12, "Jazz", ["https://www.youtube.com/watch?v=ATMEvlKf7FM"]),
("Dream", "Goldfish", "English", 6.21, 2.87, "Jazz", ["https://www.youtube.com/watch?v=QdLlliazYD0"]),
("Nights at the Turntable", "Gerry Mulligan", "English", 6.83, 2.62, "Jazz", ["https://www.youtube.com/watch?v=k8PKTELkP-s"]),
("River Quay", "Pat Metheny", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=2iUYDoo-m_A"]),
("In My Bed (Bugz in the Attic Dub)", "Amy Winehouse", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=hqNk_yIAJpo"]),
("Take Five", "Charlie Parker", "English", 6.16, 3.03, "Jazz", ["https://www.youtube.com/watch?v=tT9Eh8wNMkw"]),
("I Love Paris", "Etta Jones", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=i0LYpFEFAZ0"]),
("Like A Star (Paris Session)", "Corinne Bailey Rae", "English", 6.04, 3.13, "Jazz", ["https://www.youtube.com/watch?v=gvH9Ccqk5qc"]),
("But Not for Me", "Doris Day", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=1w60ugvxzrg"]),
("Beast of Burden", "Donald Byrd", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=jMUUr_JmDH4"]),
("Wave", "Nancy Wilson", "English", 6.43, 2.47, "Jazz", ["https://www.youtube.com/watch?v=gBD8T_dsqw8"]),
("Sirens", "Jazzanova", "English", 7.15, 3.4, "Jazz", ["https://www.youtube.com/watch?v=1X7eAIR3X0k"]),
("Hit The Spot", "Leslie Mendelson", "English", 6.96, 2.65, "Jazz", ["https://www.youtube.com/watch?v=YXGcwJxN1WI"]),
("Dreamsville", "Pat Martino", "English", 7.36, 3.76, "Jazz", ["https://www.youtube.com/watch?v=LfYbWqdsQ-U"]),
("Greenest Grass (Live version)", "Room Eleven", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=IFKSX6GT46M"]),
("(A different) Bitch", "Room Eleven", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=5TLWMYmBqZc"]),
("Come Closer (Live version)", "Room Eleven", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=jAF1al6xysI"]),
("A Soft Place To Fall", "Nelson Rangell", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=ik5OF96Pbbs"]),
("Manoir Des Mes Reves", "Django Reinhardt", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=rG9vbKcWDE0"]),
("Feel The Vibe", "Loose Ends", "English", 7.03, 4.02, "Jazz", ["https://www.youtube.com/watch?v=gHXPr-GtYUk"]),
("High Night", "Till Brönner", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=0Kby_r8fbak"]),
("When My Angers Starts To Cry", "Beady Belle", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=AKYJcddcnE4"]),
("Só Danço Samba", "Till Brönner", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=EjLZTaXWdnM"]),
("Smooth Jazz Christmas Overture", "Dave Koz", "English", 6.96, 2.65, "Jazz", ["https://www.youtube.com/watch?v=Vx3q2jlQQSo"]),
("Summertime", "Julie London", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=cJj_K5I3fj8"]),
("Tudo Que Voce Podia Ser", "Azymuth", "English", 6.87, 3.37, "Jazz", ["https://www.youtube.com/watch?v=WjfuLpoRy6k"]),
("Memphis in June", "Julie London", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=zIvpdzZLEU8"]),
("Lover Man (Oh, Where Can You Be)", "Billie Holiday", "English", 5.91, 3.57, "Jazz", ["https://www.youtube.com/watch?v=soJwavwA-Os"]),
("Sun-Earth Rock", "Sun Ra", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=AwZ6Mf59ZOQ"]),
("Noon At The Moon", "Calm", "English", 6.19, 2.96, "Jazz", ["https://www.youtube.com/watch?v=Cj8ytmphxUk"]),
("Scar", "Joe Henry", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=YtG3VUlDW-0"]),
("Let It Snow! Let It Snow! Let It Snow!", "The Manhattan Transfer", "English", 5.34, 3.05, "Jazz", ["https://www.youtube.com/watch?v=Gis5jFbkxlw"]),
("In Between the Heartaches", "George Duke", "English", 7.04, 2.96, "Jazz", ["https://www.youtube.com/watch?v=NMU_g4K06DE"]),
("L", "Nat King Cole", "English", 7.22, 3.68, "Jazz", ["https://www.youtube.com/watch?v=f_HmF84G7ZY"]),
("One Note Samba (Part 2)", "Modern Jazz Quartet With Laurindo Almeida", "English", 7.96, 4.75, "Jazz", ["https://www.youtube.com/watch?v=5rzdodR2NAM"]),
("A Child Is Born", "Paul McCandless", "English", 6.96, 2.65, "Jazz", ["https://www.youtube.com/watch?v=jS0bOHWU_mU"]),
("Some Traveling Music", "Frank Sinatra", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=sDCfULV4h6I"]),
("A Taste Of Honey", "Chet Baker", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=c9xv7W_dg5Y"]),
("A Stomach is Burning", "Melanie De Biasio", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=voe5qhG_kQU"]),
("I Remember Clifford", "Art Blakey & The Jazz Messengers", "English", 7.51, 3.31, "Jazz", ["https://www.youtube.com/watch?v=KVvRZWhFF4w"]),
("First-Time Love", "Dave Grusin", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=CQuv3F5sULk"]),
("Mystic Bounce (Madlib remix)", "Ronnie Foster", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=W2eAKUX2CEA"]),
("Take The L Train (To 8th Ave.)", "Brooklyn Funk Essentials", "English", 6.57, 3.28, "Jazz", ["https://www.youtube.com/watch?v=Rxcb3m5C_VE"]),
("Gabriel's Oboe", "Chris Botti", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=3efw2hzl6BQ"]),
("Get It On", "Monodeluxe", "English", 7.08, 3.58, "Jazz", ["https://www.youtube.com/watch?v=zS8GrWtmOrY"]),
("The Breeze", "Leena Conquest", "English", 6.83, 2.62, "Jazz", ["https://www.youtube.com/watch?v=xyuSM2Zw6RE"]),
("Byrd Plays", "Incognito", "English", 6.96, 2.65, "Jazz", ["https://www.youtube.com/watch?v=h9ilM924kdk"]),
("59", "Jacky Terrasson", "English", 6.41, 3.06, "Jazz", ["https://www.youtube.com/watch?v=SZgHi4QYl88"]),
("The Sirens' Call", "Jazzanova", "English", 6.6, 4.01, "Jazz", ["https://www.youtube.com/watch?v=ACX99zAYAQ0"]),
("Faster Than An Arrow", "Frank Gambale", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=4tftAZ09lnI"]),
("Dinosaur", "Al Jarreau", "English", 6.28, 2.83, "Jazz", ["https://www.youtube.com/watch?v=EBPiU7Ealrw"]),
("Liquid Streets", "Roy Hargrove", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=GFLdjFGXHN4"]),
("Meditation", "TOM & JOY", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=s6yZ_qmT-og"]),
("I Dreamed In The Cities At Night", "Benjamin Herman", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=kioEd3A2D1c"]),
("Proof", "Pat Metheny", "English", 7.38, 2.92, "Jazz", ["https://www.youtube.com/watch?v=FWEWXV42QiM"]),
("Song For the Oud", "James Uhart", "English", 6.74, 3.11, "Jazz", ["https://www.youtube.com/watch?v=c7Ecsr7-pSw"]),
("Summertime", "Johnny Hartman", "English", 7.25, 2.49, "Jazz", ["https://www.youtube.com/watch?v=ORPD0jIc7AE"]),
("Birds Of Paradise", "Carla Bley", "English", 6.92, 2.84, "Jazz", ["https://www.youtube.com/watch?v=zTE_yz5awlM"]),
("Moon Over Naples", "Bert Kaempfert", "English", 6.32, 3.99, "Jazz", ["https://www.youtube.com/watch?v=k7vefHS7rW4"]),
("Alabama", "John Coltrane", "English", 2.52, 2.55, "Jazz", ["https://www.youtube.com/watch?v=saN1BwlxJxA"]),
("Hey Eugene", "Pink Martini", "English", 7.1, 4.48, "Jazz", ["https://www.youtube.com/watch?v=VI0E-Gy4A9E"]),
("Moonlight Serenade", "Frank Sinatra", "English", 6.75, 3.04, "Jazz", ["https://www.youtube.com/watch?v=wtP3zld5jZY"]),
("Le Poisson Des Mers Du Sud", "Isabelle Antena", "English", 7.23, 3.28, "Jazz", ["https://www.youtube.com/watch?v=WiciTWHB4SU"]),
("Samba Em Prelúdio", "Esperanza Spalding", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=8vOkY94xjnY"]),
("Audrey", "Dave Brubeck", "English", 6.73, 2.22, "Jazz", ["https://www.youtube.com/watch?v=hASG4nniLQs"]),
("Lover's Leap", "Béla Fleck and the Flecktones", "English", 6.43, 3.08, "Jazz", ["https://www.youtube.com/watch?v=SaFOGVyHOOw"]),
("Besame Mucho", "Dave Brubeck", "English", 6.11, 2.58, "Jazz", ["https://www.youtube.com/watch?v=P_a2f9HI2JI"]),
("Everything I Love Featuring Nas & Cee-Lo (Explicit Album Version)", "Diddy", "English", 6.38, 2.4, "Jazz", ["https://www.youtube.com/watch?v=dTsSA3zymUg"]),
("I X Love", "Charles Mingus", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=xeFrreOGfjM"]),
("A Man Alone", "Frank Sinatra", "English", 4.58, 2.7, "Jazz", ["https://www.youtube.com/watch?v=Js2hLlNooAk"]),
("Wolverine Blues", "Jelly Roll Morton", "English", 4.91, 3.75, "Jazz", ["https://www.youtube.com/watch?v=ID79UK_JUpw"]),
("River", "Herbie Hancock", "English", 6.47, 2.69, "Jazz", ["https://www.youtube.com/watch?v=xrBilBQ1C54"]),
("Dear Prudence", "Brad Mehldau", "English", 3.52, 0.95, "Jazz", ["https://www.youtube.com/watch?v=2bCueJXqLYQ"]),
("Sunset Road", "Béla Fleck and the Flecktones", "English", 6.93, 2.21, "Jazz", ["https://www.youtube.com/watch?v=uTfe8gVQ4A4"]),
("Someone's Rocking My Dreamboat", "The Ink Spots", "English", 3.38, 3.08, "Jazz", ["https://www.youtube.com/watch?v=1QhP8kGt3JY"]),
("Some Enchanted Evening", "Frank Sinatra", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=ng3XJnC8IN8"]),
("If I Could See", "Mahavishnu Orchestra", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=J9lvqaIAKpM"]),
("Koto Song", "Dave Brubeck", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=WbjEImQybec"]),
("Dusty McNugget", "Brad Mehldau", "English", 5.58, 2.79, "Jazz", ["https://www.youtube.com/watch?v=hLXA6gyQEBw"]),
("Ginza Samba", "Vince Guaraldi", "English", 7.05, 1.91, "Jazz", ["https://www.youtube.com/watch?v=ennmuhY9OxY"]),
("Trav'lin' Light", "Chet Baker", "English", 6.11, 2.96, "Jazz", ["https://www.youtube.com/watch?v=MarhbIP6LmE"]),
("Ain't Misbehavin'", "Dinah Washington", "English", 6.97, 4.22, "Jazz", ["https://www.youtube.com/watch?v=v7lRDPtkrEs"]),
("You'll Never Walk Alone", "Frank Sinatra", "English", 6.34, 3.4, "Jazz", ["https://www.youtube.com/watch?v=OV5_LQArLa0"]),
("Soon As I Get Home", "Lizz Wright", "English", 7.29, 3.4, "Jazz", ["https://www.youtube.com/watch?v=YDonbahSk4c"]),
("Autumn of Our Love", "Spyro Gyra", "English", 6.42, 2.36, "Jazz", ["https://www.youtube.com/watch?v=vNuBdDMBjFY"]),
("Out of Nowhere", "Dave Brubeck", "English", 6.11, 2.58, "Jazz", ["https://www.youtube.com/watch?v=Z4Zq1-qfA5U"]),
("Rue Fontaine", "Bexarametric, Tedzsee & Burt Morton", "English", 3.78, 3.3, "Jazz", [""]),
("Manteca Theme", "Dizzy Gillespie", "English", 4.38, 3.55, "Jazz", ["https://www.youtube.com/watch?v=A5tRGMHfKrE"]),
("I Wanna Get Married", "Nellie McKay", "English", 6.74, 4.22, "Jazz", ["https://www.youtube.com/watch?v=8Lf-UHVpbvY"]),
("Love Potion #9", "Herb Alpert and the Tijuana Brass", "English", 5.42, 4.39, "Jazz", ["https://www.youtube.com/watch?v=4AjUHjEYCfo"]),
("This Is Mr Noel Gallagher", "Burnage Boys", "English", 6.12, 3.91, "Jazz", ["https://www.youtube.com/watch?v=WK1svG7uLpI"]),
("Miles Davis", "Rainald Grebe", "English", 5.42, 4.39, "Jazz", ["https://www.youtube.com/watch?v=5weypp358Wk"]),
("Acapella", "A-Live", "English", 5.42, 4.39, "Jazz", ["https://www.youtube.com/watch?v=S7dcA_N6EmQ"]),
("Molecular Structure", "Mose Allison", "English", 4.42, 4.0, "Jazz", ["https://www.youtube.com/watch?v=BiNVwrag0Lk"]),
("Simone", "Boz Scaggs", "English", 2.97, 3.46, "Jazz", ["https://www.youtube.com/watch?v=--c_2dyMYhg"]),
("Really", "Nellie McKay", "English", 4.7, 4.22, "Jazz", ["https://www.youtube.com/watch?v=40VVkbxzJzE"]),
("Somebody's Body", "George Duke", "English", 3.59, 3.35, "Jazz", ["https://www.youtube.com/watch?v=ZirO-wSVM1E"]),




# ("Invaders Must Die", "Prodigy", "English", 4.68, 5.29, "Electronic", ["https://www.youtube.com/watch?v=EiqFcc_l_Kk"]),
# ("Come to Daddy", "Aphex Twin", "English", 3.89, 4.84, "Electronic", ["https://www.youtube.com/watch?v=TZ827lkktYs"]),
# ("Pills", "Primal Scream", "English", 3.38, 5.12, "Electronic", ["https://www.youtube.com/watch?v=vjAgDSlfMOI"]),
# ("Acid Annie", "Natalia Kills", "English", 3.08, 5.87, "Electronic", ["https://www.youtube.com/watch?v=gVnGYEIvTdE"]),
# ("Tension", "Orbital", "English", 2.61, 3.71, "Electronic", ["https://www.youtube.com/watch?v=yqJS5hbZw-0"]),
# ("The Carpathians", "Ben Frost", "English", 3.24, 5.72, "Electronic", ["https://www.youtube.com/watch?v=HDWpI8PwZI0"]),
# ("The Way It Is (live remix)", "The Prodigy", "English", 3.08, 5.87, "Electronic", ["https://www.youtube.com/watch?v=kCx5W2gKoME"]),
# ("American Trash", "Innerpartysystem", "English", 5.16, 5.23, "Electronic", ["https://www.youtube.com/watch?v=AqtXtnUGPiA"]),
# ("Come To Daddy, Pappy Mix", "Aphex Twin", "English", 3.47, 4.88, "Electronic", ["https://www.youtube.com/watch?v=TZ827lkktYs"]),
# ("Sphere", "John Frusciante", "English", 3.08, 5.87, "Electronic", ["https://www.youtube.com/watch?v=qcATA_snLWk"]),
# ("Come to Daddy [Pappy Mix]", "Aphex Twin", "English", 4.33, 6.01, "Electronic", ["https://www.youtube.com/watch?v=TZ827lkktYs"]),
# ("Control Freak", "Recoil", "English", 4.55, 6.01, "Electronic", ["https://www.youtube.com/watch?v=IPQUfkmdFQs"]),
# ("Elephant", "Idiot Pilot", "English", 5.33, 5.99, "Electronic", ["https://www.youtube.com/watch?v=MZf0NkR2Gm8"]),
# ("Super Space Invaders", "Eisenfunk", "English", 5.39, 5.11, "Electronic", ["https://www.youtube.com/watch?v=WhaVFlSxmjk"]),
# ("Cobrastyle", "Teddy Bears", "English", 7.03, 6.21, "Electronic", ["https://www.youtube.com/watch?v=LMKsY4wzoxY"]),
# ("Power to the Beats", "Utah Saints", "English", 5.08, 5.8, "Electronic", ["https://www.youtube.com/watch?v=RnTHnfNnC1Q"]),
# ("Spiritual Trance", "Infected Mushroom", "English", 4.7, 4.47, "Electronic", ["https://www.youtube.com/watch?v=0IARjKJOZ2w"]),
# ("Jenny (feat. Spalding Rockwell)", "Armand van Helden", "English", 5.55, 4.92, "Electronic", ["https://www.youtube.com/watch?v=B3w9C1JdpR8"]),
# ("Decent Cancer", "Ashbury Heights", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=l5_6xGskygI"]),
# ("Fama Tuba", "Helium Vola", "English", 4.72, 4.58, "Electronic", ["https://www.youtube.com/watch?v=BAmPrmjh748"]),
# ("Young Boys", "Lords of Acid", "English", 6.55, 6.61, "Electronic", ["https://www.youtube.com/watch?v=r7KvxOxgc9g"]),
# ("El Consuelo Final", "Amduscia", "English", 4.08, 5.2, "Electronic", ["https://www.youtube.com/watch?v=xQIlW6zyhUk"]),
# ("The Girls", "Diorama", "English", 3.19, 5.77, "Electronic", ["https://www.youtube.com/watch?v=GIN9x3mLUpk"]),
# ("Dusted Decks", "The Chemical Brothers", "English", 4.16, 4.93, "Electronic", ["https://www.youtube.com/watch?v=RAN2tDnVY1w"]),
# ("Real Solution #9 [Mambo Mania Mix]", "White Zombie", "English", 3.08, 5.87, "Electronic", ["https://www.youtube.com/watch?v=Cvlpktjsq0A"]),
# ("I'm Gonna Show You Crazy", "Bebe Rexha", "English", 2.53, 6.2, "Electronic", ["https://www.youtube.com/watch?v=kEDZZin4_eM"]),
# ("The Childcatcher", "Patrick Wolf", "English", 4.84, 4.75, "Electronic", ["https://www.youtube.com/watch?v=TZc_YPXTHIU"]),
# ("Trainwreck", "Banks", "English", 2.53, 6.2, "Electronic", ["https://www.youtube.com/watch?v=vSNqIerZBFs"]),
# ("Obsession", "Gesaffelstein", "English", 3.88, 5.1, "Electronic", ["https://www.youtube.com/watch?v=lM9rRMJ3Ayc"]),
# ("Stillbirth", "Alice Glass", "English", 4.23, 4.79, "Electronic", ["https://www.youtube.com/watch?v=m8iSUekqabI"]),
# ("Mouth (The Stingray Mix)", "Bush", "English", 6.0, 5.42, "Electronic", ["https://www.youtube.com/watch?v=JCO92__Zn-8"]),
# ("Motorhead", "Primal Scream", "English", 3.23, 2.62, "Electronic", ["https://www.youtube.com/watch?v=uIasD6UYlcc"]),
# ("Stress", "Crim3s", "English", 2.53, 6.2, "Electronic", ["https://www.youtube.com/watch?v=wCT6HoJB2a0"]),
# ("Cowboy Bob", "Butthole Surfers", "English", 2.53, 6.2, "Electronic", ["https://www.youtube.com/watch?v=uZ_NEY3aHFw"]),
# ("Mao Tse Tung Said", "Alabama 3", "English", 5.78, 5.0, "Electronic", ["https://www.youtube.com/watch?v=Ea9b2oHX6C8"]),
# ("Pretty Little Eyes", "The Presets", "English", 5.93, 6.01, "Electronic", ["https://www.youtube.com/watch?v=xLC-U_SAMEQ"]),
# ("Stalker", "Recoil", "English", 4.74, 3.98, "Electronic", ["https://www.youtube.com/watch?v=Rdg-U1PNGtw"]),
# ("Pressure Torture", "Venetian Snares", "English", 4.55, 5.95, "Electronic", ["https://www.youtube.com/watch?v=Bku2jgyN21g"]),
# ("They Want Us To Make A Symphon", "Le Tigre", "English", 3.04, 3.2, "Electronic", ["https://www.youtube.com/watch?v=DQcIS1qtzz4"]),
# ("Uuu...", "Jyoji", "English", 5.75, 4.6, "Electronic", ["https://www.youtube.com/watch?v=H8K_hVf_PI0"]),
# ("Cobalto Azul", "Jyoji", "English", 6.06, 4.56, "Electronic", ["https://www.youtube.com/watch?v=u6Aar8b3uwU"]),
# ("Overdrive Generation", "Kimza International", "English", 5.75, 4.66, "Electronic", [""]),
# ("8 Hour Fake (Prototune)", "Kimza International", "English", 5.94, 4.41, "Electronic", [""]),
# ("Busting Barriers", "Neo-Rista", "English", 5.94, 4.52, "Electronic", ["https://www.youtube.com/watch?v=ciV_5zuWaao"]),
# ("Welcome To The Breakdown", "I Fight Dragons", "English", 2.06, 2.95, "Electronic", ["https://www.youtube.com/watch?v=vrmLVO_yHX0"]),
# ("EXEC_PAJA/.#Misya extracting", "志方あきこ", "English", 5.31, 4.84, "Electronic", ["https://www.youtube.com/watch?v=9Rmwxx3scuA"]),
# ("Turbolence", "Deadbeat", "English", 4.12, 5.9, "Electronic", ["https://www.youtube.com/watch?v=JXy0XnzTQuc"]),
# ("Justice of the Karma Law", "Tangerine Dream", "English", 5.43, 4.75, "Electronic", ["https://www.youtube.com/watch?v=z6RWBC2mG_I"]),
# ("満ち潮の夜", "Zabadak", "English", 5.59, 4.49, "Electronic", ["https://www.youtube.com/watch?v=Q4QDMpxTmTY"]),
# ("Hanakisou", "志方あきこ", "English", 3.9, 4.53, "Electronic", ["https://www.youtube.com/watch?v=CeSFzWWcAv0"]),
# ("セイレーン － Ήρωες Αργοναύτες －", "志方あきこ", "English", 5.58, 4.29, "Electronic", ["https://www.youtube.com/watch?v=TUpVx1f6Y6A"]),
# ("Rovina", "志方あきこ", "English", 5.97, 4.48, "Electronic", ["https://www.youtube.com/watch?v=VsTjhyuw1lE"]),
# ("We Had Such A Lovely Time", "Keshco", "English", 5.39, 4.68, "Electronic", ["https://www.youtube.com/watch?v=oSN5aNHixkI"]),
# ("Sexual", "Goddess", "English", 5.63, 4.72, "Electronic", ["https://www.youtube.com/watch?v=Nj3ffdoQnJM"]),
# ("Indecent Majority", "Keshco", "English", 5.69, 4.92, "Electronic", [""]),
# ("Living Sacrifice", "Tomoko Imoto", "English", 5.39, 4.27, "Electronic", [""]),
# ("Rovaio", "志方あきこ", "English", 5.84, 4.74, "Electronic", [""]),
# ("Lingerie", "Goddess", "English", 5.26, 4.72, "Electronic", ["https://www.youtube.com/watch?v=cm8mCEy_OUk"]),
# ("Preamble", "Kalliope", "English", 6.0, 4.46, "Electronic", ["https://www.youtube.com/watch?v=kDvRypydv-A"]),
# ("Je T'Aime", "Goddess", "English", 5.98, 4.45, "Electronic", ["https://www.youtube.com/watch?v=t_iimkho2kA"]),
# ("幕間 ~カオカゲウオ:その頃団員達は見当違いな方向を大捜索中", "志方あきこ", "English", 6.1, 4.44, "Electronic", ["https://www.youtube.com/watch?v=ClyLXBVu7RU"]),
# ("She Walks in Beauty", "Kalliope", "English", 6.2, 4.55, "Electronic", ["https://www.youtube.com/watch?v=DJgg2zp0sk8"]),
# ("Crossing the Rubicon", "Kalliope", "English", 6.02, 4.46, "Electronic", ["https://www.youtube.com/watch?v=wuax6BNxILA"]),
# ("Charon's Valley", "Kalliope", "English", 6.2, 4.55, "Electronic", ["https://www.youtube.com/watch?v=-ebUaxGZt2E"]),
# ("The Fall of Rome", "Kalliope", "English", 6.18, 4.61, "Electronic", ["https://www.youtube.com/watch?v=EDNLNOdvQIo"]),
# ("Alvorada", "Stupid", "English", 6.02, 4.46, "Electronic", ["https://www.youtube.com/watch?v=mhMhodStpE4"]),
# ("Loki", "Shikata Akiko & Shimotsuki Haruka & Mitose Noriko", "English", 5.95, 4.37, "Electronic", ["https://www.youtube.com/watch?v=GoBa408iCus"]),
# ("Feyd Rautha Dark Heart", "Grimes", "English", 3.89, 4.51, "Electronic", ["https://www.youtube.com/watch?v=do-mtWMXiPA"]),
# ("Beast Infection", "Grimes", "English", 3.71, 4.83, "Electronic", ["https://www.youtube.com/watch?v=Vm_nE4YEYHQ"]),
# ("Shadout Mapes", "Grimes", "English", 4.02, 4.75, "Electronic", ["https://www.youtube.com/watch?v=PwPi43oB6vM"]),
# ("I Am Shell I Am Bone", "Gazelle Twin", "English", 4.09, 4.74, "Electronic", ["https://www.youtube.com/watch?v=t01dYTecfS8"]),
# ("I Turn My Arm (Renaissance Man Remix)", "Gazelle Twin", "English", 3.05, 6.24, "Electronic", ["https://www.youtube.com/watch?v=lHxzDgSRn2M"]),
# ("Red Giant", "Robert Tunstall", "English", 5.71, 4.53, "Electronic", ["https://www.youtube.com/watch?v=PQmDUEv939A"]),
# ("Blow Hole", "Zan-zan-zawa-veia", "English", 4.59, 5.05, "Electronic", [""]),
# ("Very Happy", "Oh Yes, By All Means", "English", 2.99, 3.99, "Electronic", ["https://www.youtube.com/watch?v=iQQaJnhz3Cs"]),
# ("Vile Core", "Zan-zan-zawa-veia", "English", 4.75, 5.18, "Electronic", ["https://www.youtube.com/watch?v=qED9FfGykhA"]),
# ("New Living Dead", "The Demonic Paradise", "English", 3.25, 5.93, "Electronic", ["https://www.youtube.com/watch?v=l_ulVpYs6vg"]),
# ("Beck Depression Inventory", "My Boyfriend the Pilot", "English", 3.05, 6.24, "Electronic", ["https://www.youtube.com/watch?v=nKz5eJXXKm4"]),
# ("Kill", "Moth", "English", 3.05, 6.24, "Electronic", ["https://www.youtube.com/watch?v=DYeBhC1-6bw"]),
# ("NELLIE ..", "BennY RevivaL", "English", 3.05, 6.24, "Electronic", ["https://www.youtube.com/watch?v=CMJEKP66fsg"]),
# ("Worm Lake", "Zan-zan-zawa-veia", "English", 4.81, 4.67, "Electronic", [""]),
# ("Hairy Candy", "Tobacco", "English", 6.53, 5.21, "Electronic", ["https://www.youtube.com/watch?v=10nWEnTKCNY"]),
# ("Fa", "Fennesz", "English", 4.14, 6.62, "Electronic", ["https://www.youtube.com/watch?v=cAqQk8xj-Wc"]),
# ("Signal", "Warren Suicide", "English", 6.55, 5.63, "Electronic", ["https://www.youtube.com/watch?v=jZPiPyTeJ3g"]),
# ("Mr. Fingers", "Animal Collective", "English", 5.76, 5.88, "Electronic", ["https://www.youtube.com/watch?v=sUtOZ57Qqro"]),
# ("Night section", "Data Romance", "English", 6.07, 5.25, "Electronic", ["https://www.youtube.com/watch?v=HCAOOpNLvGM"]),
# ("Purple Haze", "Tangerine Dream", "English", 4.14, 6.62, "Electronic", ["https://www.youtube.com/watch?v=JanNdU7Zu9I"]),
# ("Break", "Endless Blue", "English", 5.67, 5.71, "Electronic", ["https://www.youtube.com/watch?v=Nquull_4p9Y"]),
# ("batman dancefloor", "Nebulo", "English", 2.07, 3.31, "Electronic", ["https://www.youtube.com/watch?v=75d5-SqD6DU"]),
# ("Fog - Bloody Beetroots Rmx24bit MST", "Fact", "English", 6.04, 6.79, "Electronic", [""]),
# ("skeleton dancers", "Ostranenie", "English", 4.61, 5.36, "Electronic", ["https://www.youtube.com/watch?v=X8xEkBeAqtU"]),
# ("Torn Strings", "AWT", "English", 4.84, 4.81, "Electronic", ["https://www.youtube.com/watch?v=sFA4i4ZKLGo"]),
# ("Photon Castle", "Skerror", "English", 4.61, 5.36, "Electronic", [""]),
# ("ㄴㄱㄴㄱㅈ", "Jibby", "English", 5.05, 5.4, "Electronic", [""]),
# ("Bay Leaf x Jibby - A - ZERO (어지러)", "Jibby", "English", 5.18, 5.43, "Electronic", ["https://www.youtube.com/watch?v=-LI-i3H1yow"]),
# ("Poik", "frifrafro", "English", 4.14, 6.62, "Electronic", ["https://www.youtube.com/watch?v=oorVWW9ywG0"]),
# ("Chain Hang Low", "Jibby", "English", 5.05, 5.4, "Electronic", ["https://www.youtube.com/watch?v=Nc13qQe3TPo"]),
# ("Does Your Chain Hang Low?", "Jibby", "English", 5.05, 5.4, "Electronic", ["https://www.youtube.com/watch?v=Nc13qQe3TPo"]),
# ("Army Of Me -21st", "Björk", "English", 4.59, 6.03, "Electronic", ["https://www.youtube.com/watch?v=jPeheoBa2_Y"]),
# ("Bird Flu", "M.I.A.", "English", 6.54, 6.06, "Electronic", ["https://www.youtube.com/watch?v=6Kq16GAuNcg"]),
# ("You Don't Know Me", "Jax Jones", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=WH9C6oLEtOg"]),
# ("Difficult", "Uffie", "English", 6.34, 5.4, "Electronic", ["https://www.youtube.com/watch?v=HVpir5vSA78"]),
# ("22nd Century", "Kelis", "English", 5.91, 5.09, "Electronic", ["https://www.youtube.com/watch?v=OprOS-GJr14"]),
# ("Colouring of Pigeons", "The Knife", "English", 6.67, 5.15, "Electronic", ["https://www.youtube.com/watch?v=FaT7ZCxI71k"]),
# ("No Enemiesz", "Kiesza", "English", 4.39, 4.88, "Electronic", ["https://www.youtube.com/watch?v=uxY0To9Cb2g"]),
# ("Corporate Cannibal", "Grace Jones", "English", 4.62, 5.44, "Electronic", ["https://www.youtube.com/watch?v=FgMn2OJmx3w"]),
# ("Swingbreaks", "Parov Stelar", "English", 4.62, 4.85, "Electronic", ["https://www.youtube.com/watch?v=IvwGcYqQz4A"]),
# ("Internacional", "Brazilian Girls", "English", 5.71, 6.25, "Electronic", ["https://www.youtube.com/watch?v=LUQsYUlSmGY"]),
# ("Circles", "Nelly Furtado", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=8GlxlKvIEBk"]),
# ("Face to Face/Short Circuit", "Daft Punk", "English", 6.13, 5.1, "Electronic", ["https://www.youtube.com/watch?v=dKJfJMMsqX4"]),
# ("Snot", "LFO", "English", 5.18, 5.39, "Electronic", ["https://www.youtube.com/watch?v=v3gpK9ZrEVY"]),
# ("Puppet", "Alaska Thunderfuck", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=CvAxkG4-S6Y"]),
# ("Gimme Your Money", "Annie", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=ICj75Zzl9QA"]),
# ("Kaboom (feat Kalenna)", "Lady Gaga", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=iGjxngVwwRg"]),
# ("Fallen", "One Dove", "English", 6.31, 5.8, "Electronic", ["https://www.youtube.com/watch?v=tX4IfDBcf28"]),
# ("That Kind of Girl", "Nadia Oh", "English", 6.08, 5.88, "Electronic", ["https://www.youtube.com/watch?v=ur76jAuLitE"]),
# ("Bubble (Speechbubble)", "Fluke", "English", 5.41, 4.57, "Electronic", ["https://www.youtube.com/watch?v=4aqaw92wzpk"]),
# ("Fall for You", "Kylie Minogue", "English", 5.71, 6.25, "Electronic", ["https://www.youtube.com/watch?v=n1ZzVr9wFuQ"]),
# ("Keep The Trance", "Madonna", "English", 6.91, 6.11, "Electronic", ["https://www.youtube.com/watch?v=WYsC6aCqywM"]),
# ("Summer-Spring", "Juvelen", "English", 4.41, 4.38, "Electronic", ["https://www.youtube.com/watch?v=Hfg4-dTQi4w"]),
# ("Post-Modern Sleaze (Flight from Nashville)", "Sneaker Pimps", "English", 3.85, 5.35, "Electronic", ["https://www.youtube.com/watch?v=WFB31_8V6Ms"]),
# ("A Quien Le Importa", "Fangoria", "English", 5.47, 5.2, "Electronic", ["https://www.youtube.com/watch?v=XX_hWpPnd3I"]),
# ("Bangalicious feat. 土屋アンナ", "ravex", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=Rj3H4yjnzzw"]),
# ("フレアスタック", "Aural Vampire", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=ol7VeMIl9fs"]),
# ("Cotton Candy", "Amanda Lepore", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=XRoTLMdNrWY"]),
# ("We Don't Care [Dirty Version]", "Audio Bullys", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=QXMUY4deXs0"]),
# ("Romeo (Beats Mix)", "Basement Jaxx", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=x2wUbgAAydY"]),
# ("Let It Will Be (Paper Faces Vocal Edit)", "Madonna", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=ABWGSCaDUQ0"]),
# ("Fame Kills", "Dangerous Muse", "English", 6.16, 5.16, "Electronic", ["https://www.youtube.com/watch?v=aegcTvy0n80"]),
# ("Spooky (Boo! Dub mix)", "New Order", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=rIfUb_lvhKY"]),
# ("Loops of Drum and Bass", "The Chemical Brothers", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=N2QE_2q-CUk"]),
# ("Live Life Now", "Cheryl Cole", "English", 3.05, 5.04, "Electronic", ["https://www.youtube.com/watch?v=5PcURA-ObLA"]),
# ("She's My Man (Goose Remix)", "Scissor Sisters", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=LFpTIfcRqFA"]),
# ("Fuck You And Your Money", "MiChi", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=VqmiMhpln2s"]),
# ("Spin Spin Sugar (radio mix)", "Sneaker Pimps", "English", 5.93, 5.51, "Electronic", ["https://www.youtube.com/watch?v=MZWWI3ryw2o"]),
# ("Slow (Extended Mix)", "Kylie Minogue", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=Omrp4QR_Rpo"]),
# ("Beauty Survivor", "Angelababy", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=OzrL3qqkD9I"]),
# ("Make It Bump feat. 倖田來未", "Far East Movement", "English", 6.27, 6.43, "Electronic", ["https://www.youtube.com/watch?v=14fgSywDa4U"]),
# ("torn", "Ferri", "English", 6.2, 5.77, "Electronic", ["https://www.youtube.com/watch?v=ZhM7ovjWGgg"]),
# ("Did It Again (Remix) (Feat. Kid Cudi)", "Shakira", "English", 7.1, 6.05, "Electronic", ["https://www.youtube.com/watch?v=xE2onUOa5u4"]),
# ("eternal return", "Ferri", "English", 5.59, 5.25, "Electronic", ["https://www.youtube.com/watch?v=GbjYOoXugwU"]),
# ("Making A Scene", "Fritz Helder & the Phantoms", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=1emvZ7chc7g"]),
# ("Hallows (Alternate Version)", "Lady Laudanum", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=zBdjy0joY5Q"]),
# ("Freed From Desire the un-remix", "Gala", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=p3l7fgvrEKM"]),
# ("Shove It", "Santigold", "English", 4.9, 5.83, "Electronic", ["https://www.youtube.com/watch?v=Tg9cN2A7-cA"]),
# ("Esta Noche es Nuestra", "Naty Botero", "English", 5.7, 6.03, "Electronic", ["https://www.youtube.com/watch?v=_eAzAPZdATA"]),
# ("Tis Agapis Ta Thymata", "Ivi Adamou", "English", 6.18, 6.01, "Electronic", ["https://www.youtube.com/watch?v=fOJfeNA7x4w"]),
# ("Ouch That Feels Good", "Dale Bozzio", "English", 6.48, 6.08, "Electronic", ["https://www.youtube.com/watch?v=W_Pke5NWD0c"]),
# ("Не обижай меня", "Art Voyage", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=Ougtzba0yEU"]),
# ("Boulevard of Broken Songs [Green Day vs. Oasis vs. Travis vs. Aerosmith vs. Eminem]", "Party Ben", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=wvwzDogVBf8"]),
# ("Friday Fierce 12-31-2010", "Brian Ashby", "English", 4.0, 5.7, "Electronic", [""]),
# ("Black Widow (Rennie Pilgrem Mix)", "Überzone & Rennie Pilgrem", "English", 4.0, 5.7, "Electronic", ["https://www.youtube.com/watch?v=xc7gosQ4YUc"]),
# ("Firestarter", "Prodigy", "English", 3.78, 5.13, "Electronic", ["https://www.youtube.com/watch?v=wmin5WkOuPw"]),
# ("How to Be Eaten by a Woman", "The Glitch Mob", "English", 6.24, 4.79, "Electronic", ["https://www.youtube.com/watch?v=Y0zvkthoxwU"]),
# ("Pop Song", "Starfucker", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=huzalkQKRTw"]),
# ("Avalanche", "Zola Jesus", "English", 5.48, 4.66, "Electronic", ["https://www.youtube.com/watch?v=kUMhCfhWZ40"]),
# ("Swords", "Zola Jesus", "English", 4.36, 4.4, "Electronic", ["https://www.youtube.com/watch?v=Y7ePdQizTkM"]),
# ("$$ Troopers", "Huoratron", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=PPwCmhvmHeM"]),
# ("茎", "椎名林檎", "English", 5.05, 4.47, "Electronic", ["https://www.youtube.com/watch?v=fVCCe2tuL20"]),
# ("73 Yips", "Aphex Twin", "English", 5.48, 5.05, "Electronic", ["https://www.youtube.com/watch?v=IaGz9WbZGos"]),
# ("My Girlfriend Insanity", "Helalyn Flowers", "English", 3.84, 4.82, "Electronic", ["https://www.youtube.com/watch?v=VD5lB119tfw"]),
# ("Mum-Man", "LFO", "English", 4.34, 4.82, "Electronic", ["https://www.youtube.com/watch?v=B6ZxdtVWjGA"]),
# ("Mummy, I've Had an Accident...", "LFO", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=HDfIA_7ihlc"]),
# ("Radio Waves II", "Van She", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=NWWhNtPmOqI"]),
# ("Share This Poison", "Razed in Black", "English", 5.22, 4.58, "Electronic", ["https://www.youtube.com/watch?v=ugfAQlkJIgQ"]),
# ("Illusion", "Ashbury Heights", "English", 4.26, 4.86, "Electronic", ["https://www.youtube.com/watch?v=dJHt2sSogB0"]),
# ("Blue Orb", "カヒミ・カリィ", "English", 5.42, 5.08, "Electronic", ["https://www.youtube.com/watch?v=ihsK6oOZ0kA"]),
# ("Lies", "Mankind Is Obsolete", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=e33hzT4Dh04"]),
# ("Lust End", "Prurient", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=hGpji5VF8k8"]),
# ("Rock and Roll", "Whitehouse", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=kuD4K60P16k"]),
# ("Artificial Intelligence", "Alien Produkt", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=J22XotZBLP8"]),
# ("Field of the Dead", "Detroit Diesel", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=-gnBJThTXhM"]),
# ("King of the Delta Blues", "Steven Wilson", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=lm_vCd0VyI0"]),
# ("War", "Acylum", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=P3iCsi7fzAk"]),
# ("Don't Feed The Robots (radio edit)", "Implant", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=c0koibr28iQ"]),
# ("My Mechatronics", "Psyborg Corp.", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=Eqq1BxJM9DA"]),
# ("Dead On Impact", "Dismantled", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=psMlf1IPc2o"]),
# ("Fireworks", "X-Wife", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=KAA8sm8LSJw"]),
# ("The Freakshow", "Project Rotten", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=E-9a6xMUC1Q"]),
# ("Dogday", "Genocide Organ", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=TcCsAtKass4"]),
# ("Pseudo Death Wish", "Third Realm", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=cCsfglZIF6Y"]),
# ("City of Straw", "Sightings", "English", 4.26, 4.86, "Electronic", ["https://www.youtube.com/watch?v=Mn0QG6YiE_Y"]),
# ("Omen reprise (denu remix)", "The Prodigy", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=Mh3X6ksCKew"]),
# ("Otho", "Alesia", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=8MWP-KsqEDE"]),
# ("Detect", "Erode", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=azsJnPp3KjQ"]),
# ("Red Water (Fallout Edit)", "Extize", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=cmsVD6uW5T0"]),
# ("The Velvet Machina", "Psyborg Corp.", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=RTM_ekul0pY"]),
# ("Drugs vs. Violence", "Implant", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=97kkyjOQiK4"]),
# ("We Are The Night", "Extize", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=EtGpeJ9Mr8U"]),
# ("Ordain And Establish", "Opir", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=yWgKJzOnl3M"]),
# ("Lost", "Elyzium", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=CBZ-STZ7AL0"]),
# ("Earnestly Pursued Oblivion", "Opir", "English", 5.91, 5.97, "Electronic", ["https://www.youtube.com/watch?v=3LjMAQrpOC8"]),
# ("When I Know You're Gone", "Blind Faith and Envy", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=SWFzrRQZflg"]),
# ("Mecha Tremors (Radio Edit)", "Animassacre", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=KeCUPj5sKl8"]),
# ("A Desperate Cry For Help", "Battery Cage", "English", 3.26, 5.75, "Electronic", ["https://www.youtube.com/watch?v=ApsNKUlqTj4"]),
# ("The Washington Consensus", "Opir", "English", 2.98, 5.92, "Electronic", ["https://www.youtube.com/watch?v=RKkn1IndLhI"]),
# ("Stahlwerk", "Volt", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=lCe_tBuTkx4"]),
# ("スペクトル", "ルルティア", "English", 4.8, 4.87, "Electronic", ["https://www.youtube.com/watch?v=mFY3dNt4OXU"]),
# ("Syndrome", "Mariel Ito", "English", 4.26, 4.86, "Electronic", ["https://www.youtube.com/watch?v=D98PrBm8lhU"]),
# ("XRUZT", "Return to Mono", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=2TjVg-CDUzs"]),
# ("Morrígan", "Tran Qual", "English", 5.17, 3.88, "Electronic", ["https://www.youtube.com/watch?v=Le-BWQ-h48Q"]),
# ("closed circuit loop", "Phuq", "English", 3.5, 3.91, "Electronic", [""]),
# ("Nightfall", "Return to Mono", "English", 5.44, 4.68, "Electronic", ["https://www.youtube.com/watch?v=K3sAi3XEn5M"]),
# ("Finally", "Mariel Ito", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=MmTMdkmUYCI"]),
# ("White and Black", "The Rorschach Audio", "English", 5.72, 5.5, "Electronic", ["https://www.youtube.com/watch?v=DcLcnf25DAA"]),
# ("CAME FROM", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=r96L8-C4JjE"]),
# ("XTRA", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=r96L8-C4JjE"]),
# ("POOR FATE", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=cRwXr92ds6U"]),
# ("00 TYPE", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=-LWrNXMNeqg"]),
# ("WORST STRIKE", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=cRwXr92ds6U"]),
# ("VORTEX", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=Y5VWa0b-XTQ"]),
# ("ORBITS", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=aabWdBvdk7w"]),
# ("DEALER", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=wz_FdVaoecc"]),
# ("SQUAD", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=FNoBLlDdmjU"]),
# ("Prana", "High Level Static", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=1VJ8LcxKSKE"]),
# ("SNOWBLIND", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=DECxluN8OZM"]),
# ("SPIRICOM", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=3bHcYNW0azA"]),
# ("D0PX", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=3bHcYNW0azA"]),
# ("BODY", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=EADc95uIe58"]),
# ("ECRIT ELON", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=3bHcYNW0azA"]),
# ("ventrex", "Phuq", "English", 4.09, 4.11, "Electronic", ["https://www.youtube.com/watch?v=jjtc2tIrTes"]),
# ("EOLS (ALT TAKE)", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=nDAf0wPwpGI"]),
# ("Bound", "Raivyn", "English", 4.26, 4.86, "Electronic", ["https://www.youtube.com/watch?v=NGl3JraCYHI"]),
# ("FAIL SAFE (VERSION) / WORK", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=gu1U40TR-_M"]),
# ("The Red Fog", "Gears of Destruction", "English", 3.48, 4.61, "Electronic", ["https://www.youtube.com/watch?v=Eip_z0KHIAY"]),
# ("A Lifetime of Pain", "Gears of Destruction", "English", 3.4, 4.76, "Electronic", ["https://www.youtube.com/watch?v=QUn3OKzFgg0"]),
# ("elements", "My Boyfriend the Pilot", "English", 5.02, 5.25, "Electronic", ["https://www.youtube.com/watch?v=G9CvAXb3xOY"]),
# ("FIRE HAZARD", "Five Star Hotel x PIVOTAL", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=QKbeF9gMvHA"]),
# ("Child's Party", "The Demo Saou", "English", 6.48, 5.45, "Electronic", ["https://www.youtube.com/watch?v=WyhCZ6dSFew"]),
# ("Marschmusik", "LSF", "English", 4.26, 4.86, "Electronic", ["https://www.youtube.com/watch?v=tLwamOWkUns"]),
# ("Alfred's Bad Trip", "The Demo Saou", "English", 1.72, 2.81, "Electronic", [""]),
# ("DJ Zick - Straitjackets (Bondage Edit)", "DJ Zick Finland", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=231l5J7v7-A"]),
# ("Diagnosis Without Consequences", "The Demonic Paradise", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=hDroA6eawsE"]),
# ("ready 4 war", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("settings eternal", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("dysphoria", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=ZsU75duhJCw"]),
# ("overworld theme", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("NeuroTransmitter (bad ending)", "Hamster Alliance", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=WUDPxgTDkaA"]),
# ("shrug/fate 2", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("Plus3", "LSF", "English", 3.71, 5.5, "Electronic", ["https://www.youtube.com/watch?v=pyf8cbqyfPs"]),
# ("Prophetic Dream", "Prophetic Dream", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=Ly029ZPg9Iw"]),
# ("department of defense", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("The Poorhouse", "DJ Citalopram", "English", 3.15, 5.94, "Electronic", [""]),
# ("Making Backroom Baby", "The Demo Saou", "English", 3.44, 5.63, "Electronic", [""]),
# ("antidote", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=HnNmpC2gRNk"]),
# ("mj", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=lrKZNqIR2U0"]),
# ("workxxx", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("weather patterns", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=vIUhUsKePW4"]),
# ("stadium trax", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("in the open feat. Dylan Brady", "Five Star Hotel", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=e2l0HlLefaw"]),
# ("Wild West Wedgie", "The Demo Saou", "English", 3.44, 5.63, "Electronic", [""]),
# ("standing in the red rain", "Five Star Hotel x Smith Comma John", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=GlSkiiYsqMQ"]),
# ("The Red Fog(Demo)", "Gears of Destruction", "English", 4.32, 4.86, "Electronic", ["https://www.youtube.com/watch?v=LLE5_zkIbzA"]),
# ("WARM ISCHEMIA/PERSONAL APOCALYPSE (edit)", "Navicon Torture Technologies", "English", 3.44, 5.63, "Electronic", [""]),
# ("Nonsense #6", "Sala della Pallacorda", "English", 3.44, 5.63, "Electronic", ["https://www.youtube.com/watch?v=4ossH4OmpfU"]),
# ("Juwpilir Balls Song", "The Demo Saou", "English", 3.44, 5.63, "Electronic", [""]),
# ("Water Test", "Sala della Pallacorda", "English", 3.44, 5.63, "Electronic", [""]),
# ("DJ Zick - ☣[X]Irm[X]sc[X]her[X]☣ (Killer Cut)", "DJ Zick Finland", "English", 3.44, 5.63, "Electronic", [""]),
# ("Gr8 Mate", "0A:8F", "English", 3.74, 4.55, "Electronic", [""]),
# ("Demons in My Head", "Gears of Destruction", "English", 3.78, 4.78, "Electronic", ["https://www.youtube.com/watch?v=xD9eHSWQoGQ"]),
# ("Phantom Limb", "Fuck Buttons", "English", 6.26, 6.16, "Electronic", ["https://www.youtube.com/watch?v=ewID5S1o92Q"]),
# ("The Step", "!!!", "English", 5.26, 3.95, "Electronic", ["https://www.youtube.com/watch?v=T1ur2jQtPpk"]),
# ("Supremacy II", "Polygon Window", "English", 4.9, 4.46, "Electronic", ["https://www.youtube.com/watch?v=J77T8Tus9UQ"]),
# ("Terror Flynn Ok Ok", "Slagsmålsklubben", "English", 5.37, 4.34, "Electronic", ["https://www.youtube.com/watch?v=Ua1fnMy1O5U"]),
# ("full throttle", "Prodigy", "English", 5.58, 4.82, "Electronic", ["https://www.youtube.com/watch?v=sxWziWWeZdk"]),
# ("Space Island", "!!!", "English", 6.44, 4.83, "Electronic", ["https://www.youtube.com/watch?v=8mSRcXoCYTI"]),
# ("Morning", "Hyper", "English", 6.34, 5.81, "Electronic", ["https://www.youtube.com/watch?v=xDAgRSDAD1A"]),
# ("Claustrophobic sting", "Prodigy", "English", 5.09, 4.8, "Electronic", ["https://www.youtube.com/watch?v=otvGhjzsA5s"]),
# ("Higher State of Consciousness", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Everybody in the Place", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=WY87o9IZXWg"]),
# ("Speed Freak (Moby remix)", "Orbital", "English", 4.5, 4.75, "Electronic", ["https://www.youtube.com/watch?v=1DlQNT2FXDM"]),
# ("Your Love", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=zShqIyH18Vg"]),
# ("Charly", "Prodigy", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=cSTBFZ-To2E"]),
# ("Fire", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=F1U0qvtQnE8"]),
# ("Resistor", "Yello", "English", 4.86, 5.04, "Electronic", ["https://www.youtube.com/watch?v=Sn3rvb0dkQE"]),
# ("Rock The Casbah", "Solar Twins", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=-E5kglUmi2A"]),
# ("Charly (Original Mix)", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=cSTBFZ-To2E"]),
# ("I See Fire", "Hyper", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=AmSR05rA8qE"]),
# ("Ride", "Chable & Bonnici", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=9-vzhZz9pZA"]),
# ("Music Reach", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=SVFEQ_lck44"]),
# ("Chip", "Doshy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=-23cQK_MdPo"]),
# ("Contact", "Eat Static", "English", 5.18, 4.62, "Electronic", ["https://www.youtube.com/watch?v=4RPSX5t2rpk"]),
# ("Hyperspeed", "Prodigy", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=87Cn6iX4rYU"]),
# ("Feeling So Real (Unashamed Ecstatic Piano mix)", "Moby", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=8_TO9jFSgS4"]),
# ("Ridiculosous", "Electrocution 250", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=nGRpahTo5xw"]),
# ("C9:: Cadenza", "Ryoji Ikeda", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=OdpwR4TfJ88"]),
# ("London Acid City", "Lochi", "English", 5.17, 4.76, "Electronic", ["https://www.youtube.com/watch?v=B2iGM01Mgx0"]),
# ("Dominica", "F2", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=t0gT-lQrCC4"]),
# ("Sim the Builder (Hyper Remix)", "Mark Mothersbaugh", "English", 6.74, 5.92, "Electronic", ["https://www.youtube.com/watch?v=On1yoI1tOBQ"]),
# ("Neurodancer", "Wippenberg", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=t6EbjnPKico"]),
# ("Confusion (Pump Panel Reconstruction)", "New Order", "English", 5.18, 4.62, "Electronic", ["https://www.youtube.com/watch?v=c_L_-CKg6pw"]),
# ("Ready to Flow", "Urban Trance Plant", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=s7CgJuNoFOo"]),
# ("The Heat [The Energy]", "Prodigy", "English", 5.09, 4.8, "Electronic", ["https://www.youtube.com/watch?v=_NksGaZJQRI"]),
# ("Higher State of Consciousness (Dex & Jonesey's Higher Stated Mix)", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Osaka Acid", "Sushi", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=T5cC4yvapFE"]),
# ("12.01", "Lemon D", "English", 5.3, 4.57, "Electronic", ["https://www.youtube.com/watch?v=IwMWBrxhWd0"]),
# ("Dance Wiv Me (Agent X Remix)", "Dizzee Rascal", "English", 6.74, 5.92, "Electronic", ["https://www.youtube.com/watch?v=vLkcV2necbk"]),
# ("Fade 2 Black", "Goldie", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=qNRfSnFoWtg"]),
# ("Glass Ball", "Rebel Yelle", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=qFGC2D5BnJs"]),
# ("Firefight", "Cosmic Trigger", "English", 5.18, 4.62, "Electronic", ["https://www.youtube.com/watch?v=5DFrUyTwB2E"]),
# ("Nothing Can Save Us London", "Star Power", "English", 5.18, 4.62, "Electronic", ["https://www.youtube.com/watch?v=uV4Ma9qp4r8"]),
# ("Thunderflash", "Weathermen", "English", 5.94, 4.36, "Electronic", ["https://www.youtube.com/watch?v=ulEE20xHfE0"]),
# ("Urban & Free", "Dynamo City", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=J6UMBD3_gaA"]),
# ("Silver Surfer", "Time Stretch Armstrong", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=gT0rfiREdqI"]),
# ("Time Bomb", "Overrider", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=5jVZZMt3k7s"]),
# ("Acid War", "D.O.M.", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=PgzmydKU9Go"]),
# ("Higher State of Consciousness [Original Tweekin' Acid Funk Mix]", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Burn Out", "Last Of The Hunted Sonz Of Freestyle", "English", 5.42, 4.8, "Electronic", ["https://www.youtube.com/watch?v=DWLyJslZgoA"]),
# ("Manadeva (1996 remix)", "Astral Projection", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=onquKz5SYRs"]),
# ("CX4", "Apraxia", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=a9_r4fEcCR8"]),
# ("Chuggernaut", "Smog Blanket", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=u6hWYnprQjY"]),
# ("Metallican", "Vytear", "English", 4.65, 5.04, "Electronic", ["https://www.youtube.com/watch?v=XPUcplhGz7U"]),
# ("Quazar", "Kektex", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=6UmfZhGVBgQ"]),
# ("Hustlers Revenge", "6400 Crew Present Joeski", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=VzpxRWkrHuE"]),
# ("Pressure Cell", "Weathermen", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=G8sxdZgFU2s"]),
# ("Konami Code III", "The Gothsicles", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=iE1DLwOabsQ"]),
# ("Timebomb", "Overrider", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=DhgXM7e_D9w"]),
# ("Comet (bright take one)", "Duplex-Ache", "English", 5.76, 5.88, "Electronic", ["https://www.youtube.com/watch?v=Aguu1RdGYs4"]),
# ("Crispy Bacon (Live)", "Laurent Garnier", "English", 5.18, 4.62, "Electronic", ["https://www.youtube.com/watch?v=t3NlTpHiFwg"]),
# ("FSE NEWB 418 (Rough Unfinished Mix)", "The Flaming Schwarzkopf Experience", "English", 5.11, 5.52, "Electronic", [""]),
# ("Boom Boom Fire", "Dr. Love", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=16N8Zhhhins"]),
# ("Squid", "Intellect 3000", "English", 5.73, 4.23, "Electronic", ["https://www.youtube.com/watch?v=F9bEHHoArO8"]),
# ("Higher State Of Consciousness (The Deep & Slow Mix)", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Higher State Of Consciousness (Hardhouse Mix)", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Wa (Full)", "이정현", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=XBuTBwh1OKE"]),
# ("Free Lemonade Remix", "Total Eclipse", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=wHQvJ3H15L0"]),
# ("Shudder - King of Snake", "Underworld", "English", 5.34, 4.86, "Electronic", ["https://www.youtube.com/watch?v=z_j3fTedFA0"]),
# ("Yeke Yeke (Hard Floor Mix)", "Mory Kante", "English", 5.2, 4.33, "Electronic", ["https://www.youtube.com/watch?v=YDZ9HYCAknc"]),
# ("Higher State Of Consciousness (Original Tweekin Acid Mix)", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Infection", "Biostorm", "English", 5.09, 4.8, "Electronic", ["https://www.youtube.com/watch?v=rL8T9FUynZ8"]),
# ("Drop Some Drums", "[Love]tatoo", "English", 5.11, 5.52, "Electronic", ["https://www.youtube.com/watch?v=nbemueZrk9Y"]),
# ("Creator (vs. Switch and FreQ Nasty)", "Santogold", "English", 5.39, 5.73, "Electronic", ["https://www.youtube.com/watch?v=Hb25qkXiMMk"]),
# ("Higher State Of Consciousness (Jules & Skins Vocals by Sonique Long and Epic Mix)", "Josh Wink", "English", 6.79, 5.79, "Electronic", ["https://www.youtube.com/watch?v=7v33CUWtpkI"]),
# ("Bang On", "Propellerheads", "English", 4.64, 4.8, "Electronic", ["https://www.youtube.com/watch?v=6s6Xj3M8C8I"]),
# ("The Game Has Changed", "Daft Punk", "English", 6.31, 5.12, "Electronic", ["https://www.youtube.com/watch?v=c1Eeu0lsozc"]),
# ("Nostrand", "Ratatat", "English", 6.68, 4.27, "Electronic", ["https://www.youtube.com/watch?v=eJyKpsJbVdo"]),
# ("Canon", "Justice", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=JZ-0IEtHEac"]),
# ("Where Is the Line", "Björk", "English", 5.89, 4.86, "Electronic", ["https://www.youtube.com/watch?v=yx2xqUzCLhU"]),
# ("54 Cymru Beats", "Aphex Twin", "English", 4.9, 5.5, "Electronic", ["https://www.youtube.com/watch?v=_a83fSZR2Ag"]),
# ("Vertebræ by Vertebræ", "Björk", "English", 6.28, 5.16, "Electronic", ["https://www.youtube.com/watch?v=YelGD7HAhOk"]),
# ("Cop Killer", "John Maus", "English", 4.64, 5.1, "Electronic", ["https://www.youtube.com/watch?v=GgxVktBNZjY"]),
# ("Exotic Talk", "RJD2", "English", 6.06, 5.1, "Electronic", ["https://www.youtube.com/watch?v=y_nry8s365o"]),
# ("*", "M83", "English", 5.82, 4.91, "Electronic", ["https://www.youtube.com/watch?v=lAwYodrBr2Q"]),
# ("An Echo A Stain", "Björk", "English", 4.22, 3.84, "Electronic", ["https://www.youtube.com/watch?v=Xgslrh-efv8"]),
# ("Walking Through That Door", "Future Islands", "English", 6.58, 5.75, "Electronic", ["https://www.youtube.com/watch?v=5qHwbHor_p8"]),
# ("Family", "Björk", "English", 4.6, 3.74, "Electronic", ["https://www.youtube.com/watch?v=zyCyEzejNQI"]),
# ("The Great Fire", "Future Islands", "English", 5.85, 6.03, "Electronic", ["https://www.youtube.com/watch?v=GLFRPStz8Co"]),
# ("Phone Call", "The Faint", "English", 6.54, 6.7, "Electronic", ["https://www.youtube.com/watch?v=M2wzr-jCckI"]),
# ("Kobwebz", "Gonjasufi", "English", 3.81, 4.35, "Electronic", ["https://www.youtube.com/watch?v=KHuw9QsRjaU"]),
# ("The Stamen Of The Shamen", "Shpongle", "English", 5.92, 4.95, "Electronic", ["https://www.youtube.com/watch?v=XPOAzOwkOjo"]),
# ("Car Chase Terror!", "M83", "English", 4.75, 5.31, "Electronic", ["https://www.youtube.com/watch?v=ALBJsKpt7Kg"]),
# ("Is This Power", "The Field", "English", 4.32, 3.52, "Electronic", ["https://www.youtube.com/watch?v=YVNqWZPqxDE"]),
# ("Rykketid", "Trentemøller", "English", 5.59, 5.14, "Electronic", ["https://www.youtube.com/watch?v=RvFQ-5OI_90"]),
# ("An Apology", "Future Islands", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=r5gSCCVVsCM"]),
# ("Hands On Us", "The Notwist", "English", 1.12, 1.02, "Electronic", ["https://www.youtube.com/watch?v=0x3AurxI2YQ"]),
# ("Into The Trees (Serenetti Part 3)", "Trentemøller", "English", 4.47, 5.3, "Electronic", ["https://www.youtube.com/watch?v=zVBrDID1mY4"]),
# ("Physical Fraction", "Trentemøller", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=Sd65verb2tE"]),
# ("Sacrifice", "Clint Mansell", "English", 6.48, 5.12, "Electronic", ["https://www.youtube.com/watch?v=T5rIfF0qCeU"]),
# ("Punch up at a wedding", "Radiohead", "English", 6.68, 6.33, "Electronic", ["https://www.youtube.com/watch?v=2DWP6a25iJI"]),
# ("Encom Part II", "Daft Punk", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=WeENy4_7jDE"]),
# ("Slow with Horns / Run for Your Life", "Dan Deacon", "English", 5.23, 5.61, "Electronic", ["https://www.youtube.com/watch?v=mNRD-ALMMVI"]),
# ("Jimmy Joe Roche", "Dan Deacon", "English", 3.63, 3.44, "Electronic", ["https://www.youtube.com/watch?v=5L9Jx9JKRWg"]),
# ("Get Older", "Dan Deacon", "English", 3.51, 4.5, "Electronic", ["https://www.youtube.com/watch?v=OPddleVa6yg"]),
# ("Ghosts 'n Stuff (Featuring Rob Swire)", "deadmau5", "English", 6.24, 5.01, "Electronic", ["https://www.youtube.com/watch?v=6rHngwaJKUE"]),
# ("Bass Nipple", "Infected Mushroom", "English", 6.42, 5.7, "Electronic", ["https://www.youtube.com/watch?v=tKeP0OLpYss"]),
# ("You Will Be Sad", "Washed Out", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=UCUorLXccdE"]),
# ("This Empty Love", "Innerpartysystem", "English", 5.53, 5.55, "Electronic", ["https://www.youtube.com/watch?v=uKPd8o87E2o"]),
# ("Summer's Gone", "Pretty Lights", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=agMmGoJW5bQ"]),
# ("The Famous Biting Guy", "Xploding Plastix", "English", 7.32, 5.95, "Electronic", ["https://www.youtube.com/watch?v=i3WFziKOLy8"]),
# ("Daybreak", "OVERWERK", "English", 6.8, 5.09, "Electronic", ["https://www.youtube.com/watch?v=JnIKLRA31nw"]),
# ("Can't Stop Looking", "Cash Cash", "English", 6.7, 5.79, "Electronic", ["https://www.youtube.com/watch?v=tUi4fcgLW9Q"]),
# ("Pandemonium", "The Prodigy", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=1JQ6-E43aiE"]),
# ("Goodbye, 2007", "65daysofstatic", "English", 5.5, 5.84, "Electronic", ["https://www.youtube.com/watch?v=KMdS48qIl0I"]),
# ("Ghost Town", "DJ Shadow", "English", 5.41, 4.92, "Electronic", ["https://www.youtube.com/watch?v=FJGBBlSQy2w"]),
# ("White Trash", "Junior Senior", "English", 7.11, 5.71, "Electronic", ["https://www.youtube.com/watch?v=Pu2Wt_yjcGo"]),
# ("Crumax Rins", "Plaid", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=d5WXW3wqzI8"]),
# ("Distant (Rubicon II)", "VNV Nation", "English", 3.63, 2.87, "Electronic", ["https://www.youtube.com/watch?v=zY_65GUB0ZU"]),
# ("Pariterapiaa", "PMMP", "English", 5.52, 5.64, "Electronic", ["https://www.youtube.com/watch?v=w26aDHbFgJg"]),
# ("Empire", "Hybrid", "English", 5.77, 5.03, "Electronic", ["https://www.youtube.com/watch?v=OW7RgjjKxOw"]),
# ("See Me Now", "Infected Mushroom", "English", 6.42, 5.7, "Electronic", ["https://www.youtube.com/watch?v=siJPhASkXi4"]),
# ("In the Morning", "Bang Gang", "English", 6.2, 5.85, "Electronic", ["https://www.youtube.com/watch?v=vJ5FNqqWzGI"]),
# ("Youth Alcoholic", "Fox n' Wolf", "English", 7.01, 6.46, "Electronic", ["https://www.youtube.com/watch?v=j0071mjNOpI"]),
# ("In 4 The Kill Pon De Skream", "Major Lazer & La Roux", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=jEsTgEscs-4"]),
# ("The Pills Won't Help You Know (feat. Midlake)", "The Chemical Brothers", "English", 4.48, 3.31, "Electronic", ["https://www.youtube.com/watch?v=lNHRnhOP3hE"]),
# ("East", "Blue Stahli", "English", 6.42, 5.7, "Electronic", ["https://www.youtube.com/watch?v=YY_Y60TmDDQ"]),
# ("Red Hot Drops", "Chad VanGaalen", "English", 6.69, 4.32, "Electronic", ["https://www.youtube.com/watch?v=O5CwHXPp4rU"]),
# ("Liege", "Para One", "English", 5.45, 5.17, "Electronic", ["https://www.youtube.com/watch?v=q8QVJUTn9tM"]),
# ("Traffickers", "The Reflecting Skin", "English", 4.23, 4.74, "Electronic", ["https://www.youtube.com/watch?v=EDT9vygEqo0"]),
# ("Ripping Out Tears", "Nitin Sawhney", "English", 5.37, 5.34, "Electronic", ["https://www.youtube.com/watch?v=B9lYKOjKzAc"]),
# ("orbit", "She", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=SHFmUt1Jit4"]),
# ("Wild in Blue", "Suicide", "English", 5.38, 5.15, "Electronic", ["https://www.youtube.com/watch?v=LLlvFBw157o"]),
# ("Tel Aviv's Gravel Toothbrush", "Prefuse 73", "English", 5.65, 6.6, "Electronic", ["https://www.youtube.com/watch?v=kYOnqLBKK2E"]),
# ("Mt. Saint Michel Mix+St. Michaels Mount", "Aphex Twin", "English", 4.98, 4.96, "Electronic", ["https://www.youtube.com/watch?v=6kN4zRfnHm8"]),
# ("Marrakech", "Hybrid", "English", 5.55, 4.48, "Electronic", ["https://www.youtube.com/watch?v=tfoLzYJkr0I"]),
# ("Only This Moment (Alan Braxe And Fred Falke Remix)", "Röyksopp", "English", 6.76, 5.19, "Electronic", ["https://www.youtube.com/watch?v=LEIlsvlbTM4"]),
# ("Gone Darker", "Electrelane", "English", 7.12, 6.0, "Electronic", ["https://www.youtube.com/watch?v=XcwNyUdUq3I"]),
# ("Ethiobirds", "Andrew Bird", "English", 5.89, 3.69, "Electronic", ["https://www.youtube.com/watch?v=X68GuutXJuY"]),
# ("Boomin'", "TobyMac", "English", 7.64, 6.19, "Electronic", ["https://www.youtube.com/watch?v=wTqNj8YnQV4"]),
# ("2010", "Cornelius", "English", 6.85, 5.41, "Electronic", ["https://www.youtube.com/watch?v=mgEXQG5SuWc"]),
# ("Magoo Opening", "Cornelius", "English", 6.43, 4.93, "Electronic", ["https://www.youtube.com/watch?v=Eqh458PKV_E"]),
# ("Distroia", "Mouse on Mars", "English", 5.5, 3.99, "Electronic", ["https://www.youtube.com/watch?v=oKYNxQ1G4WA"]),
# ("Für Immer 16", "Stereo Total", "English", 5.8, 5.67, "Electronic", ["https://www.youtube.com/watch?v=pBO7xMVHbBw"]),
# ("Going to Georgia", "Atom and His Package", "English", 4.22, 3.53, "Electronic", ["https://www.youtube.com/watch?v=_YDKtNaL8Dk"]),
# ("17 years", "Ratatat", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=3NFqqUoh3BI"]),
# ("A Time To Fear (Who's Afraid)", "Art of Noise", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=SpsWLtn1OrQ"]),
# ("Auto-Lite: Spark Plugs", "Raymond Scott", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=nnoOrgOQI3k"]),
# ("Loco", "Loving Paris", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=q8tIeTdh1tU"]),
# ("Flight Of The Bumblebee", "Jean-Jacques Perrey", "English", 5.63, 3.97, "Electronic", ["https://www.youtube.com/watch?v=T9JiYl6lrhs"]),
# ("Slash Dot Dash (London Less Fat edit)", "Fatboy Slim", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=63f35w3j3IE"]),
# ("Baptism (KTF Megatron Remix)", "Crystal Castles", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=iiEIamatkBE"]),
# ("Manic, Impressive!", "Shortwave Dahlia", "English", 6.33, 5.3, "Electronic", ["https://www.youtube.com/watch?v=zZCix6MO8P8"]),
# ("Key Lime (live)", "Germlin", "English", 5.96, 5.3, "Electronic", ["https://www.youtube.com/watch?v=5U9rMkgoGL8"]),
# ("Sky High", "Nick Hopkin", "English", 4.88, 5.04, "Electronic", ["https://www.youtube.com/watch?v=SoZ1e5kjjcs"]),
# ("Series 20", "Nick Hopkin", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=6ethollg-PI"]),
# ("concrete bleed", "ameobatube", "English", 4.15, 4.67, "Electronic", [""]),
# ("Boris Belayav", "Mezzanine Stairs", "English", 4.19, 4.55, "Electronic", [""]),
# ("Loop One", "Martin Dolgener", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=iWu9FPO2LHw"]),
# ("Tastes Like Chicken", "Nick Hopkin", "English", 5.41, 4.55, "Electronic", ["https://www.youtube.com/watch?v=fx75wNIzggs"]),
# ("Surf Bitches", "Blood Ninjas", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=YCdL2VBhEIw"]),
# ("Gumby hat", "T.H.N.", "English", 4.19, 4.55, "Electronic", ["https://www.youtube.com/watch?v=4hqGq-Kr9Hs"]),
# ("My Pants On Fire", "Easy Bake Omen", "English", 6.28, 5.43, "Electronic", ["https://www.youtube.com/watch?v=tscMSXk_jaQ"]),
# ("anger-E", "bINARY üBER-gLOTS", "English", 3.35, 5.24, "Electronic", ["https://www.youtube.com/watch?v=Wecb4d8F33g"]),
# ("Sleepy Dinosaur", "Flying Lotus", "English", 3.14, 4.76, "Electronic", ["https://www.youtube.com/watch?v=Wp71DxpEclk"]),
# ("Riot", "Flying Lotus", "English", 3.37, 4.86, "Electronic", ["https://www.youtube.com/watch?v=yXULecvImpw"]),
# ("Seven Stars", "Air", "English", 5.51, 3.18, "Electronic", ["https://www.youtube.com/watch?v=HkxiEaJ_M_I"]),
# ("Worlds Converged", "Midnight Juggernauts", "English", 4.46, 4.03, "Electronic", ["https://www.youtube.com/watch?v=47TdlIGpUMw"]),
# ("Stem / Long Stem / Transmission 2", "DJ Shadow", "English", 4.77, 3.7, "Electronic", ["https://www.youtube.com/watch?v=M1g-pF2ZezM"]),
# ("Stress", "Thomas Bangalter", "English", 4.21, 4.95, "Electronic", ["https://www.youtube.com/watch?v=ePbz_h3Wn0w"]),
# ("Come To Daddy, Mummy Mix", "Aphex Twin", "English", 4.85, 5.06, "Electronic", ["https://www.youtube.com/watch?v=rSorN9vhLz4"]),
# ("Mental Universe", "Free the Robots", "English", 3.14, 4.76, "Electronic", ["https://www.youtube.com/watch?v=Y_lMeTibExI"]),
# ("Flash Ram", "Brainiac", "English", 3.54, 5.17, "Electronic", ["https://www.youtube.com/watch?v=6qGzZDrSTro"]),
# ("The Symbiote", "Eprom", "English", 3.14, 4.76, "Electronic", ["https://www.youtube.com/watch?v=qoFGcx8WT5E"]),
# ("Lector's Cell", "The Reds", "English", 4.62, 4.92, "Electronic", ["https://www.youtube.com/watch?v=b-7zYS_u1ks"]),
# ("Interlude I", "Covenant", "English", 4.65, 4.31, "Electronic", ["https://www.youtube.com/watch?v=MiGIR174768"]),
# ("Leed's House", "The Reds", "English", 5.07, 4.62, "Electronic", ["https://www.youtube.com/watch?v=OPESM7ng920"]),
# ("Dum Dum Dice", "The Reds", "English", 4.9, 4.88, "Electronic", ["https://www.youtube.com/watch?v=ERq2njLvQhY"]),
# ("End of Line (feat. Search & Destroy)", "Vex'd", "English", 4.11, 4.42, "Electronic", ["https://www.youtube.com/watch?v=1FMNcP8dnBE"]),
# ("Can't Bring You Back", "The Reds", "English", 4.52, 4.07, "Electronic", ["https://www.youtube.com/watch?v=L1IG5I5yCxE"]),
# ("Till the End", "The Reds", "English", 4.53, 5.08, "Electronic", ["https://www.youtube.com/watch?v=TbuKbgIX1mc"]),
# ("Lost Forever", "The Reds", "English", 5.17, 4.72, "Electronic", ["https://www.youtube.com/watch?v=8U2AIvnyHvU"]),
# ("Ecuador", "Buz", "English", 2.18, 3.44, "Electronic", ["https://www.youtube.com/watch?v=KqaGdcQh5jA"]),
# ("Crazy Disco (Bastille Remix)", "Bastille", "English", 3.14, 4.76, "Electronic", ["https://www.youtube.com/watch?v=xoUVtthd7gY"]),
# ("Lookout", "The Reds", "English", 4.53, 5.08, "Electronic", ["https://www.youtube.com/watch?v=2rBhVDqPQF0"]),
# ("Strangeness", "The Reds", "English", 4.24, 5.33, "Electronic", ["https://www.youtube.com/watch?v=mz_kdE7JwQo"]),
# ("Shaken Cold", "The Reds", "English", 4.9, 4.88, "Electronic", ["https://www.youtube.com/watch?v=libVrj7OIrA"]),
# ("Beat Away", "The Reds", "English", 4.53, 5.08, "Electronic", ["https://www.youtube.com/watch?v=OPfs75hg_t8"]),
# ("Try It On You", "The Reds", "English", 4.24, 5.33, "Electronic", ["https://www.youtube.com/watch?v=zwcmZ0mGQno"]),
# ("Survival", "Somatic Responses", "English", 4.11, 4.42, "Electronic", ["https://www.youtube.com/watch?v=sBjEktnTAu0"]),
# ("Mister Z", "The Reds", "English", 4.24, 5.33, "Electronic", ["https://www.youtube.com/watch?v=X91RpUDAJZs"]),
# ("Three Forty Seven", "Bruce Cohen", "English", 4.24, 5.33, "Electronic", ["https://www.youtube.com/watch?v=Vka-XwxP9Q4"]),
# ("Alexandria by Night", "Tim Doyle", "English", 4.43, 4.42, "Electronic", ["https://www.youtube.com/watch?v=PCMzW1WjmqE"]),
# ("The Story Of Cruub Pilot Hunter And Lt. Annifer Dubay", "The Psychedelic Avengers", "English", 3.14, 4.76, "Electronic", [""]),
# ("Supremacy Bleeds (Counterstrike Duomix)", "Silent Killer", "English", 3.14, 4.76, "Electronic", ["https://www.youtube.com/watch?v=jXKEShZfJXQ"]),
# ("Empty", "t r y ^ d", "English", 4.86, 3.11, "Electronic", ["https://www.youtube.com/watch?v=PsKg8dEWWco"]),
# ("The Highest Flood", "Forest Swords", "English", 5.57, 5.18, "Electronic", ["https://www.youtube.com/watch?v=bnh7ilPUSGo"]),
# ("Ready to Lose", "The Knife", "English", 3.67, 5.57, "Electronic", ["https://www.youtube.com/watch?v=kEicst24c3A"]),
# ("Kitty Cat", "Amon Tobin", "English", 5.75, 5.27, "Electronic", ["https://www.youtube.com/watch?v=Vt8OEVS9OJQ"]),
# ("Changeling / Transmission 1", "DJ Shadow", "English", 5.33, 4.12, "Electronic", ["https://www.youtube.com/watch?v=p--tqVejRvU"]),
# ("If It Really Is Me", "Polygon Window", "English", 4.65, 4.28, "Electronic", ["https://www.youtube.com/watch?v=YRqcPPKSjhQ"]),
# ("Rino's Prayer", "Leftfield", "English", 5.6, 4.77, "Electronic", ["https://www.youtube.com/watch?v=hNA78IYKXjI"]),
# ("Rain of Brass Petals", "Akira Yamaoka", "English", 2.02, 3.02, "Electronic", ["https://www.youtube.com/watch?v=nagAombKIcE"]),
# ("Star Shpongled Banner (Brothomstates remix)", "Shpongle", "English", 4.63, 3.7, "Electronic", ["https://www.youtube.com/watch?v=aHrtDStSwxs"]),
# ("Dis (Reverberation)", "Hecq", "English", 1.98, 2.79, "Electronic", ["https://www.youtube.com/watch?v=z6j6nqV7u2w"]),
# ("Flight Of The Albatross", "Fairmont", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=aX9Mrs4e90Y"]),
# ("Berserker", "Gary Numan", "English", 3.01, 3.22, "Electronic", ["https://www.youtube.com/watch?v=PzMO0EOo8Rc"]),
# ("Terre Zippy", "Speedy J", "English", 4.83, 4.83, "Electronic", ["https://www.youtube.com/watch?v=_wNgM9NTSwo"]),
# ("Theme from Earthshaker", "Cabaret Voltaire", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=RtaPaBp7lqU"]),
# ("Behind The Water Tower", "Exhaust", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=igefw4zK1Ok"]),
# ("3", "Hecq", "English", 4.91, 4.12, "Electronic", ["https://www.youtube.com/watch?v=r-RiSVCPT0s"]),
# ("Pain Game", "Fonik", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=xorOPrOk6j8"]),
# ("7", "Hecq", "English", 3.62, 4.79, "Electronic", ["https://www.youtube.com/watch?v=fgWYfUiZggI"]),
# ("Double Vision", "Cabaret Voltaire", "English", 4.17, 3.89, "Electronic", ["https://www.youtube.com/watch?v=A74ISkG4fos"]),
# ("Mudo e Surdo", "Paus", "English", 3.29, 3.42, "Electronic", ["https://www.youtube.com/watch?v=eDKXeHv6qkI"]),
# ("Atoll", "Lustmord", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=X-omfuBabf8"]),
# ("Trinity", "Lustmord", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=3fPukacg6Jg"]),
# ("17", "Hecq", "English", 3.67, 5.57, "Electronic", ["https://www.youtube.com/watch?v=qSho64MvT0Y"]),
# ("Gjallarhorn", "The Panacea", "English", 3.67, 5.57, "Electronic", ["https://www.youtube.com/watch?v=oph0Ku_Kai4"]),
# ("L'Hôtel", "Yello", "English", 2.45, 3.71, "Electronic", ["https://www.youtube.com/watch?v=RWGUq64-Jtc"]),
# ("As the Bubble Expands", "Speedy J", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=xrmyAp_BnM0"]),
# ("Nicolas Ritter", "Matthew Herbert", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=09Wtl2UIjNI"]),
# ("White Irises Blind (Minimal Mix)", "Scorn", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=oUwc1McPixc"]),
# ("On Ice (Disembodied in Dub)", "Scorn", "English", 4.83, 4.83, "Electronic", ["https://www.youtube.com/watch?v=IJFoKIxKBRc"]),
# ("Back & Spine (Fidel Astro Remix)", "Kasper Bjørke", "English", 4.26, 4.44, "Electronic", ["https://www.youtube.com/watch?v=NtJiT0rta80"]),
# ("April", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=zyRFufOmeFY"]),
# ("Sing Me a Lullaby", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=8ej2YOBsimQ"]),
# ("survive (live mix)", "Somatic Responses", "English", 4.51, 4.83, "Electronic", ["https://www.youtube.com/watch?v=uWe4QZ2-Yfc"]),
# ("Fort Gnox", "Zan-zan-zawa-veia", "English", 4.88, 4.46, "Electronic", ["https://www.youtube.com/watch?v=p5cyDPUl_kg"]),
# ("Never Doin' That Again", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=z8ir6d2JZMI"]),
# ("Guardians (part 2)", "Alchemorph Soundtracks", "English", 4.01, 5.0, "Electronic", ["https://www.youtube.com/watch?v=zEPEJW_Ln6o"]),
# ("Necropolis", "A Covenant of Thorns", "English", 4.6, 5.07, "Electronic", ["https://www.youtube.com/watch?v=NVVTljhQXdk"]),
# ("Tha Dog Song", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=72BsBa9MyHE"]),
# ("Cavalier", "They Live", "English", 3.67, 5.57, "Electronic", ["https://www.youtube.com/watch?v=H5ZXrH8sUF0"]),
# ("Floating", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=KY9u_VLcvmI"]),
# ("Over-sked", "Zan-zan-zawa-veia", "English", 6.83, 5.39, "Electronic", ["https://www.youtube.com/watch?v=fGJcbB6MaYg"]),
# ("Dead Cows", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=hBj0-dIU8HI"]),
# ("Saudade - Part One", "Stray Ghost", "English", 3.95, 5.57, "Electronic", ["https://www.youtube.com/watch?v=_7s67-A2pRs"]),
# ("Carjacker", "Slambo", "English", 5.81, 4.74, "Electronic", ["https://www.youtube.com/watch?v=aMn_2lIgzg4"]),
# ("Consumed", "Slambo", "English", 5.77, 4.66, "Electronic", ["https://www.youtube.com/watch?v=7Dqgr0wNyPo"]),
# ("Dragonkin", "Slambo", "English", 5.78, 4.85, "Electronic", ["https://www.youtube.com/watch?v=_fi9LRMThFs"]),
# ("The Warmth", "Slambo", "English", 5.78, 4.85, "Electronic", ["https://www.youtube.com/watch?v=vsWxs1tuwDk"]),



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
