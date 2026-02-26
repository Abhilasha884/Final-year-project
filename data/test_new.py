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

# r"C:\Users\lapto\Downloads\ffmpeg-2026-02-04-git-627da1111c-essentials_build\ffmpeg-2026-02-04-git-627da1111c-essentials_build\bin"

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


("Adagio for Strings", "Samuel Barber", "English", 0.34, 0.44, "Classical", ["https://www.youtube.com/watch?v=WAoLJ8GbA4Y"]),
("Gymnopédie No.1", "Erik Satie", "English", 0.58, 0.32, "Classical", ["https://www.youtube.com/watch?v=TL0xzp4zzBE"]),
("Clair de Lune", "Claude Debussy", "English", 0.64, 0.36, "Classical", ["https://www.youtube.com/watch?v=-Bxpm0EmOMU"]),
("Canon in D", "Johann Pachelbel", "English", 0.72, 0.52, "Classical", ["https://www.youtube.com/watch?v=Ptk_1Dc2iPY"]),
("Moonlight Sonata", "Ludwig van Beethoven", "English", 0.42, 0.38, "Classical", ["https://www.youtube.com/watch?v=Hu7hscHkfPw"]),
("Lacrimosa", "Wolfgang Amadeus Mozart", "English", 0.36, 0.47, "Classical", ["https://www.youtube.com/watch?v=MafAZeag1_0"]),
("Ave Maria", "Franz Schubert", "English", 0.48, 0.39, "Classical", ["https://www.youtube.com/watch?v=XpYGgtrMTYs"]),
("Nocturne Op.9 No.2", "Frédéric Chopin", "English", 0.6, 0.41, "Classical", ["https://www.youtube.com/watch?v=cW-VRsOeIwM"]),
("Air on the G String", "Johann Sebastian Bach", "English", 0.7, 0.38, "Classical", ["https://www.youtube.com/watch?v=CvglW3KNSsQ"]),
("Toccata and Fugue in D Minor", "Johann Sebastian Bach", "English", 0.46, 0.78, "Classical", ["https://www.youtube.com/watch?v=erXG9vnN-GI"]),
("Brandenburg Concerto No.3", "Johann Sebastian Bach", "English", 0.76, 0.68, "Classical", ["https://www.youtube.com/watch?v=pdsyNwUoON0"]),
("Symphony No.5", "Ludwig van Beethoven", "English", 0.52, 0.86, "Classical", ["https://www.youtube.com/watch?v=3ug835LFixU"]),
("Für Elise", "Ludwig van Beethoven", "English", 0.68, 0.5, "Classical", ["https://www.youtube.com/watch?v=e4d0LOuP4Uw"]),
("Hungarian Rhapsody No.2", "Franz Liszt", "English", 0.8, 0.74, "Classical", ["https://www.youtube.com/watch?v=FT36za3Gyos"]),
("Boléro", "Maurice Ravel", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=Dh9bUD-hC0A"]),
("Pavane", "Gabriel Fauré", "English", 0.6, 0.42, "Classical", ["https://www.youtube.com/watch?v=wQDoN40-_C4"]),
("The Swan", "Camille Saint-Saëns", "English", 0.68, 0.4, "Classical", ["https://www.youtube.com/watch?v=3qrKjywjo7Q"]),
("Nessun Dorma", "Giacomo Puccini", "English", 0.7, 0.66, "Classical", ["https://www.youtube.com/watch?v=cWc7vYjgnTs"]),
("O Mio Babbino Caro", "Giacomo Puccini", "English", 0.62, 0.48, "Classical", ["https://www.youtube.com/watch?v=9J9XMe7zvVg"]),
("Meditation from Thaïs", "Jules Massenet", "English", 0.74, 0.38, "Classical", ["https://www.youtube.com/watch?v=l4nKmbH-Two"]),
("William Tell Overture", "Gioachino Rossini", "English", 0.82, 0.88, "Classical", ["https://www.youtube.com/watch?v=j3T8-aeOrbg"]),
("Ride of the Valkyries", "Richard Wagner", "English", 0.54, 0.92, "Classical", ["https://www.youtube.com/watch?v=hQM97_iNXhk"]),
("Peer Gynt Morning Mood", "Edvard Grieg", "English", 0.88, 0.44, "Classical", ["https://www.youtube.com/watch?v=7lKo6TYDXCQ"]),
("In the Hall of the Mountain King", "Edvard Grieg", "English", 0.48, 0.86, "Classical", ["https://www.youtube.com/watch?v=OqvHWUZZdP0"]),
("The Four Seasons: Spring", "Antonio Vivaldi", "English", 0.9, 0.78, "Classical", ["https://www.youtube.com/watch?v=3LiztfE1X7E"]),
("Winter Largo", "Antonio Vivaldi", "English", 0.62, 0.42, "Classical", ["https://www.youtube.com/watch?v=ZPdk5GaIDjo"]),
("Fantaisie-Impromptu", "Frédéric Chopin", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=H4v4Ipl_UJI"]),
("Prelude in E Minor", "Frédéric Chopin", "English", 0.44, 0.4, "Classical", ["https://www.youtube.com/watch?v=90wBhBZjAUQ"]),
("Waltz in A Minor", "Frédéric Chopin", "English", 0.6, 0.58, "Classical", ["https://www.youtube.com/watch?v=DFzV544BEzU"]),
("Eine kleine Nachtmusik", "Wolfgang Amadeus Mozart", "English", 0.86, 0.66, "Classical", ["https://www.youtube.com/watch?v=r_oK8dKIBYc"]),
("Turkish March", "Wolfgang Amadeus Mozart", "English", 0.88, 0.74, "Classical", ["https://www.youtube.com/watch?v=Cy10pGVmc20"]),
("Symphony No.40", "Wolfgang Amadeus Mozart", "English", 0.52, 0.72, "Classical", ["https://www.youtube.com/watch?v=0sGqkMU-mGQ"]),
("The Blue Danube", "Johann Strauss II", "English", 0.92, 0.62, "Classical", ["https://www.youtube.com/watch?v=_CTYymbbEL4"]),
("Radetzky March", "Johann Strauss I", "English", 0.9, 0.8, "Classical", ["https://www.youtube.com/watch?v=MsoAK2QyhzE"]),
("Swan Lake Theme", "Pyotr Ilyich Tchaikovsky", "English", 0.62, 0.48, "Classical", ["https://www.youtube.com/watch?v=9cNQFB0TDfY"]),
("Nutcracker Waltz of the Flowers", "Pyotr Ilyich Tchaikovsky", "English", 0.84, 0.66, "Classical", ["https://www.youtube.com/watch?v=Zp1aDnVySf8"]),
("1812 Overture", "Pyotr Ilyich Tchaikovsky", "English", 0.7, 0.92, "Classical", ["https://www.youtube.com/watch?v=UH-PHoO7rZM"]),
("Romeo and Juliet Fantasy Overture", "Pyotr Ilyich Tchaikovsky", "English", 0.58, 0.7, "Classical", ["https://www.youtube.com/watch?v=f6qZUCi7ToQ"]),
("Serenade", "Franz Schubert", "English", 0.72, 0.46, "Classical", ["https://www.youtube.com/watch?v=xpd5-KGcVmY"]),
("Symphony No.8 Unfinished", "Franz Schubert", "English", 0.56, 0.64, "Classical", ["https://www.youtube.com/watch?v=3tisvEpblig"]),
("On the Nature of Daylight", "Max Richter", "English", 0.58, 0.4, "Classical", ["https://www.youtube.com/watch?v=InyT9Gyoz_o"]),
("Spiegel im Spiegel", "Arvo Pärt", "English", 0.56, 0.28, "Classical", ["https://www.youtube.com/watch?v=FZe3mXlnfNc"]),
("Cantus in Memory of Benjamin Britten", "Arvo Pärt", "English", 0.42, 0.4, "Classical", ["https://www.youtube.com/watch?v=TVZZgfXNFW8"]),
("Metamorphosis One", "Philip Glass", "English", 0.7, 0.46, "Classical", ["https://www.youtube.com/watch?v=C2inNYauU1o"]),
("Koyaanisqatsi Theme", "Philip Glass", "English", 0.52, 0.54, "Classical", ["https://www.youtube.com/watch?v=_4Vt0UGwmgQ"]),
("Gnossienne No.1", "Erik Satie", "English", 0.56, 0.3, "Classical", ["https://www.youtube.com/watch?v=4Y42P0G-kGY"]),
("Arabesque No.1", "Claude Debussy", "English", 0.7, 0.46, "Classical", ["https://www.youtube.com/watch?v=uzeLpaPjZNU"]),
("Prélude à l'après-midi d'un faune", "Claude Debussy", "English", 0.64, 0.4, "Classical", ["https://www.youtube.com/watch?v=Y9iDOt2WbjY"]),
("Piano Concerto No.21", "Wolfgang Amadeus Mozart", "English", 0.74, 0.54, "Classical", ["https://www.youtube.com/watch?v=fNU-XAZjhzA"]),
("Piano Sonata No.14", "Ludwig van Beethoven", "English", 0.54, 0.36, "Classical", ["https://www.youtube.com/watch?v=UFLm9-LL8Kw"]),
("Symphony No.7", "Ludwig van Beethoven", "English", 0.66, 0.74, "Classical", ["https://www.youtube.com/watch?v=TTFG1jVKbPA"]),
("Symphony No.6 Pastoral", "Ludwig van Beethoven", "English", 0.78, 0.56, "Classical", ["https://www.youtube.com/watch?v=xWDIwsfgQnE"]),
("Violin Concerto", "Ludwig van Beethoven", "English", 0.72, 0.5, "Classical", ["https://www.youtube.com/watch?v=cokCgWPRZPg"]),
("Piano Sonata No.8 Pathetique", "Ludwig van Beethoven", "English", 0.6, 0.52, "Classical", ["https://www.youtube.com/watch?v=OWF66WkAnXs"]),
("Symphony No.3 Eroica", "Ludwig van Beethoven", "English", 0.66, 0.82, "Classical", ["https://www.youtube.com/watch?v=DWwppYEEdcI"]),
("Goldberg Variations", "Johann Sebastian Bach", "English", 0.74, 0.46, "Classical", ["https://www.youtube.com/watch?v=eZCSOdi19jQ"]),
("Mass in B Minor", "Johann Sebastian Bach", "English", 0.62, 0.56, "Classical", ["https://www.youtube.com/watch?v=3FLbiDrn8IE"]),
("Well-Tempered Clavier I", "Johann Sebastian Bach", "English", 0.7, 0.52, "Classical", ["https://www.youtube.com/watch?v=gVah1cr3pU0"]),
("Violin Partita No.2", "Johann Sebastian Bach", "English", 0.68, 0.66, "Classical", ["https://www.youtube.com/watch?v=44Wz92zQe04"]),
("Magnificat", "Johann Sebastian Bach", "English", 0.78, 0.68, "Classical", ["https://www.youtube.com/watch?v=EsUWG2axB3w"]),
("Piano Concerto No.23", "Wolfgang Amadeus Mozart", "English", 0.76, 0.58, "Classical", ["https://www.youtube.com/watch?v=m0Tk3sliZ0U"]),
("Symphony No.25", "Wolfgang Amadeus Mozart", "English", 0.64, 0.72, "Classical", ["https://www.youtube.com/watch?v=Bus28U_RZ8c"]),
("Symphony No.36 Linz", "Wolfgang Amadeus Mozart", "English", 0.78, 0.68, "Classical", ["https://www.youtube.com/watch?v=ThY-JK10nrc"]),
("Piano Sonata No.16", "Wolfgang Amadeus Mozart", "English", 0.82, 0.66, "Classical", ["https://www.youtube.com/watch?v=1vDxlnJVvW8"]),
("Clarinet Concerto", "Wolfgang Amadeus Mozart", "English", 0.72, 0.5, "Classical", ["https://www.youtube.com/watch?v=YT_63UntRJE"]),
("Piano Concerto No.1", "Pyotr Ilyich Tchaikovsky", "English", 0.76, 0.84, "Classical", ["https://www.youtube.com/watch?v=2DmfJu3oNDM"]),
("Violin Concerto", "Pyotr Ilyich Tchaikovsky", "English", 0.7, 0.76, "Classical", ["https://www.youtube.com/watch?v=2Q_DzWUvcL8"]),
("Manfred Symphony", "Pyotr Ilyich Tchaikovsky", "English", 0.56, 0.78, "Classical", ["https://www.youtube.com/watch?v=_Ws05zhoL9g"]),
("Capriccio Italien", "Pyotr Ilyich Tchaikovsky", "English", 0.82, 0.8, "Classical", ["https://www.youtube.com/watch?v=tFQU8OnE3_M"]),
("Symphony No.4", "Pyotr Ilyich Tchaikovsky", "English", 0.58, 0.82, "Classical", ["https://www.youtube.com/watch?v=Y7G5ithbFys"]),
("Symphony No.2", "Johannes Brahms", "English", 0.72, 0.6, "Classical", ["https://www.youtube.com/watch?v=EDYauTx76yI"]),
("Symphony No.4", "Johannes Brahms", "English", 0.6, 0.74, "Classical", ["https://www.youtube.com/watch?v=o69YVL_XKJo"]),
("Hungarian Dance No.5", "Johannes Brahms", "English", 0.82, 0.78, "Classical", ["https://www.youtube.com/watch?v=Nzo3atXtm54"]),
("Violin Concerto", "Johannes Brahms", "English", 0.7, 0.68, "Classical", ["https://www.youtube.com/watch?v=UFl9xuYP5T8"]),
("Intermezzo Op.118 No.2", "Johannes Brahms", "English", 0.66, 0.44, "Classical", ["https://www.youtube.com/watch?v=7Wo4IPNMzWQ"]),
("Symphony No.1", "Gustav Mahler", "English", 0.62, 0.76, "Classical", ["https://www.youtube.com/watch?v=4XbHLFkg_Mw"]),
("Symphony No.5 Adagietto", "Gustav Mahler", "English", 0.7, 0.46, "Classical", ["https://www.youtube.com/watch?v=Bj6KLv7kv2Q"]),
("Symphony No.2 Resurrection", "Gustav Mahler", "English", 0.66, 0.82, "Classical", ["https://www.youtube.com/watch?v=4MPuoOj5TIw"]),
("Das Lied von der Erde", "Gustav Mahler", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=5c7LzrI0ELw"]),
("Kindertotenlieder", "Gustav Mahler", "English", 0.52, 0.48, "Classical", ["https://www.youtube.com/watch?v=Sx1fv5q7Wiw"]),
("Finlandia", "Jean Sibelius", "English", 0.78, 0.8, "Classical", ["https://www.youtube.com/watch?v=fE0RbPsC9uE"]),
("Symphony No.5", "Jean Sibelius", "English", 0.72, 0.74, "Classical", ["https://www.youtube.com/watch?v=RRS6dIgn_QI"]),
("Violin Concerto", "Jean Sibelius", "English", 0.66, 0.72, "Classical", ["https://www.youtube.com/watch?v=J0w0t4Qn6LY"]),
("Valse Triste", "Jean Sibelius", "English", 0.54, 0.5, "Classical", ["https://www.youtube.com/watch?v=Iys6ZqDFerA"]),
("Symphony No.1", "Jean Sibelius", "English", 0.6, 0.68, "Classical", ["https://www.youtube.com/watch?v=dCIw_4oJ4Gg"]),
("The Firebird Suite", "Igor Stravinsky", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=rYcz-g8WpMc"]),
("The Rite of Spring", "Igor Stravinsky", "English", 0.52, 0.92, "Classical", ["https://www.youtube.com/watch?v=gqJdR8_cw18"]),
("Petrushka", "Igor Stravinsky", "English", 0.7, 0.84, "Classical", ["https://www.youtube.com/watch?v=esD90diWZds"]),
("Pulcinella Suite", "Igor Stravinsky", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=glbnrYF_0ck"]),
("Symphony of Psalms", "Igor Stravinsky", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=AEx9NxFJ09Y"]),
("Water Music Suite", "George Frideric Handel", "English", 0.84, 0.68, "Classical", ["https://www.youtube.com/watch?v=Kuw8YjSbKd4"]),
("Music for the Royal Fireworks", "George Frideric Handel", "English", 0.86, 0.72, "Classical", ["https://www.youtube.com/watch?v=EkttBYzD-jY"]),
("Messiah: Hallelujah Chorus", "George Frideric Handel", "English", 0.92, 0.74, "Classical", ["https://www.youtube.com/watch?v=IUZEtVbJT5c"]),
("Concerto Grosso Op.6 No.5", "George Frideric Handel", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=4ISLEvMU9jY"]),
("Sarabande", "George Frideric Handel", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=xOLQd_pUbxs"]),
("Symphony No.94 Surprise", "Joseph Haydn", "English", 0.86, 0.68, "Classical", ["https://www.youtube.com/watch?v=tF5kr251BRs"]),
("Symphony No.101 Clock", "Joseph Haydn", "English", 0.88, 0.7, "Classical", ["https://www.youtube.com/watch?v=qPx4m5FJRaE"]),
("Symphony No.104 London", "Joseph Haydn", "English", 0.84, 0.72, "Classical", ["https://www.youtube.com/watch?v=wtQ-soq1Ito"]),
("Trumpet Concerto", "Joseph Haydn", "English", 0.9, 0.74, "Classical", ["https://www.youtube.com/watch?v=NHjgSiTBddM"]),
("The Creation", "Joseph Haydn", "English", 0.82, 0.6, "Classical", ["https://www.youtube.com/watch?v=EuIs7R2BpvQ"]),
("Requiem", "Gabriel Fauré", "English", 0.64, 0.48, "Classical", ["https://www.youtube.com/watch?v=p-uzBqbMUvc"]),
("Cantique de Jean Racine", "Gabriel Fauré", "English", 0.7, 0.44, "Classical", ["https://www.youtube.com/watch?v=g16zSj6Ynko"]),
("Pavane Op.50", "Gabriel Fauré", "English", 0.68, 0.46, "Classical", ["https://www.youtube.com/watch?v=wQDoN40-_C4"]),
("Nocturne Op.33 No.1", "Gabriel Fauré", "English", 0.66, 0.42, "Classical", ["https://www.youtube.com/watch?v=UIt0S63cRSQ"]),
("Requiem: Pie Jesu", "Gabriel Fauré", "English", 0.62, 0.4, "Classical", ["https://www.youtube.com/watch?v=xTf14maKtT8"]),
("Peer Gynt Suite Morning Mood", "Edvard Grieg", "English", 0.9, 0.46, "Classical", ["https://www.youtube.com/watch?v=7lKo6TYDXCQ"]),
("Holberg Suite Sarabande", "Edvard Grieg", "English", 0.72, 0.48, "Classical", ["https://www.youtube.com/watch?v=j1wQ8ZMZq60"]),
("Lyric Pieces Arietta", "Edvard Grieg", "English", 0.7, 0.4, "Classical", ["https://www.youtube.com/watch?v=5TbQftYOKms"]),
("Norwegian Dance No.2", "Edvard Grieg", "English", 0.84, 0.68, "Classical", ["https://www.youtube.com/watch?v=jlZ3mwQfFMk"]),
("Piano Concerto in A Minor", "Edvard Grieg", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=Kjx-AxpZUxQ"]),
("Caprice No.24", "Niccolò Paganini", "English", 0.82, 0.9, "Classical", ["https://www.youtube.com/watch?v=YCsVEsQlm7o"]),
("Violin Concerto No.1", "Niccolò Paganini", "English", 0.8, 0.86, "Classical", ["https://www.youtube.com/watch?v=b_wdho14Xck"]),
("Moto Perpetuo", "Niccolò Paganini", "English", 0.88, 0.92, "Classical", ["https://www.youtube.com/watch?v=YCsVEsQlm7o"]),
("La Campanella", "Niccolò Paganini", "English", 0.84, 0.88, "Classical", ["https://www.youtube.com/watch?v=6ruHDWSNvB8"]),
("Variations on God Save the King", "Niccolò Paganini", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=r3tlG9nssZo"]),
("Adagio in G Minor", "Tomaso Albinoni", "English", 0.6, 0.46, "Classical", ["https://www.youtube.com/watch?v=u99f9RAvwu4"]),
("Concerto for Oboe in D Minor", "Tomaso Albinoni", "English", 0.74, 0.66, "Classical", ["https://www.youtube.com/watch?v=dLxJLrjvl8A"]),
("Sonata in G Minor", "Tomaso Albinoni", "English", 0.68, 0.54, "Classical", ["https://www.youtube.com/watch?v=u99f9RAvwu4"]),
("Sinfonia in C Major", "Tomaso Albinoni", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=k0kwj8SiN7I"]),
("Concerto Op.9 No.2", "Tomaso Albinoni", "English", 0.72, 0.64, "Classical", ["https://www.youtube.com/watch?v=dLxJLrjvl8A"]),
("Piano Concerto No.1", "Sergei Rachmaninoff", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=y6EX3t2Mdnw"]),
("Piano Concerto No.2", "Sergei Rachmaninoff", "English", 0.72, 0.82, "Classical", ["https://www.youtube.com/watch?v=rEGOihjqO9w"]),
("Piano Concerto No.3", "Sergei Rachmaninoff", "English", 0.74, 0.86, "Classical", ["https://www.youtube.com/watch?v=DPJL488cfRw"]),
("Rhapsody on a Theme of Paganini", "Sergei Rachmaninoff", "English", 0.68, 0.76, "Classical", ["https://www.youtube.com/watch?v=ppJ5uITLECE"]),
("Vocalise", "Sergei Rachmaninoff", "English", 0.66, 0.44, "Classical", ["https://www.youtube.com/watch?v=hiT_v5DOsUw"]),
("Symphony No.1", "Alexander Borodin", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=UiMv7-8WM7U"]),
("Polovtsian Dances", "Alexander Borodin", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=kqKclPhsK0o"]),
("String Quartet No.2", "Alexander Borodin", "English", 0.72, 0.54, "Classical", ["https://www.youtube.com/watch?v=2YAzUC6LzNk"]),
("Prince Igor Overture", "Alexander Borodin", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=R7uZOXDOcHA"]),
("In the Steppes of Central Asia", "Alexander Borodin", "English", 0.7, 0.52, "Classical", ["https://www.youtube.com/watch?v=g4tlQxaHetI"]),
("Scheherazade", "Nikolai Rimsky-Korsakov", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=zY4w4_W30aQ"]),
("Capriccio Espagnol", "Nikolai Rimsky-Korsakov", "English", 0.84, 0.78, "Classical", ["https://www.youtube.com/watch?v=Lh6mDL-VwYw"]),
("Russian Easter Overture", "Nikolai Rimsky-Korsakov", "English", 0.82, 0.8, "Classical", ["https://www.youtube.com/watch?v=M7BPlOirPig"]),
("Antar Symphony", "Nikolai Rimsky-Korsakov", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=NiTInQcjy4o"]),
("Flight of the Bumblebee", "Nikolai Rimsky-Korsakov", "English", 0.86, 0.94, "Classical", ["https://www.youtube.com/watch?v=59QXMCsx_5E"]),
("The Planets: Uranus", "Gustav Holst", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=fUyVFKA1MaU"]),
("The Planets: Mercury", "Gustav Holst", "English", 0.7, 0.6, "Classical", ["https://www.youtube.com/watch?v=3dcMSNgvxpU"]),
("The Planets: Neptune", "Gustav Holst", "English", 0.56, 0.34, "Classical", ["https://www.youtube.com/watch?v=oFMXNUHuWug"]),
("The Planets: Jupiter", "Gustav Holst", "English", 0.82, 0.72, "Classical", ["https://www.youtube.com/watch?v=BUM_zT3YKHs"]),
("The Planets: Venus", "Gustav Holst", "English", 0.64, 0.4, "Classical", ["https://www.youtube.com/watch?v=mp5gksq_OEI"]),
("Symphony No.6", "Gustav Mahler", "English", 0.58, 0.86, "Classical", ["https://www.youtube.com/watch?v=YsEo1PsSmbg"]),
("Symphony No.7", "Gustav Mahler", "English", 0.62, 0.82, "Classical", ["https://www.youtube.com/watch?v=c8_CWVAWzPs"]),
("Symphony No.8", "Gustav Mahler", "English", 0.7, 0.88, "Classical", ["https://www.youtube.com/watch?v=gOddn8-35c0"]),
("Symphony No.9", "Gustav Mahler", "English", 0.6, 0.74, "Classical", ["https://www.youtube.com/watch?v=tkChdHBuoiQ"]),
("Das Klagende Lied", "Gustav Mahler", "English", 0.54, 0.7, "Classical", ["https://www.youtube.com/watch?v=ukpldF06G24"]),
("Symphony No.2", "Carl Nielsen", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=bq_c8Za33To"]),
("Symphony No.3", "Carl Nielsen", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=LXgeRS4i4ME"]),
("Symphony No.4 The Inextinguishable", "Carl Nielsen", "English", 0.74, 0.86, "Classical", ["https://www.youtube.com/watch?v=IC0Lf03sGw4"]),
("Symphony No.5", "Carl Nielsen", "English", 0.68, 0.8, "Classical", ["https://www.youtube.com/watch?v=ALryN2UJYDw"]),
("Helios Overture", "Carl Nielsen", "English", 0.82, 0.74, "Classical", ["https://www.youtube.com/watch?v=VBoJPsY09tU"]),
("Symphony No.1", "Edward Elgar", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=sCuSuwDXxUA"]),
("Symphony No.2", "Edward Elgar", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=R2-43p3GVTQ"]),
("Enigma Variations", "Edward Elgar", "English", 0.74, 0.68, "Classical", ["https://www.youtube.com/watch?v=7iM5dymBBI4"]),
("Pomp and Circumstance No.1", "Edward Elgar", "English", 0.88, 0.82, "Classical", ["https://www.youtube.com/watch?v=Spx4kmY67Wc"]),
("Cello Concerto", "Edward Elgar", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=8V41D0Uczf0"]),
("Symphony No.1", "William Walton", "English", 0.68, 0.78, "Classical", ["https://www.youtube.com/watch?v=FWehYkPwjF4"]),
("Viola Concerto", "William Walton", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=2xAZPIjvxYg"]),
("Belshazzar's Feast", "William Walton", "English", 0.72, 0.8, "Classical", ["https://www.youtube.com/watch?v=ex3Yn7-ebzw"]),
("Facade Suite", "William Walton", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=BX0BLdAFV-c"]),
("Orb and Sceptre", "William Walton", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=v6qjUdaDE_Q"]),
("Fantasia on a Theme by Thomas Tallis", "Ralph Vaughan Williams", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=e6pEIHtffqQ"]),
("The Lark Ascending", "Ralph Vaughan Williams", "English", 0.88, 0.46, "Classical", ["https://www.youtube.com/watch?v=IOWN5fQnzGk"]),
("Symphony No.5", "Ralph Vaughan Williams", "English", 0.72, 0.6, "Classical", ["https://www.youtube.com/watch?v=EcR8KUuca3A"]),
("Norfolk Rhapsody", "Ralph Vaughan Williams", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=5DeT3DkyXc8"]),
("Sea Symphony", "Ralph Vaughan Williams", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=2kR3FzJ1Hh4"]),
("Symphony No.1", "Benjamin Britten", "English", 0.7, 0.76, "Classical", ["https://www.youtube.com/watch?v=90ckhD19YgE"]),
("War Requiem", "Benjamin Britten", "English", 0.52, 0.72, "Classical", ["https://www.youtube.com/watch?v=625WOYzdvFw"]),
("Young Person's Guide to the Orchestra", "Benjamin Britten", "English", 0.86, 0.74, "Classical", ["https://www.youtube.com/watch?v=4vbvhU22uAM"]),
("Simple Symphony", "Benjamin Britten", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=90ckhD19YgE"]),
("Serenade for Tenor", "Benjamin Britten", "English", 0.64, 0.56, "Classical", ["https://www.youtube.com/watch?v=mkLyK-oSQ7A"]),
("Concerto for Orchestra", "Béla Bartók", "English", 0.72, 0.82, "Classical", ["https://www.youtube.com/watch?v=0DKuzKnkIro"]),
("Music for Strings, Percussion and Celesta", "Béla Bartók", "English", 0.64, 0.78, "Classical", ["https://www.youtube.com/watch?v=QElT9KD4uX8"]),
("Romanian Folk Dances", "Béla Bartók", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=Z50Ooqv1GFg"]),
("Bluebeard's Castle", "Béla Bartók", "English", 0.58, 0.72, "Classical", ["https://www.youtube.com/watch?v=q27M5ugAoJI"]),
("Piano Concerto No.2", "Béla Bartók", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=Qr_JUGdoVSY"]),
("Sinfonietta", "Leoš Janáček", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=9aFTv50AoEQ"]),
("Taras Bulba", "Leoš Janáček", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=JFz2Hle2YSg"]),
("Glagolitic Mass", "Leoš Janáček", "English", 0.7, 0.74, "Classical", ["https://www.youtube.com/watch?v=JfYqCCgWRek"]),
("Jenůfa Suite", "Leoš Janáček", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=XFyPYsC74eQ"]),
("The Cunning Little Vixen Suite", "Leoš Janáček", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=N0W_nFSpeO0"]),
("Symphony No.1", "Karol Szymanowski", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=MYu96EtWS04"]),
("Symphony No.3 Song of the Night", "Karol Szymanowski", "English", 0.72, 0.76, "Classical", ["https://www.youtube.com/watch?v=a1yW8Ynlx7k"]),
("Violin Concerto No.1", "Karol Szymanowski", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=uT00G955oUg"]),
("Myths", "Karol Szymanowski", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=U5mUn1AQ4JU"]),
("Stabat Mater", "Karol Szymanowski", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=soC2WFYgWH0"]),
("Symphony No.1", "Alfred Schnittke", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=zQ8AIfxMIe4"]),
("Concerto Grosso No.1", "Alfred Schnittke", "English", 0.62, 0.78, "Classical", ["https://www.youtube.com/watch?v=eE3xPdT5jx8"]),
("Faust Cantata", "Alfred Schnittke", "English", 0.56, 0.7, "Classical", ["https://www.youtube.com/watch?v=OnwX39FJGwU"]),
("Choir Concerto", "Alfred Schnittke", "English", 0.6, 0.66, "Classical", ["https://www.youtube.com/watch?v=d-Oxs3ynYpA"]),
("Requiem", "Alfred Schnittke", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=SdrJzKTrVaE"]),
("Symphony No.1", "Edward Elgar", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=sCuSuwDXxUA"]),
("Symphony No.2", "Edward Elgar", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=R2-43p3GVTQ"]),
("Enigma Variations", "Edward Elgar", "English", 0.74, 0.68, "Classical", ["https://www.youtube.com/watch?v=7iM5dymBBI4"]),
("Pomp and Circumstance No.1", "Edward Elgar", "English", 0.88, 0.82, "Classical", ["https://www.youtube.com/watch?v=Spx4kmY67Wc"]),
("Cello Concerto", "Edward Elgar", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=8V41D0Uczf0"]),
("Symphony No.1", "William Walton", "English", 0.68, 0.78, "Classical", ["https://www.youtube.com/watch?v=FWehYkPwjF4"]),
("Viola Concerto", "William Walton", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=2xAZPIjvxYg"]),
("Belshazzar's Feast", "William Walton", "English", 0.72, 0.8, "Classical", ["https://www.youtube.com/watch?v=ex3Yn7-ebzw"]),
("Facade Suite", "William Walton", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=BX0BLdAFV-c"]),
("Orb and Sceptre", "William Walton", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=v6qjUdaDE_Q"]),
("Fantasia on a Theme by Thomas Tallis", "Ralph Vaughan Williams", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=ihx5LCF1yJY"]),
("The Lark Ascending", "Ralph Vaughan Williams", "English", 0.88, 0.46, "Classical", ["https://www.youtube.com/watch?v=IOWN5fQnzGk"]),
("Symphony No.5", "Ralph Vaughan Williams", "English", 0.72, 0.6, "Classical", ["https://www.youtube.com/watch?v=EcR8KUuca3A"]),
("Norfolk Rhapsody", "Ralph Vaughan Williams", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=5DeT3DkyXc8"]),
("Sea Symphony", "Ralph Vaughan Williams", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=2kR3FzJ1Hh4"]),
("Symphony No.1", "Benjamin Britten", "English", 0.7, 0.76, "Classical", ["https://www.youtube.com/watch?v=90ckhD19YgE"]),
("War Requiem", "Benjamin Britten", "English", 0.52, 0.72, "Classical", ["https://www.youtube.com/watch?v=625WOYzdvFw"]),
("Young Person's Guide to the Orchestra", "Benjamin Britten", "English", 0.86, 0.74, "Classical", ["https://www.youtube.com/watch?v=4vbvhU22uAM"]),
("Simple Symphony", "Benjamin Britten", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=90ckhD19YgE"]),
("Serenade for Tenor", "Benjamin Britten", "English", 0.64, 0.56, "Classical", ["https://www.youtube.com/watch?v=AYq71Ypxu8w"]),
("Concerto for Orchestra", "Béla Bartók", "English", 0.72, 0.82, "Classical", ["https://www.youtube.com/watch?v=0DKuzKnkIro"]),
("Music for Strings, Percussion and Celesta", "Béla Bartók", "English", 0.64, 0.78, "Classical", ["https://www.youtube.com/watch?v=2EsNGS9vYe8"]),
("Romanian Folk Dances", "Béla Bartók", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=Z50Ooqv1GFg"]),
("Bluebeard's Castle", "Béla Bartók", "English", 0.58, 0.72, "Classical", ["https://www.youtube.com/watch?v=q27M5ugAoJI"]),
("Piano Concerto No.2", "Béla Bartók", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=Qr_JUGdoVSY"]),
("Sinfonietta", "Leoš Janáček", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=9aFTv50AoEQ"]),
("Taras Bulba", "Leoš Janáček", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=JFz2Hle2YSg"]),
("Glagolitic Mass", "Leoš Janáček", "English", 0.7, 0.74, "Classical", ["https://www.youtube.com/watch?v=JfYqCCgWRek"]),
("Jenůfa Suite", "Leoš Janáček", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=XFyPYsC74eQ"]),
("The Cunning Little Vixen Suite", "Leoš Janáček", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=N0W_nFSpeO0"]),
("Symphony No.1", "Karol Szymanowski", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=MYu96EtWS04"]),
("Symphony No.3 Song of the Night", "Karol Szymanowski", "English", 0.72, 0.76, "Classical", ["https://www.youtube.com/watch?v=hgGdW5HEAZY"]),
("Violin Concerto No.1", "Karol Szymanowski", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=uT00G955oUg"]),
("Myths", "Karol Szymanowski", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=U5mUn1AQ4JU"]),
("Stabat Mater", "Karol Szymanowski", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=soC2WFYgWH0"]),
("Symphony No.1", "Alfred Schnittke", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=zQ8AIfxMIe4"]),
("Concerto Grosso No.1", "Alfred Schnittke", "English", 0.62, 0.78, "Classical", ["https://www.youtube.com/watch?v=eE3xPdT5jx8"]),
("Faust Cantata", "Alfred Schnittke", "English", 0.56, 0.7, "Classical", ["https://www.youtube.com/watch?v=OnwX39FJGwU"]),
("Choir Concerto", "Alfred Schnittke", "English", 0.6, 0.66, "Classical", ["https://www.youtube.com/watch?v=d-Oxs3ynYpA"]),
("Requiem", "Alfred Schnittke", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=xbFk3FA0D6E"]),
("Symphony No.3", "Henryk Górecki", "English", 0.54, 0.48, "Classical", ["https://www.youtube.com/watch?v=ZTmspJxshEg"]),
("Totus Tuus", "Henryk Górecki", "English", 0.66, 0.42, "Classical", ["https://www.youtube.com/watch?v=Gz7s_NJZ7X4"]),
("Beatus Vir", "Henryk Górecki", "English", 0.62, 0.46, "Classical", ["https://www.youtube.com/watch?v=PU3rp94JUc4"]),
("Concerto-Cantata", "Henryk Górecki", "English", 0.6, 0.5, "Classical", ["https://www.youtube.com/watch?v=QBNHTzP0aw0"]),
("Three Pieces in Old Style", "Henryk Górecki", "English", 0.7, 0.56, "Classical", ["https://www.youtube.com/watch?v=Wb73v05adG4"]),
("Symphony No.4", "Witold Lutosławski", "English", 0.68, 0.82, "Classical", ["https://www.youtube.com/watch?v=hIGgmKyK0bc"]),
("Concerto for Orchestra", "Witold Lutosławski", "English", 0.72, 0.86, "Classical", ["https://www.youtube.com/watch?v=-IYw_8xzkWE"]),
("Chain 2", "Witold Lutosławski", "English", 0.66, 0.74, "Classical", ["https://www.youtube.com/watch?v=Vxp9Q4k28q4"]),
("Livre pour orchestre", "Witold Lutosławski", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=v3c0zgGsgkA"]),
("Partita for Violin and Orchestra", "Witold Lutosławski", "English", 0.74, 0.76, "Classical", ["https://www.youtube.com/watch?v=UtKizDvvGL8"]),
("Symphony No.1", "Krzysztof Penderecki", "English", 0.6, 0.78, "Classical", ["https://www.youtube.com/watch?v=RZr4X0ClVo8"]),
("Polish Requiem", "Krzysztof Penderecki", "English", 0.56, 0.66, "Classical", ["https://www.youtube.com/watch?v=0ukgUxGlOz4"]),
("Credo", "Krzysztof Penderecki", "English", 0.62, 0.64, "Classical", ["https://www.youtube.com/watch?v=lipV5O4jqCI"]),
("St. Luke Passion", "Krzysztof Penderecki", "English", 0.54, 0.72, "Classical", ["https://www.youtube.com/watch?v=TvJLftkZTU8"]),
("Symphony No.8 Songs of Transience", "Krzysztof Penderecki", "English", 0.64, 0.6, "Classical", ["https://www.youtube.com/watch?v=Tpxh4K3RWrc"]),
("Symphony No.1", "John Adams", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=5LoUm_r7It8"]),
("Harmonielehre", "John Adams", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=peuuOYs5o38"]),
("Short Ride in a Fast Machine", "John Adams", "English", 0.88, 0.92, "Classical", ["https://www.youtube.com/watch?v=5LoUm_r7It8"]),
("Shaker Loops", "John Adams", "English", 0.78, 0.84, "Classical", ["https://www.youtube.com/watch?v=7twYdU3CgGU"]),
("Doctor Atomic Symphony", "John Adams", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=53EtWPealDE"]),
("Symphony No.1", "Philip Glass", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=5lGNtPXcA04"]),
("Violin Concerto", "Philip Glass", "English", 0.76, 0.74, "Classical", ["https://www.youtube.com/watch?v=5lGNtPXcA04"]),
("The Hours", "Philip Glass", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=Wkof3nPK--Y"]),
("Einstein on the Beach", "Philip Glass", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=YlVrdvCPq20"]),
("Akhnaten", "Philip Glass", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=rSn_UAquOfw"]),
("Symphony No.1", "Sergei Taneyev", "English", 0.68, 0.7, "Classical", ["https://www.youtube.com/watch?v=HHJhgxZ9NOs"]),
("Symphony No.4", "Sergei Taneyev", "English", 0.66, 0.72, "Classical", ["https://www.youtube.com/watch?v=EMYaIYKLALI"]),
("John of Damascus", "Sergei Taneyev", "English", 0.62, 0.6, "Classical", ["https://www.youtube.com/watch?v=wa73B0E20-U"]),
("Cantata at the Reading of a Psalm", "Sergei Taneyev", "English", 0.64, 0.58, "Classical", ["https://www.youtube.com/watch?v=dxOVM1u9SSU"]),
("String Quartet No.9", "Sergei Taneyev", "English", 0.7, 0.66, "Classical", ["https://www.youtube.com/watch?v=hJ3XQKxlmLc"]),
("Symphony No.1", "Alexander Glazunov", "English", 0.74, 0.7, "Classical", ["https://www.youtube.com/watch?v=pe8s8MMyUrw"]),
("Symphony No.5", "Alexander Glazunov", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=95C8Ds6_a00"]),
("The Seasons", "Alexander Glazunov", "English", 0.8, 0.68, "Classical", ["https://www.youtube.com/watch?v=EHYMrot35SQ"]),
("Violin Concerto", "Alexander Glazunov", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=jq8oxIgyp8E"]),
("Raymonda Suite", "Alexander Glazunov", "English", 0.82, 0.7, "Classical", ["https://www.youtube.com/watch?v=wXEEoVI3m3I"]),
("Symphony No.1", "Nikolai Myaskovsky", "English", 0.6, 0.68, "Classical", ["https://www.youtube.com/watch?v=gbcqwEdYjjM"]),
("Symphony No.6", "Nikolai Myaskovsky", "English", 0.58, 0.7, "Classical", ["https://www.youtube.com/watch?v=MEughaH6Zww"]),
("Symphony No.21", "Nikolai Myaskovsky", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=c0hU4wQCCQM"]),
("Cello Concerto", "Nikolai Myaskovsky", "English", 0.62, 0.64, "Classical", ["https://www.youtube.com/watch?v=O3d_5TlAzaE"]),
("Sinfonietta", "Nikolai Myaskovsky", "English", 0.7, 0.6, "Classical", ["https://www.youtube.com/watch?v=GX2N0-iavn8"]),
("Symphony No.1", "Mieczysław Weinberg", "English", 0.64, 0.74, "Classical", ["https://www.youtube.com/watch?v=kldCrwaJMro"]),
("Symphony No.5", "Mieczysław Weinberg", "English", 0.6, 0.72, "Classical", ["https://www.youtube.com/watch?v=BYOVegrNuuE"]),
("Violin Concerto", "Mieczysław Weinberg", "English", 0.68, 0.7, "Classical", ["https://www.youtube.com/watch?v=-8gpHVXlnNc"]),
("Cello Concerto", "Mieczysław Weinberg", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=dPu-p7IbYTg"]),
("Polish Tunes", "Mieczysław Weinberg", "English", 0.72, 0.64, "Classical", ["https://www.youtube.com/watch?v=u0st8qwJXNg"]),
("Symphony No.1", "Aram Khachaturian", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=TaPYtH5SZ7E"]),
("Symphony No.2", "Aram Khachaturian", "English", 0.7, 0.8, "Classical", ["https://www.youtube.com/watch?v=-j1Ai_TNe-k"]),
("Gayane Suite", "Aram Khachaturian", "English", 0.84, 0.82, "Classical", ["https://www.youtube.com/watch?v=8ob0nRhSKAw"]),
("Masquerade Suite", "Aram Khachaturian", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=pnvsy9ZIQg0"]),
("Violin Concerto", "Aram Khachaturian", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=PjDkTYPgMWs"]),
("Symphony No.1", "Sergei Lyapunov", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=g3DIAqmSPeI"]),
("Transcendental Etudes", "Sergei Lyapunov", "English", 0.78, 0.84, "Classical", ["https://www.youtube.com/watch?v=ZuWteoXlJqs"]),
("Piano Concerto No.1", "Sergei Lyapunov", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=vjWcYtInBM0"]),
("Rhapsody on Ukrainian Themes", "Sergei Lyapunov", "English", 0.82, 0.76, "Classical", ["https://www.youtube.com/watch?v=U-I0D4kN1FI"]),
("Tarantella", "Sergei Lyapunov", "English", 0.86, 0.88, "Classical", ["https://www.youtube.com/watch?v=dn-aYobnzlI"]),
("Symphony No.1", "Reinhold Glière", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=wO_ViXnO5bI"]),
("Symphony No.3 Ilya Muromets", "Reinhold Glière", "English", 0.66, 0.82, "Classical", ["https://www.youtube.com/watch?v=XGr7Eaz-_Ac"]),
("Horn Concerto", "Reinhold Glière", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=tmfnfPAMn50"]),
("The Red Poppy Suite", "Reinhold Glière", "English", 0.82, 0.76, "Classical", ["https://www.youtube.com/watch?v=4dwYlX8kL64"]),
("Harp Concerto", "Reinhold Glière", "English", 0.76, 0.64, "Classical", ["https://www.youtube.com/watch?v=t7Pai-BASak"]),
("Symphony No.1", "Heitor Villa-Lobos", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=4QIy8dnkHdw"]),
("Bachianas Brasileiras No.5", "Heitor Villa-Lobos", "English", 0.84, 0.6, "Classical", ["https://www.youtube.com/watch?v=pUCuEd1tjCg"]),
("Chôros No.10", "Heitor Villa-Lobos", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=fyKS0cmkRp0"]),
("Uirapuru", "Heitor Villa-Lobos", "English", 0.74, 0.7, "Classical", ["https://www.youtube.com/watch?v=72Pc-aamp0M"]),
("Floresta do Amazonas", "Heitor Villa-Lobos", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=bZ6oyNp9nCQ"]),
("Sinfonietta", "Leoš Janáček", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=9aFTv50AoEQ"]),
("Taras Bulba", "Leoš Janáček", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=JFz2Hle2YSg"]),
("Glagolitic Mass", "Leoš Janáček", "English", 0.7, 0.74, "Classical", ["https://www.youtube.com/watch?v=JfYqCCgWRek"]),
("Jenůfa Suite", "Leoš Janáček", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=XFyPYsC74eQ"]),
("The Cunning Little Vixen Suite", "Leoš Janáček", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=N0W_nFSpeO0"]),
("Symphony No.1", "Karol Szymanowski", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=MYu96EtWS04"]),
("Symphony No.3 Song of the Night", "Karol Szymanowski", "English", 0.72, 0.76, "Classical", ["https://www.youtube.com/watch?v=hgGdW5HEAZY"]),
("Violin Concerto No.1", "Karol Szymanowski", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=uT00G955oUg"]),
("Myths", "Karol Szymanowski", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=U5mUn1AQ4JU"]),
("Stabat Mater", "Karol Szymanowski", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=soC2WFYgWH0"]),
("Symphony No.1", "Alfred Schnittke", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=zQ8AIfxMIe4"]),
("Concerto Grosso No.1", "Alfred Schnittke", "English", 0.62, 0.78, "Classical", ["https://www.youtube.com/watch?v=eE3xPdT5jx8"]),
("Faust Cantata", "Alfred Schnittke", "English", 0.56, 0.7, "Classical", ["https://www.youtube.com/watch?v=OnwX39FJGwU"]),
("Choir Concerto", "Alfred Schnittke", "English", 0.6, 0.66, "Classical", ["https://www.youtube.com/watch?v=d-Oxs3ynYpA"]),
("Requiem", "Alfred Schnittke", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=SdrJzKTrVaE"]),
("Symphony No.3", "Henryk Górecki", "English", 0.54, 0.48, "Classical", ["https://www.youtube.com/watch?v=ZTmspJxshEg"]),
("Totus Tuus", "Henryk Górecki", "English", 0.66, 0.42, "Classical", ["https://www.youtube.com/watch?v=Gz7s_NJZ7X4"]),
("Beatus Vir", "Henryk Górecki", "English", 0.62, 0.46, "Classical", ["https://www.youtube.com/watch?v=PU3rp94JUc4"]),
("Concerto-Cantata", "Henryk Górecki", "English", 0.6, 0.5, "Classical", ["https://www.youtube.com/watch?v=QBNHTzP0aw0"]),
("Three Pieces in Old Style", "Henryk Górecki", "English", 0.7, 0.56, "Classical", ["https://www.youtube.com/watch?v=Wb73v05adG4"]),
("Symphony No.4", "Witold Lutosławski", "English", 0.68, 0.82, "Classical", ["https://www.youtube.com/watch?v=hIGgmKyK0bc"]),
("Concerto for Orchestra", "Witold Lutosławski", "English", 0.72, 0.86, "Classical", ["https://www.youtube.com/watch?v=-IYw_8xzkWE"]),
("Chain 2", "Witold Lutosławski", "English", 0.66, 0.74, "Classical", ["https://www.youtube.com/watch?v=Vxp9Q4k28q4"]),
("Livre pour orchestre", "Witold Lutosławski", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=v3c0zgGsgkA"]),
("Partita for Violin and Orchestra", "Witold Lutosławski", "English", 0.74, 0.76, "Classical", ["https://www.youtube.com/watch?v=UtKizDvvGL8"]),
("Symphony No.1", "Krzysztof Penderecki", "English", 0.6, 0.78, "Classical", ["https://www.youtube.com/watch?v=RZr4X0ClVo8"]),
("Polish Requiem", "Krzysztof Penderecki", "English", 0.56, 0.66, "Classical", ["https://www.youtube.com/watch?v=0ukgUxGlOz4"]),
("Credo", "Krzysztof Penderecki", "English", 0.62, 0.64, "Classical", ["https://www.youtube.com/watch?v=lipV5O4jqCI"]),
("St. Luke Passion", "Krzysztof Penderecki", "English", 0.54, 0.72, "Classical", ["https://www.youtube.com/watch?v=TvJLftkZTU8"]),
("Symphony No.8 Songs of Transience", "Krzysztof Penderecki", "English", 0.64, 0.6, "Classical", ["https://www.youtube.com/watch?v=Tpxh4K3RWrc"]),
("Symphony No.1", "John Adams", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=5LoUm_r7It8"]),
("Harmonielehre", "John Adams", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=peuuOYs5o38"]),
("Short Ride in a Fast Machine", "John Adams", "English", 0.88, 0.92, "Classical", ["https://www.youtube.com/watch?v=5LoUm_r7It8"]),
("Shaker Loops", "John Adams", "English", 0.78, 0.84, "Classical", ["https://www.youtube.com/watch?v=7twYdU3CgGU"]),
("Doctor Atomic Symphony", "John Adams", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=53EtWPealDE"]),
("Symphony No.1", "Philip Glass", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=5lGNtPXcA04"]),
("Violin Concerto", "Philip Glass", "English", 0.76, 0.74, "Classical", ["https://www.youtube.com/watch?v=5lGNtPXcA04"]),
("The Hours", "Philip Glass", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=Wkof3nPK--Y"]),
("Einstein on the Beach", "Philip Glass", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=YlVrdvCPq20"]),
("Akhnaten", "Philip Glass", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=5PcgXev7VlU"]),
("Sinfonietta", "Leoš Janáček", "English", 0.8, 0.82, "Classical", ["https://www.youtube.com/watch?v=9aFTv50AoEQ"]),
("Taras Bulba", "Leoš Janáček", "English", 0.72, 0.78, "Classical", ["https://www.youtube.com/watch?v=JFz2Hle2YSg"]),
("Glagolitic Mass", "Leoš Janáček", "English", 0.7, 0.74, "Classical", ["https://www.youtube.com/watch?v=JfYqCCgWRek"]),
("Jenůfa Suite", "Leoš Janáček", "English", 0.66, 0.68, "Classical", ["https://www.youtube.com/watch?v=XFyPYsC74eQ"]),
("The Cunning Little Vixen Suite", "Leoš Janáček", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=9kSedgoHpWA"]),
("Symphony No.1", "Karol Szymanowski", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=MYu96EtWS04"]),
("Symphony No.3 Song of the Night", "Karol Szymanowski", "English", 0.72, 0.76, "Classical", ["https://www.youtube.com/watch?v=hgGdW5HEAZY"]),
("Violin Concerto No.1", "Karol Szymanowski", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=uT00G955oUg"]),
("Myths", "Karol Szymanowski", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=U5mUn1AQ4JU"]),
("Stabat Mater", "Karol Szymanowski", "English", 0.6, 0.62, "Classical", ["https://www.youtube.com/watch?v=soC2WFYgWH0"]),
("Symphony No.1", "Arvo Pärt", "English", 0.58, 0.46, "Classical", ["https://www.youtube.com/watch?v=TkdieJXNVhY"]),
("Tabula Rasa", "Arvo Pärt", "English", 0.48, 0.36, "Classical", ["https://www.youtube.com/watch?v=SMN9h85DrFM"]),
("Spiegel im Spiegel", "Arvo Pärt", "English", 0.56, 0.28, "Classical", ["https://www.youtube.com/watch?v=FZe3mXlnfNc"]),
("Cantus in Memory of Benjamin Britten", "Arvo Pärt", "English", 0.42, 0.4, "Classical", ["https://www.youtube.com/watch?v=TVZZgfXNFW8"]),
("Fratres", "Arvo Pärt", "English", 0.52, 0.48, "Classical", ["https://www.youtube.com/watch?v=7PS5QMsGaRw"]),
("On the Nature of Daylight", "Max Richter", "English", 0.58, 0.4, "Classical", ["https://www.youtube.com/watch?v=InyT9Gyoz_o"]),
("November", "Max Richter", "English", 0.62, 0.44, "Classical", ["https://www.youtube.com/watch?v=FPKgk5_YmpA"]),
("Infra 5", "Max Richter", "English", 0.46, 0.38, "Classical", ["https://www.youtube.com/watch?v=9rpvCKLT9Dw"]),
("Dream 3", "Max Richter", "English", 0.66, 0.36, "Classical", ["https://www.youtube.com/watch?v=AwpWZVG5SsQ"]),
("Horizon Variations", "Max Richter", "English", 0.64, 0.42, "Classical", ["https://www.youtube.com/watch?v=xpxKDDgrq_M"]),
("Metamorphosis One", "Philip Glass", "English", 0.7, 0.46, "Classical", ["https://www.youtube.com/watch?v=C2inNYauU1o"]),
("Opening", "Philip Glass", "English", 0.68, 0.52, "Classical", ["https://www.youtube.com/watch?v=-nBE9U7q1Uc"]),
("Glassworks", "Philip Glass", "English", 0.72, 0.48, "Classical", ["https://www.youtube.com/watch?v=-nBE9U7q1Uc"]),
("Etude No. 2", "Philip Glass", "English", 0.6, 0.44, "Classical", ["https://www.youtube.com/watch?v=mS0zt7tZYSE"]),
("Koyaanisqatsi Theme", "Philip Glass", "English", 0.52, 0.54, "Classical", ["https://www.youtube.com/watch?v=OacVy8_nJi0"]),

("Gnossienne No.1", "Erik Satie", "English", 0.56, 0.3, "Classical", ["https://www.youtube.com/watch?v=4Y42P0G-kGY"]),
("Gymnopédie No.2", "Erik Satie", "English", 0.58, 0.32, "Classical", ["https://www.youtube.com/watch?v=IM54ja7p86U"]),
("Gymnopédie No.3", "Erik Satie", "English", 0.6, 0.34, "Classical", ["https://www.youtube.com/watch?v=FS6o3qFimsc"]),
("Ogives", "Erik Satie", "English", 0.52, 0.28, "Classical", ["https://www.youtube.com/watch?v=hI7a0Eh_IPo"]),
("Je te veux", "Erik Satie", "English", 0.66, 0.48, "Classical", ["https://www.youtube.com/watch?v=gs1zusawMGE"]),
("Arabesque No.1", "Claude Debussy", "English", 0.7, 0.46, "Classical", ["https://www.youtube.com/watch?v=uzeLpaPjZNU"]),
("Arabesque No.2", "Claude Debussy", "English", 0.68, 0.48, "Classical", ["https://www.youtube.com/watch?v=uzeLpaPjZNU"]),
("Images", "Claude Debussy", "English", 0.66, 0.42, "Classical", ["https://www.youtube.com/watch?v=S0FIGtKSJ9k"]),
("La fille aux cheveux de lin", "Claude Debussy", "English", 0.72, 0.38, "Classical", ["https://www.youtube.com/watch?v=jGSZPRk6aXA"]),
("Préludes Book II", "Claude Debussy", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=JiYNcAb8-WY"]),
("String Quartet in F Major", "Maurice Ravel", "English", 0.72, 0.6, "Classical", ["https://www.youtube.com/watch?v=ieRQyyPowH0"]),
("Valses nobles et sentimentales", "Maurice Ravel", "English", 0.74, 0.62, "Classical", ["https://www.youtube.com/watch?v=V8kLmFvYso4"]),
("Boléro", "Maurice Ravel", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=Dh9bUD-hC0A"]),
("Alborada del gracioso", "Maurice Ravel", "English", 0.78, 0.7, "Classical", ["https://www.youtube.com/watch?v=kGgEroiMBCY"]),
("Rapsodie espagnole", "Maurice Ravel", "English", 0.76, 0.68, "Classical", ["https://www.youtube.com/watch?v=bbIAPqQcWkQ"]),
("Symphony No.3", "Sergei Prokofiev", "English", 0.7, 0.78, "Classical", ["https://www.youtube.com/watch?v=uSRuFlBo8ys"]),
("Symphony No.5", "Sergei Prokofiev", "English", 0.76, 0.82, "Classical", ["https://www.youtube.com/watch?v=bXxIopzRZco"]),
("Scythian Suite", "Sergei Prokofiev", "English", 0.72, 0.86, "Classical", ["https://www.youtube.com/watch?v=Y4U7wNZu-CU"]),
("Peter and the Wolf", "Sergei Prokofiev", "English", 0.82, 0.66, "Classical", ["https://www.youtube.com/watch?v=px8FakwGPDM"]),
("Cinderella Suite", "Sergei Prokofiev", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=YOV7yWEv54o"]),
("Symphony No.3", "Jean Sibelius", "English", 0.7, 0.68, "Classical", ["https://www.youtube.com/watch?v=2hYaL3mJl1E"]),
("Kullervo", "Jean Sibelius", "English", 0.62, 0.74, "Classical", ["https://www.youtube.com/watch?v=hDzor0VXy0M"]),
("Pohjola's Daughter", "Jean Sibelius", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=DMrHBMTVcF0"]),
("Tapiola", "Jean Sibelius", "English", 0.58, 0.62, "Classical", ["https://www.youtube.com/watch?v=gkX-Uc0e3TQ"]),
("The Oceanides", "Jean Sibelius", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=s5IbPFGPAMQ"]),
("Symphony No.2", "Camille Saint-Saëns", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=_nKlBXJ73EY"]),
("Samson et Dalila", "Camille Saint-Saëns", "English", 0.64, 0.58, "Classical", ["https://www.youtube.com/watch?v=noHQXogDsA0"]),
("Danse Macabre", "Camille Saint-Saëns", "English", 0.5, 0.6, "Classical", ["https://www.youtube.com/watch?v=71fZhMXlGT4"]),
("Carnival of the Animals", "Camille Saint-Saëns", "English", 0.86, 0.66, "Classical", ["https://www.youtube.com/watch?v=UmoZNL-LBKA"]),
("Introduction and Rondo Capriccioso", "Camille Saint-Saëns", "English", 0.8, 0.76, "Classical", ["https://www.youtube.com/watch?v=8UTq1eZrDkI"]),
("Faust Symphony", "Franz Liszt", "English", 0.62, 0.78, "Classical", ["https://www.youtube.com/watch?v=oGvZZYUaOYo"]),
("Dante Symphony", "Franz Liszt", "English", 0.6, 0.8, "Classical", ["https://www.youtube.com/watch?v=agXmxO9xwis"]),
("Les Préludes", "Franz Liszt", "English", 0.76, 0.78, "Classical", ["https://www.youtube.com/watch?v=jb2bkVQwtBs"]),
("Mazeppa", "Franz Liszt", "English", 0.66, 0.84, "Classical", ["https://www.youtube.com/watch?v=hb4H_9TKQ8I"]),
("Hungarian Fantasy", "Franz Liszt", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=3fLbHr0Fz1M"]),
("Symphony No.2 Lobgesang", "Felix Mendelssohn", "English", 0.8, 0.7, "Classical", ["https://www.youtube.com/watch?v=IQ7CBEjf7H0"]),
("Italian Symphony", "Felix Mendelssohn", "English", 0.86, 0.76, "Classical", ["https://www.youtube.com/watch?v=Mya0bUfjG5E"]),
("Scottish Symphony", "Felix Mendelssohn", "English", 0.68, 0.7, "Classical", ["https://www.youtube.com/watch?v=xewrmdNzAuw"]),
("Hebrides Overture", "Felix Mendelssohn", "English", 0.74, 0.64, "Classical", ["https://www.youtube.com/watch?v=zcogD-hHEYs"]),
("Violin Concerto in E Minor", "Felix Mendelssohn", "English", 0.8, 0.74, "Classical", ["https://www.youtube.com/watch?v=SshDX7Oz6Tk"]),
("Symphony No.1 Spring", "Robert Schumann", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=4YpkT_WGVnA"]),
("Symphony No.2", "Robert Schumann", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=YD-dluM2gJQ"]),
("Piano Concerto in A Minor", "Robert Schumann", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=cRNWaTJW-24"]),
("Kinderszenen", "Robert Schumann", "English", 0.74, 0.48, "Classical", ["https://www.youtube.com/watch?v=jnB51JbW2VQ"]),
("Carnaval", "Robert Schumann", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=NMRal1P3NF8"]),
("Symphony No.8", "Antonín Dvořák", "English", 0.84, 0.72, "Classical", ["https://www.youtube.com/watch?v=QXAv-NGppFw"]),
("Symphony No.9 From the New World", "Antonín Dvořák", "English", 0.82, 0.76, "Classical", ["https://www.youtube.com/watch?v=uOmaQSqnPfw"]),
("Slavonic Dances", "Antonín Dvořák", "English", 0.88, 0.78, "Classical", ["https://www.youtube.com/watch?v=Zf-Z9YLz3MQ"]),
("Cello Concerto", "Antonín Dvořák", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=wBFeeOt_SGY"]),
("Serenade for Strings", "Antonín Dvořák", "English", 0.84, 0.6, "Classical", ["https://www.youtube.com/watch?v=CRcbDMg56yg"]),
("Water Music Suite No.2", "George Frideric Handel", "English", 0.84, 0.68, "Classical", ["https://www.youtube.com/watch?v=oRoEo7lZO6s"]),
("Music for the Royal Fireworks", "George Frideric Handel", "English", 0.86, 0.72, "Classical", ["https://www.youtube.com/watch?v=EkttBYzD-jY"]),
("Messiah Overture", "George Frideric Handel", "English", 0.88, 0.7, "Classical", ["https://www.youtube.com/watch?v=0T8qhdlGT0U"]),
("Concerto Grosso Op.6 No.6", "George Frideric Handel", "English", 0.78, 0.64, "Classical", ["https://www.youtube.com/watch?v=CqjXi2O_YQ0"]),
("Sarabande", "George Frideric Handel", "English", 0.7, 0.5, "Classical", ["https://www.youtube.com/watch?v=xOLQd_pUbxs"]),
("Symphony No.88", "Joseph Haydn", "English", 0.86, 0.68, "Classical", ["https://www.youtube.com/watch?v=56qZblncQrs"]),
("Symphony No.92 Oxford", "Joseph Haydn", "English", 0.88, 0.7, "Classical", ["https://www.youtube.com/watch?v=3yNT2NJykZ4"]),
("Symphony No.103 Drumroll", "Joseph Haydn", "English", 0.84, 0.72, "Classical", ["https://www.youtube.com/watch?v=hC9xOwSd-jY"]),
("The Creation Overture", "Joseph Haydn", "English", 0.82, 0.6, "Classical", ["https://www.youtube.com/watch?v=EuIs7R2BpvQ"]),
("Trumpet Concerto", "Joseph Haydn", "English", 0.9, 0.74, "Classical", ["https://www.youtube.com/watch?v=NHjgSiTBddM"]),
("Requiem in D Minor", "Wolfgang Amadeus Mozart", "English", 0.4, 0.5, "Classical", ["https://www.youtube.com/watch?v=XmttZ-BnwaI"]),
("Mass in C Minor", "Wolfgang Amadeus Mozart", "English", 0.62, 0.48, "Classical", ["https://www.youtube.com/watch?v=vhMAhuY4DA8"]),
("Symphony No.31 Paris", "Wolfgang Amadeus Mozart", "English", 0.74, 0.66, "Classical", ["https://www.youtube.com/watch?v=3uj8P9G3OAU"]),
("Symphony No.33", "Wolfgang Amadeus Mozart", "English", 0.7, 0.64, "Classical", ["https://www.youtube.com/watch?v=3gIUIPYgSxk"]),
("Piano Concerto No.20", "Wolfgang Amadeus Mozart", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=FBVITUka_30"]),
("Symphony No.2", "Ludwig van Beethoven", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=bEiYmeeV6sI"]),
("Symphony No.4", "Ludwig van Beethoven", "English", 0.74, 0.72, "Classical", ["https://www.youtube.com/watch?v=vJ1kI5IeJmg"]),
("Piano Sonata No.21 Waldstein", "Ludwig van Beethoven", "English", 0.78, 0.74, "Classical", ["https://www.youtube.com/watch?v=J3l18HTo5rY"]),
("Piano Sonata No.23 Appassionata", "Ludwig van Beethoven", "English", 0.7, 0.86, "Classical", ["https://www.youtube.com/watch?v=efA1S8hyBms"]),
("Fidelio Overture", "Ludwig van Beethoven", "English", 0.68, 0.8, "Classical", ["https://www.youtube.com/watch?v=fq7g6du9S3s"]),
("Symphony No.3", "Johannes Brahms", "English", 0.7, 0.68, "Classical", ["https://www.youtube.com/watch?v=SsTi_40pJ0c"]),
("Academic Festival Overture", "Johannes Brahms", "English", 0.82, 0.74, "Classical", ["https://www.youtube.com/watch?v=AgJhsa-wLNU"]),
("Tragic Overture", "Johannes Brahms", "English", 0.6, 0.72, "Classical", ["https://www.youtube.com/watch?v=PUPVO3cFAWQ"]),
("German Requiem", "Johannes Brahms", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=ZXU9vqVdudM"]),
("Violin Sonata No.1", "Johannes Brahms", "English", 0.66, 0.58, "Classical", ["https://www.youtube.com/watch?v=2Ec_DnPL578"]),
("Symphony No.10", "Gustav Mahler", "English", 0.6, 0.78, "Classical", ["https://www.youtube.com/watch?v=vHyV8noUXC0"]),
("Das Lied von der Erde", "Gustav Mahler", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=5c7LzrI0ELw"]),
("Kindertotenlieder", "Gustav Mahler", "English", 0.52, 0.48, "Classical", ["https://www.youtube.com/watch?v=Sx1fv5q7Wiw"]),
("Symphony No.4", "Gustav Mahler", "English", 0.66, 0.7, "Classical", ["https://www.youtube.com/watch?v=Bj6KLv7kv2Q"]),
("Symphony No.1 Titan", "Gustav Mahler", "English", 0.62, 0.76, "Classical", ["https://www.youtube.com/watch?v=4XbHLFkg_Mw"]),
("Symphony No.6", "Jean Sibelius", "English", 0.7, 0.72, "Classical", ["https://www.youtube.com/watch?v=FChg3ERp6C8"]),
("Symphony No.7", "Jean Sibelius", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=Bi9QiDrJJmw"]),
("Finlandia", "Jean Sibelius", "English", 0.78, 0.8, "Classical", ["https://www.youtube.com/watch?v=fE0RbPsC9uE"]),
("Violin Concerto", "Jean Sibelius", "English", 0.66, 0.72, "Classical", ["https://www.youtube.com/watch?v=J0w0t4Qn6LY"]),
("Karelia Suite", "Jean Sibelius", "English", 0.76, 0.72, "Classical", ["https://www.youtube.com/watch?v=adKwG9ZuzFw"]),
("Pulcinella Suite", "Igor Stravinsky", "English", 0.72, 0.68, "Classical", ["https://www.youtube.com/watch?v=glbnrYF_0ck"]),
("Symphony in Three Movements", "Igor Stravinsky", "English", 0.7, 0.82, "Classical", ["https://www.youtube.com/watch?v=K4defp8nmXI"]),
("The Firebird Suite", "Igor Stravinsky", "English", 0.64, 0.82, "Classical", ["https://www.youtube.com/watch?v=rYcz-g8WpMc"]),
("The Rite of Spring", "Igor Stravinsky", "English", 0.52, 0.92, "Classical", ["https://www.youtube.com/watch?v=gqJdR8_cw18"]),
("Petrushka", "Igor Stravinsky", "English", 0.7, 0.84, "Classical", ["https://www.youtube.com/watch?v=esD90diWZds"]),
("St. Matthew Passion", "Johann Sebastian Bach", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=ZwVW1ttVhuQ"]),
("St. John Passion", "Johann Sebastian Bach", "English", 0.56, 0.62, "Classical", ["https://www.youtube.com/watch?v=zMf9XDQBAaI"]),
("Orchestral Suite No.1", "Johann Sebastian Bach", "English", 0.74, 0.6, "Classical", ["https://www.youtube.com/watch?v=DquhZcSwrrI"]),
("Orchestral Suite No.2", "Johann Sebastian Bach", "English", 0.76, 0.62, "Classical", ["https://www.youtube.com/watch?v=x8Rv9ppP6A8"]),
("Orchestral Suite No.4", "Johann Sebastian Bach", "English", 0.78, 0.66, "Classical", ["https://www.youtube.com/watch?v=uTUEvfdHyL0"]),
("Piano Concerto No.24", "Wolfgang Amadeus Mozart", "English", 0.72, 0.64, "Classical", ["https://www.youtube.com/watch?v=Q05YWVQZTLc"]),
("Symphony No.29", "Wolfgang Amadeus Mozart", "English", 0.7, 0.66, "Classical", ["https://www.youtube.com/watch?v=CpW_B00F6Vk"]),
("Symphony No.38 Prague", "Wolfgang Amadeus Mozart", "English", 0.74, 0.7, "Classical", ["https://www.youtube.com/watch?v=1t18CpBuJJM"]),
("Violin Concerto No.5", "Wolfgang Amadeus Mozart", "English", 0.78, 0.68, "Classical", ["https://www.youtube.com/watch?v=jX6pEd5KAZc"]),
("Piano Sonata No.17", "Wolfgang Amadeus Mozart", "English", 0.72, 0.6, "Classical", ["https://www.youtube.com/watch?v=6wiQE4LS9No"]),
("Symphony No.1", "Ludwig van Beethoven", "English", 0.74, 0.68, "Classical", ["https://www.youtube.com/watch?v=JAGip4nOsOg"]),
("Symphony No.8", "Ludwig van Beethoven", "English", 0.78, 0.72, "Classical", ["https://www.youtube.com/watch?v=gCkndZBnMLw"]),
("Piano Sonata No.12", "Ludwig van Beethoven", "English", 0.7, 0.62, "Classical", ["https://www.youtube.com/watch?v=RTnc9o4o75w"]),
("Piano Sonata No.17 Tempest", "Ludwig van Beethoven", "English", 0.68, 0.74, "Classical", ["https://www.youtube.com/watch?v=6KMGcOYHSs0"]),
("Violin Sonata No.9 Kreutzer", "Ludwig van Beethoven", "English", 0.72, 0.8, "Classical", ["https://www.youtube.com/watch?v=COGcCBJAC6I"]),
("Symphony No.1", "Johannes Brahms", "English", 0.68, 0.72, "Classical", ["https://www.youtube.com/watch?v=-B9nERqEmUA"]),
("Symphony No.3", "Johannes Brahms", "English", 0.7, 0.68, "Classical", ["https://www.youtube.com/watch?v=SsTi_40pJ0c"]),
("Clarinet Quintet", "Johannes Brahms", "English", 0.66, 0.58, "Classical", ["https://www.youtube.com/watch?v=VURcJkx9664"]),
("Piano Quartet No.1", "Johannes Brahms", "English", 0.72, 0.66, "Classical", ["https://www.youtube.com/watch?v=vOJLeHBccyQ"]),
("Violin Sonata No.2", "Johannes Brahms", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=-yH_p0nH1N8"]),
("Symphony No.3", "Gustav Mahler", "English", 0.66, 0.82, "Classical", ["https://www.youtube.com/watch?v=9Yr720ftjaA"]),
("Symphony No.6", "Gustav Mahler", "English", 0.58, 0.86, "Classical", ["https://www.youtube.com/watch?v=YsEo1PsSmbg"]),
("Symphony No.7", "Gustav Mahler", "English", 0.62, 0.82, "Classical", ["https://www.youtube.com/watch?v=c8_CWVAWzPs"]),
("Symphony No.8", "Gustav Mahler", "English", 0.7, 0.88, "Classical", ["https://www.youtube.com/watch?v=gOddn8-35c0"]),
("Das Klagende Lied", "Gustav Mahler", "English", 0.54, 0.7, "Classical", ["https://www.youtube.com/watch?v=ukpldF06G24"]),
("Violin Concerto", "Jean Sibelius", "English", 0.66, 0.72, "Classical", ["https://www.youtube.com/watch?v=J0w0t4Qn6LY"]),
("The Oceanides", "Jean Sibelius", "English", 0.68, 0.6, "Classical", ["https://www.youtube.com/watch?v=s5IbPFGPAMQ"]),
("Tapiola", "Jean Sibelius", "English", 0.58, 0.62, "Classical", ["https://www.youtube.com/watch?v=gkX-Uc0e3TQ"]),
("En Saga", "Jean Sibelius", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=ALU5OaTQNdQ"]),
("Symphony No.2", "Jean Sibelius", "English", 0.72, 0.7, "Classical", ["https://www.youtube.com/watch?v=iXU8EXL7a_4"]),
("Symphony No.10", "Dmitri Shostakovich", "English", 0.62, 0.84, "Classical", ["https://www.youtube.com/watch?v=C2T97GsY0nI"]),
("Symphony No.11 The Year 1905", "Dmitri Shostakovich", "English", 0.58, 0.82, "Classical", ["https://www.youtube.com/watch?v=Lu09CWT41NE"]),
("Symphony No.13 Babi Yar", "Dmitri Shostakovich", "English", 0.54, 0.78, "Classical", ["https://www.youtube.com/watch?v=rc4ul6LcKMQ"]),
("Violin Concerto No.1", "Dmitri Shostakovich", "English", 0.7, 0.76, "Classical", ["https://www.youtube.com/watch?v=Xt4adkgbJs4"]),
("Cello Concerto No.1", "Dmitri Shostakovich", "English", 0.72, 0.8, "Classical", ["https://www.youtube.com/watch?v=tG0laIxC0Lo"]),
("Piano Concerto No.4", "Sergei Prokofiev", "English", 0.74, 0.82, "Classical", ["https://www.youtube.com/watch?v=ccA7BqQBtaE"]),
("Piano Concerto No.5", "Sergei Prokofiev", "English", 0.76, 0.84, "Classical", ["https://www.youtube.com/watch?v=MP66Sw7Q1so"]),
("Symphony No.5", "Sergei Prokofiev", "English", 0.68, 0.78, "Classical", ["https://www.youtube.com/watch?v=bXxIopzRZco"]),
("Symphony No.6", "Sergei Prokofiev", "English", 0.64, 0.8, "Classical", ["https://www.youtube.com/watch?v=5KvT5i8rbZE"]),
("Scythian Suite", "Sergei Prokofiev", "English", 0.7, 0.88, "Classical", ["https://www.youtube.com/watch?v=Y4U7wNZu-CU"]),
("Symphony No.10", "Gustav Mahler", "English", 0.6, 0.76, "Classical", ["https://www.youtube.com/watch?v=vHyV8noUXC0"]),
("Kindertotenlieder", "Gustav Mahler", "English", 0.52, 0.48, "Classical", ["https://www.youtube.com/watch?v=Sx1fv5q7Wiw"]),
("Rückert-Lieder", "Gustav Mahler", "English", 0.58, 0.44, "Classical", ["https://www.youtube.com/watch?v=TzJyIWxjX9o"]),
("Lieder eines fahrenden Gesellen", "Gustav Mahler", "English", 0.6, 0.5, "Classical", ["https://www.youtube.com/watch?v=l8jLcizCDo0"]),
("Das Lied von der Erde", "Gustav Mahler", "English", 0.58, 0.64, "Classical", ["https://www.youtube.com/watch?v=5c7LzrI0ELw"]),
("Violin Concerto No.2", "Béla Bartók", "English", 0.7, 0.82, "Classical", ["https://www.youtube.com/watch?v=-HlqDkv1PzE"]),
("Piano Concerto No.3", "Béla Bartók", "English", 0.72, 0.84, "Classical", ["https://www.youtube.com/watch?v=l7J7L53b8U0"]),
("String Quartet No.5", "Béla Bartók", "English", 0.66, 0.78, "Classical", ["https://www.youtube.com/watch?v=0bessAMWgnQ"]),
("Dance Suite", "Béla Bartók", "English", 0.74, 0.8, "Classical", ["https://www.youtube.com/watch?v=nI1XE5ns3H8"]),
("Miraculous Mandarin Suite", "Béla Bartók", "English", 0.68, 0.86, "Classical", ["https://www.youtube.com/watch?v=b_QyNQs6nIo"]),
("Symphony No.1", "William Walton", "English", 0.68, 0.78, "Classical", ["https://www.youtube.com/watch?v=FWehYkPwjF4"]),
("Violin Concerto", "William Walton", "English", 0.7, 0.74, "Classical", ["https://www.youtube.com/watch?v=_pCl7B4jAwc"]),
("Viola Concerto", "William Walton", "English", 0.64, 0.66, "Classical", ["https://www.youtube.com/watch?v=2xAZPIjvxYg"]),
("Facade Suite No.2", "William Walton", "English", 0.76, 0.7, "Classical", ["https://www.youtube.com/watch?v=79ahguBURn4"]),
("Orb and Sceptre", "William Walton", "English", 0.84, 0.76, "Classical", ["https://www.youtube.com/watch?v=v6qjUdaDE_Q"]),


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

def is_instrumental_track(song_title, artist):

    instrumental_keywords = [
        # generic
        "instrumental",
        "interlude",
        "theme",
        "suite",
        "overture",
        "concerto",
        "symphony",
        "sonata",
        "etude",
        "prelude",
        "nocturne",
        "waltz",
        "orchestral",
        "background score",
        "score",
        "soundtrack",

        # classical forms
        "adagio",
        "allegro",
        "andante",
        "moderato",
        "vivace",
        "largo",

        # jazz/experimental instrumentals
        "solo",
        "improv",
        "improvisation",
        "jam",
        "session",

        # film & ambient music
        "theme music",
        "main title",
        "opening",
        "ending",

        # common dataset patterns
        "violin",
        "piano",
        "orchestra",
        "quartet",
        "trio"
    ]

    title_lower = song_title.lower()

    # 1️⃣ Title-based detection
    if any(word in title_lower for word in instrumental_keywords):
        return True

    # 2️⃣ Genius lyrics check
    try:
        song = genius.search_song(song_title, artist)
        if song and song.lyrics:

            lyrics_lower = song.lyrics.lower()

            instrumental_patterns = [
                "instrumental",
                "[instrumental]",
                "(instrumental)",
                "this song is instrumental"
            ]

            if any(pattern in lyrics_lower for pattern in instrumental_patterns):
                return True

            # Very short lyrics = instrumental
            if len(lyrics_lower.strip()) < 40:
                return True

    except:
        pass

    return False

    

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
        # STEP 1 — Detect instrumental
        if is_instrumental_track(song_title, artist):
            print(f"🎼 Instrumental detected: {song_title}")
            lyrics = "This song is instrumental"

        else:
            # STEP 2 — Fetch lyrics
            lyrics = fetch_lyrics(song_title, artist)

            # ❗ Vocal but lyrics missing → SKIP
            if not lyrics:
                print(f"⚠️ Vocal song but lyrics not found → skipping: {song_title}")
                failed_songs.append(song_title)
                continue

        # STEP 3 — Save lyrics safely
        lyrics_file = os.path.join(lyrics_folder, f"{song_id}.txt")
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(lyrics)

        print(f"📝 Lyrics saved for {song_title}")

        # STEP 4 — Download audio
        audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
        downloaded_path = download_audio(url_list, audio_file, proxy=VPN_PROXY)

        if not downloaded_path:
            failed_songs.append(song_title)
            continue

        # STEP 5 — Add label only for valid songs
        data.append([song_id, language, artist, valence, arousal, genre])

    except Exception as e:
        print(f"❌ Unexpected error processing {song_title}: {e}")
        failed_songs.append(song_title)





# for song_title, artist, language, valence, arousal, genre, url_list in songs_info:

#     song_id = f"{language.lower()}_{safe_filename(song_title.replace(' ', '_'))}"

#     try:
#         # STEP 1 — Detect instrumental
#         if is_instrumental_track(song_title, artist):
#             print(f"🎼 Instrumental detected: {song_title}")
#             lyrics = "This song is instrumental"   # 🔥 placeholder lyrics

#         else:
#             # STEP 2 — Fetch real lyrics
#             lyrics = fetch_lyrics(song_title, artist)

#         if not lyrics:
#             print(f"⚠️ Lyrics not found")



#         lyrics_file = os.path.join(lyrics_folder, f"{song_id}.txt")
#         with open(lyrics_file, "w", encoding="utf-8") as f:
#             f.write(lyrics)
#         print(f"📝 Lyrics saved for {song_title}")

#         # 🔥 STEP 3 — Download audio only if lyrics exist
#         audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
#         downloaded_path = download_audio(url_list, audio_file, proxy=VPN_PROXY)

#         if not downloaded_path:
#             failed_songs.append(song_title)
#             continue

#         # 🔥 STEP 4 — Add ONLY valid multimodal songs
#         data.append([song_id, language, artist, valence, arousal, genre])

#     except Exception as e:
#         print(f"❌ Unexpected error processing {song_title}: {e}")
#         failed_songs.append(song_title)



# data = []
# failed_songs = []

# for song_title, artist, language, valence, arousal, genre, url_list in songs_info:
#     song_id = f"{language.lower()}_{safe_filename(song_title.replace(' ', '_'))}"
#     try:
#         # --- Fetch Lyrics ---
#         lyrics = fetch_lyrics(song_title, artist)
#         if lyrics:
#             lyrics_file = os.path.join(lyrics_folder, f"{song_id}.txt")
#             with open(lyrics_file, "w", encoding="utf-8") as f:
#                 f.write(lyrics)
#             print(f"📝 Lyrics saved for {song_title}")
#         else:
#             print(f"⚠️ Lyrics not found for {song_title}")

#         # --- Download Audio ---
#         audio_file = os.path.join(audio_folder, f"{song_id}.mp3")
#         downloaded_path = download_audio(url_list, audio_file, proxy=VPN_PROXY)
#         if not downloaded_path:
#             failed_songs.append(song_title)
#             continue

#         # --- Add Metadata ---
#         data.append([song_id, language, artist, valence, arousal, genre])

#     except Exception as e:
#         print(f"❌ Unexpected error processing {song_title}: {e}")
#         failed_songs.append(song_title)

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
