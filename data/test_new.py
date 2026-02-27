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

# ("Deep Universe (M6 Remix)", "Klauss Goulart", "English", 3.88, 3.22, "electronic", ["https://www.youtube.com/watch?v=tLZUYcBPmyQ"]),
# ("RA.136 The Orb - 2009.01.08", "The Orb", "English", 4.13, 2.3, "electronic", ["https://www.youtube.com/watch?v=mEerEja7N1o"]),
# ("ResuRection (The Perfecto edit)", "PPK", "English", 3.93, 2.93, "electronic", ["https://www.youtube.com/watch?v=-Ey7yeQIyDM"]),
# ("Orange", "Art of Trance", "English", 4.9, 2.97, "electronic", ["https://www.youtube.com/watch?v=UPF6NW72Ihc"]),
# ("Стереолюбовь", "Инфинити", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=eMRS42Ax75w"]),
# ("Super Jupiter (Remix)", "Ernesto vs. Bastian", "English", 3.89, 3.02, "electronic", ["https://www.youtube.com/watch?v=iip2QciL_xU"]),
# ("Eastern Promise", "Clubroot", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=lap4DGJgUfQ"]),
# ("End This Day", "PIXSID", "English", 4.13, 2.6, "electronic", ["https://www.youtube.com/watch?v=4-vtWVsHudI"]),
# ("Organic Mechanic", "Felixdroid", "English", 5.17, 3.6, "electronic", [""]),
# ("Autocaz", "Ruxpin", "English", 3.37, 2.1, "electronic", ["https://www.youtube.com/watch?v=r_R7Qvt8Hmw"]),
# ("Sunbeam", "Goth-Trad", "English", 3.6, 2.73, "electronic", ["https://www.youtube.com/watch?v=xrKnGkM1nx4"]),
# ("Sharpening the Norm (live)", "Nodens Ictus", "English", 1.88, 2.2, "electronic", ["https://www.youtube.com/watch?v=WdV5JhVfW-w"]),
# ("Garden Of The Hesperides", "Ruxpin", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=1LzeZ3VHJfM"]),
# ("Slow Kids", "Towers of Asia", "English", 4.3, 3.97, "electronic", ["https://www.youtube.com/watch?v=6bUl8rSLtAA"]),
# ("Memorial", "Matters & Dunaway", "English", 4.77, 2.14, "electronic", ["https://www.youtube.com/watch?v=MFeRwMhbzrs"]),
# ("Zebadiah", "Luke Slater", "English", 2.62, 2.0, "electronic", ["https://www.youtube.com/watch?v=xN88U9RhEBs"]),
# ("Rubycon (Part One) (1995 Digital Remaster)", "Tangerine Dream", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=3LXqStt8Cg8"]),
# ("Space Technique (The Nvious Vocal Mix)", "ASC", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=CaZi73hVSXU"]),
# ("Other Data For Illustrative Purposes Only", "Tran Qual", "English", 6.0, 3.74, "electronic", ["https://www.youtube.com/watch?v=KjD3KANH-xc"]),
# ("A Tale of Two Thieves", "Tran Qual", "English", 6.68, 4.42, "electronic", ["https://www.youtube.com/watch?v=xRgqjs0krSg"]),
# ("To The Stars (Adymus Remix)", "Evave", "English", 4.5, 3.23, "electronic", ["https://www.youtube.com/watch?v=0vtf_O_VMtE"]),
# ("Stage One (Solar Stone Remix)", "Space Manoeuvres", "English", 3.49, 2.67, "electronic", ["https://www.youtube.com/watch?v=a7Ck4LNZo-w"]),
# ("Careful With The Aks, Peter - Part I", "Pete Namlook & Klaus Schulze", "English", 2.86, 1.66, "electronic", ["https://www.youtube.com/watch?v=J4zl3fuBlEc"]),
# ("Melodica (original mix)", "Leama", "English", 4.64, 3.37, "electronic", ["https://www.youtube.com/watch?v=ZLACYX7InX4"]),
# ("пасифик", "Zodiac", "English", 3.48, 2.65, "electronic", ["https://www.youtube.com/watch?v=W33mp0TYVv8"]),
# ("I Know What I Saw", "Dj Sid-the Apocalypze", "English", 1.69, 3.06, "electronic", ["https://www.youtube.com/watch?v=U5UEibnGwDA"]),
# ("Moonshine (Ambient Mix)", "Planisphere", "English", 3.6, 2.73, "electronic", ["https://www.youtube.com/watch?v=tU1Xp7D5aDk"]),
# ("A New Dawn (Virtual Vault Remix)", "Steve Forte Rio", "English", 5.07, 3.61, "electronic", ["https://www.youtube.com/watch?v=1iPv5g3BEK8"]),
# ("Space Guitar", "Statica", "English", 4.19, 3.2, "electronic", ["https://www.youtube.com/watch?v=XVxrv16Kv9w"]),
# ("Dark Matters", "Felixdroid", "English", 4.9, 4.07, "electronic", [""]),
# ("Younger Than Today (Part 1)", "Koushik", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=ncpDYrJmU4E"]),
# ("In Search of Contact", "Trestal", "English", 4.53, 2.49, "electronic", ["https://www.youtube.com/watch?v=PO894ZNorlo"]),
# ("Life Ending", "Susperia-Electrica", "English", 5.37, 3.12, "electronic", ["https://www.youtube.com/watch?v=gHXKQhg2_Eg"]),
# ("Namhouse", "Pete Namlook", "English", 4.79, 3.39, "electronic", ["https://www.youtube.com/watch?v=N8m0dS0DJcI"]),
# ("Theme", "Toshack Highway", "English", 3.34, 1.41, "electronic", ["https://www.youtube.com/watch?v=iw90XsMsfvQ"]),
# ("Travelogue", "PUSH", "English", 3.44, 2.7, "electronic", ["https://www.youtube.com/watch?v=VOJD1q88urE"]),
# ("Kometenmelodie 2 (2009 - Remaster)", "Kraftwerk", "English", 3.94, 3.36, "electronic", ["https://www.youtube.com/watch?v=FyzkykVO6wM"]),
# ("Cassiopeya", "Psyfactor", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=lUMc2ZRPEo0"]),
# ("Over The Edge", "Psyfactor", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=_30gtvueIKA"]),
# ("R.U.R", "Telemetrik", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=pTrZDER04sI"]),
# ("Cirrus Station", "Whirloop", "English", 4.62, 3.25, "electronic", ["https://www.youtube.com/watch?v=fIx8s5JFKxg"]),
# ("Time Gate (Minimal Drum Dub)", "Arnej", "English", 4.5, 3.23, "electronic", ["https://www.youtube.com/watch?v=G4SkUX-leuw"]),
# ("The End Starts Today", "Bis", "English", 4.18, 3.95, "electronic", ["https://www.youtube.com/watch?v=CvC6KtSxMdo"]),
# ("Крылья В Бой (Audiovarda Original)", "Катя First", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=_WwwPoFv__Y"]),
# ("What You're Afraid Of", "Bis", "English", 3.6, 2.4, "electronic", ["https://www.youtube.com/watch?v=xH6d6lu5XPA"]),
# ("Heavenly Star (Count Down Remix)", "Genki Rockets", "English", 4.73, 3.38, "electronic", ["https://www.youtube.com/watch?v=QH75ymQHEeA"]),
# ("Within", "Daft Punk", "English", 5.04, 2.8, "electronic", ["https://www.youtube.com/watch?v=cuj__JnGWLg"]),
# ("Etched Headplate", "Burial", "English", 4.88, 4.06, "electronic", ["https://www.youtube.com/watch?v=W_p4bCFMnHg"]),
# ("The Color of the Fire", "Boards of Canada", "English", 4.74, 4.38, "electronic", ["https://www.youtube.com/watch?v=2rU65M-eYSw"]),
# ("Mind, Drips", "Neon Indian", "English", 2.12, 1.34, "electronic", ["https://www.youtube.com/watch?v=uWs4e7oTn9g"]),
# ("Beginners Falafel", "Flying Lotus", "English", 4.3, 2.82, "electronic", ["https://www.youtube.com/watch?v=OO-5rXQR5bI"]),
# ("Sweet Love for Planet Earth", "Fuck Buttons", "English", 6.03, 4.45, "electronic", ["https://www.youtube.com/watch?v=RxVZDxK02QE"]),
# ("GNG BNG", "Flying Lotus", "English", 2.46, 2.67, "electronic", ["https://www.youtube.com/watch?v=wuSiKd9xCEw"]),
# ("Xtatic Truth", "Crystal Fighters", "English", 3.78, 2.56, "electronic", ["https://www.youtube.com/watch?v=zmvOQNBkCxA"]),
# ("Forever Heavy", "Black Moth Super Rainbow", "English", 1.67, 0.99, "electronic", ["https://www.youtube.com/watch?v=AzgfGGuQBCk"]),
# ("Mouth's Cradle", "Björk", "English", 4.85, 4.03, "electronic", ["https://www.youtube.com/watch?v=TwSvmdZ1t_A"]),
# ("Divine Moments of Truth", "Shpongle", "English", 5.31, 4.02, "electronic", ["https://www.youtube.com/watch?v=U3cgNm_f2ow"]),
# ("Kill All Hippies", "Primal Scream", "English", 3.67, 5.04, "electronic", ["https://www.youtube.com/watch?v=E86gWQs-ios"]),
# ("So Insane", "Discovery", "English", 3.71, 3.4, "electronic", ["https://www.youtube.com/watch?v=L8R_DYjCs-M"]),
# ("Supernatural", "AlunaGeorge", "English", 3.71, 3.4, "electronic", ["https://www.youtube.com/watch?v=wecvxQElEU0"]),
# ("There Might Be Coffee", "deadmau5", "English", 3.47, 2.42, "electronic", ["https://www.youtube.com/watch?v=s7kIwISX5fM"]),
# ("No More Mosquitoes", "Four Tet", "English", 2.4, 1.94, "electronic", ["https://www.youtube.com/watch?v=l651f3mucXA"]),
# ("Chameleon", "Trentemøller", "English", 1.81, 1.14, "electronic", ["https://www.youtube.com/watch?v=_r8b9fq-EKA"]),
# ("Florida", "Diplo", "English", 6.22, 3.46, "electronic", ["https://www.youtube.com/watch?v=I5Qn996xjGo"]),
# ("Ancestors", "Gonjasufi", "English", 2.43, 2.63, "electronic", ["https://www.youtube.com/watch?v=m_N63b2Tk-A"]),
# ("Staircase", "Radiohead", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=tFTLxkMmY4M"]),
# ("The Girl With the Sun in Her Head", "Orbital", "English", 6.23, 3.98, "electronic", ["https://www.youtube.com/watch?v=l7mfmouVURY"]),
# ("Drippy Eye", "Black Moth Super Rainbow", "English", 5.88, 4.02, "electronic", ["https://www.youtube.com/watch?v=rCyXSlk8sRM"]),
# ("My Head Feels Like a Frisbee", "Shpongle", "English", 5.66, 4.08, "electronic", ["https://www.youtube.com/watch?v=IWOH4OsiPHQ"]),
# ("Foetus", "Efterklang", "English", 4.51, 3.0, "electronic", ["https://www.youtube.com/watch?v=smWTYzrwHLk"]),
# ("Shakawkaw", "Infected Mushroom", "English", 4.73, 3.39, "electronic", ["https://www.youtube.com/watch?v=P1oaLauCBSs"]),
# ("Shpongleyes", "Shpongle", "English", 4.02, 2.9, "electronic", ["https://www.youtube.com/watch?v=0FUWadixMkg"]),
# ("No Day Massacre", "Mr. Oizo", "English", 2.27, 1.14, "electronic", ["https://www.youtube.com/watch?v=pZcTyWZVtqQ"]),
# ("Disco Mushroom", "Infected Mushroom", "English", 4.55, 3.58, "electronic", ["https://www.youtube.com/watch?v=Q-5QFqFBzMQ"]),
# ("Pet Monster Shotglass", "Flying Lotus", "English", 3.66, 4.91, "electronic", ["https://www.youtube.com/watch?v=QYtmkUXnVhU"]),
# ("El Wraith", "Amon Tobin", "English", 5.19, 3.8, "electronic", ["https://www.youtube.com/watch?v=0fkd2ttXgXU"]),
# ("Circuits of the Imagination", "Shpongle", "English", 3.16, 2.27, "electronic", ["https://www.youtube.com/watch?v=41kDpFYlhUI"]),
# ("Down the Line (It Takes a Number)", "Romare", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=vzYbJQm9laM"]),
# ("Song Pong", "Infected Mushroom", "English", 4.41, 3.36, "electronic", ["https://www.youtube.com/watch?v=3e0DU11VVkc"]),
# ("Davy Jones' Locker", "Panda Bear", "English", 4.65, 3.08, "electronic", ["https://www.youtube.com/watch?v=ZywZsun1W9g"]),
# ("Memory Man", "Van She", "English", 2.0, 1.34, "electronic", ["https://www.youtube.com/watch?v=V1lnvePSESY"]),
# ("Cozza Frenzy", "Bassnectar", "English", 4.16, 3.24, "electronic", ["https://www.youtube.com/watch?v=uFm749jvHK0"]),
# ("Shadow", "Brian Eno", "English", 3.0, 2.89, "electronic", ["https://www.youtube.com/watch?v=aqIPI8CgVc4"]),
# ("Controlled Environment", "Flykkiller", "English", 2.98, 1.88, "electronic", [""]),
# ("Serged", "Mount Kimbie", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=j4bxhoSX6P8"]),
# ("Ineffable Mysteries", "Shpongle", "English", 1.72, 1.47, "electronic", ["https://www.youtube.com/watch?v=G5FDls2byWI"]),
# ("Fail Forever", "When Saints Go Machine", "English", 3.34, 1.41, "electronic", ["https://www.youtube.com/watch?v=T1FG-ws-A8A"]),
# ("Arthur's Birds", "Teebs", "English", 5.6, 3.65, "electronic", ["https://www.youtube.com/watch?v=GMzv7wmiwMc"]),
# ("Ribbons", "Four Tet", "English", 3.81, 2.22, "electronic", ["https://www.youtube.com/watch?v=gkaRO9kiAPQ"]),
# ("UnBirthday", "Pogo", "English", 2.98, 1.95, "electronic", ["https://www.youtube.com/watch?v=Pz4Hf0mf_gw"]),
# ("Goddess", "Chrome Sparks", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=XJ2xZ4lMcOg"]),
# ("The Rapanthem", "Modeselektor", "English", 2.54, 2.04, "electronic", ["https://www.youtube.com/watch?v=BpEKQGq5pqQ"]),
# ("The Parachute Ending", "Birdy Nam Nam", "English", 4.03, 3.37, "electronic", ["https://www.youtube.com/watch?v=3JJsq0GbpPg"]),
# ("Triumph of the Heart", "Björk", "English", 3.59, 2.0, "electronic", ["https://www.youtube.com/watch?v=0z-rhM-dcO8"]),
# ("Ground Luminosity", "Entheogenic", "English", 5.49, 3.56, "electronic", ["https://www.youtube.com/watch?v=ua3pqbWD8E0"]),
# ("Input Out", "Orbital", "English", 2.65, 1.87, "electronic", ["https://www.youtube.com/watch?v=YJe6KCxwNoY"]),
# ("Crusher Dub", "Vex'd", "English", 2.34, 1.7, "electronic", ["https://www.youtube.com/watch?v=icMJMxKa8zI"]),
# ("New Dope", "Funki Porcini", "English", 4.29, 3.84, "electronic", ["https://www.youtube.com/watch?v=5DstSzSvDt4"]),
# ("King", "White Ring", "English", 4.97, 3.62, "electronic", ["https://www.youtube.com/watch?v=P564axwSeBc"]),
# ("(Baby I Don't Know) What you Want", "Jacques Greene", "English", 4.43, 2.51, "electronic", ["https://www.youtube.com/watch?v=lIDhX8cpjtI"]),
# ("Analog Wormz Sequel", "Mr. Oizo", "English", 2.76, 4.59, "electronic", ["https://www.youtube.com/watch?v=RwCWYnyJxoI"]),
# ("I Am the Alphabet", "Black Moth Super Rainbow", "English", 3.41, 1.72, "electronic", ["https://www.youtube.com/watch?v=Eo3e3GCe3G4"]),
# ("Window Licker", "Aphex Twin", "English", 2.82, 2.5, "electronic", ["https://www.youtube.com/watch?v=UBS4Gi1y_nc"]),
# ("La Demeure", "Stereolab", "English", 5.53, 3.46, "electronic", ["https://www.youtube.com/watch?v=ydVzw_4SvNI"]),
# ("Mind", "Com Truise", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=Q7cUI0KgSPc"]),
# ("Impossible and Overwhelming", "Bassnectar", "English", 2.21, 1.26, "electronic", ["https://www.youtube.com/watch?v=RzeEvtHL24I"]),
# ("O.O.B.E.", "The Orb", "English", 5.58, 3.19, "electronic", ["https://www.youtube.com/watch?v=UEbyC_hQkhQ"]),
# ("Breathe . Something / Stellar STar", "Flying Lotus", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=NetSZ-lKVco"]),
# ("1983 (Daedelus's Odd-Dance Party remix)", "Flying Lotus", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=wZSTxFuxBq4"]),
# ("Responsible Stu", "The Octopus Project", "English", 3.41, 2.44, "electronic", ["https://www.youtube.com/watch?v=3VOk3akD5gs"]),
# ("Boatfriend", "Black Moth Super Rainbow", "English", 3.91, 3.16, "electronic", ["https://www.youtube.com/watch?v=M8PvCOkd6fA"]),
# ("Walking Wounded (Omni Trio remix)", "Everything But the Girl", "English", 3.74, 3.08, "electronic", ["https://www.youtube.com/watch?v=e6QVgq8eaqY"]),
# ("Absolute Ego Dance", "Yellow Magic Orchestra", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=DZoA4pZPO5o"]),
# ("UT1 - Dot", "Polygon Window", "English", 2.62, 2.0, "electronic", ["https://www.youtube.com/watch?v=oyviVEgIu68"]),
# ("Lost in the Chrome Forest", "Chrome Sparks", "English", 2.58, 1.62, "electronic", ["https://www.youtube.com/watch?v=_6N1-L3wYQ0"]),
# ("Count Backwards to Black", "Black Moth Super Rainbow", "English", 2.7, 1.79, "electronic", ["https://www.youtube.com/watch?v=IZM3Mm6O_Ok"]),
# ("We've Got Commodity", "Dabrye", "English", 3.34, 1.41, "electronic", ["https://www.youtube.com/watch?v=hMlp_SuXVic"]),
# ("Elephant Machine", "Younger Brother", "English", 1.81, 1.38, "electronic", ["https://www.youtube.com/watch?v=UA3hZp90opY"]),
# ("Simoon", "Yellow Magic Orchestra", "English", 4.76, 2.42, "electronic", ["https://www.youtube.com/watch?v=01rpPco3L8A"]),
# ("Cyan", "Monolake", "English", 3.88, 2.07, "electronic", ["https://www.youtube.com/watch?v=7cwaEWVat2Y"]),
# ("Osho", "The Future Sound of London", "English", 4.76, 2.76, "electronic", ["https://www.youtube.com/watch?v=qJXhXs4EyYw"]),
# ("The Horrible Fanfare / Landslide / Exoskeleton", "Beck", "English", 2.69, 3.06, "electronic", ["https://www.youtube.com/watch?v=UsrXhVfUiBM"]),
# ("Going Nowhere (Whitey Remix)", "Cut Copy", "English", 5.89, 4.0, "electronic", ["https://www.youtube.com/watch?v=jrNrdGwMSJM"]),
# ("Perimeters", "Aes Dana", "English", 3.2, 1.92, "electronic", ["https://www.youtube.com/watch?v=Ia3sZ9S8TAM"]),
# ("We are one", "Entheogenic", "English", 5.87, 3.89, "electronic", ["https://www.youtube.com/watch?v=OnETqooAWm4"]),
# ("German Bigflies", "port-royal", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=UK-nMOr2Whg"]),
# ("Woozy", "Faithless", "English", 5.16, 4.04, "electronic", ["https://www.youtube.com/watch?v=j8m68wL9hYw"]),
# ("Lantau", "Monolake", "English", 3.17, 1.74, "electronic", ["https://www.youtube.com/watch?v=UBFFpLPNE-Y"]),
# ("Molten Love", "The Orb", "English", 5.1, 3.16, "electronic", ["https://www.youtube.com/watch?v=bFTyeyyyAhA"]),
# ("Papua New Guinea (12 Version)", "The Future Sound of London", "English", 3.44, 2.17, "electronic", ["https://www.youtube.com/watch?v=wfWMv8Y1V5E"]),
# ("Bib", "Mouse on Mars", "English", 3.41, 1.72, "electronic", ["https://www.youtube.com/watch?v=xYuyFeL0s8g"]),
# ("Fatal Beating", "Hybrid", "English", 4.32, 3.29, "electronic", ["https://www.youtube.com/watch?v=p3ULViczj-E"]),
# ("Jump Into My Mouth And Breath the Stardust", "Black Moth Super Rainbow", "English", 4.82, 2.32, "electronic", ["https://www.youtube.com/watch?v=uktBnMEjA6w"]),
# ("None In Mind", "Koushik", "English", 3.54, 3.0, "electronic", ["https://www.youtube.com/watch?v=NT7F0BsWNzA"]),
# ("Tranny Ride", "Bongripper", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=17-uU4Q3CNs"]),
# ("Level One", "Entheogenic", "English", 4.31, 2.99, "electronic", ["https://www.youtube.com/watch?v=2yDvfUqO8Qc"]),
# ("O.K.", "Fluke", "English", 4.33, 3.03, "electronic", ["https://www.youtube.com/watch?v=_phUXD0xKN8"]),
# ("Song for Isabelle", "Stimming", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=kQHz0QFuMyI"]),
# ("One for the Crime Scene, a Bullet for Your Time", "Prefuse 73", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=JJ0l_J8MLuw"]),
# ("The Dark Forest Joggers", "Black Moth Super Rainbow", "English", 3.21, 2.17, "electronic", ["https://www.youtube.com/watch?v=4HjjvPtnyrE"]),
# ("The Mello Hippo Disco Show", "The Future Sound of London", "English", 4.94, 2.78, "electronic", ["https://www.youtube.com/watch?v=Ll3_KlXMC4U"]),
# ("Lotus", "Faithless", "English", 5.21, 3.82, "electronic", ["https://www.youtube.com/watch?v=oCuadpII_24"]),
# ("Gamma Goblins Part 2", "Hallucinogen", "English", 3.05, 3.33, "electronic", ["https://www.youtube.com/watch?v=k4OPhCA7AJI"]),
# ("Beyond Zero", "Entheogenic", "English", 3.86, 2.56, "electronic", ["https://www.youtube.com/watch?v=t0ye45H0qhY"]),
# ("Drink Malk", "Ochre", "English", 2.58, 2.67, "electronic", ["https://www.youtube.com/watch?v=Zue3HCGziug"]),
# ("One More Time/Aerodynamic (Live)", "Daft Punk", "English", 3.79, 3.05, "electronic", ["https://www.youtube.com/watch?v=Hjqs8YIB7ys"]),
# ("True Believer", "Avicii", "English", 8.02, 5.33, "electronic", ["https://www.youtube.com/watch?v=MhTJJ1AHzzU"]),
# ("I Wasn't Made For Fighting", "Woodhands", "English", 5.47, 5.2, "electronic", ["https://www.youtube.com/watch?v=D8TSjYDAb0w"]),
# ("Soldier Girl Remix (Polyphonic Spree)", "RJD2", "English", 6.61, 5.61, "electronic", ["https://www.youtube.com/watch?v=7lO4jCudsps"]),
# ("The House Detective", "Stevie King", "English", 5.2, 4.44, "electronic", ["https://www.youtube.com/watch?v=J5Xtu29ORbo"]),
# ("Ocotillo Nights", "Moonjabber", "English", 6.43, 4.4, "electronic", [""]),
# ("I Was Made For Loving You (Remix)", "Queen of Japan", "English", 7.27, 4.95, "electronic", ["https://www.youtube.com/watch?v=cYAXu_LM7vw"]),
# ("FURIOUS AND FAST - / AUDIOJUNGLE / ROYAL...", "twisterium", "English", 5.33, 5.56, "electronic", ["https://www.youtube.com/watch?v=XYBkruLZOTw"]),
# ("Pluto", "Björk", "English", 5.88, 5.42, "electronic", ["https://www.youtube.com/watch?v=GmwZdlUSerU"]),
# ("Follow", "Crystal Fighters", "English", 7.53, 5.53, "electronic", ["https://www.youtube.com/watch?v=d9n7DMqbwgU"]),
# ("Being Bad Feels Pretty Good", "Does It Offend You, Yeah?", "English", 6.59, 4.95, "electronic", ["https://www.youtube.com/watch?v=SFEQUsn1oNI"]),
# ("Tenderoni", "Kele", "English", 7.01, 4.93, "electronic", ["https://www.youtube.com/watch?v=bdQioZHYpvQ"]),
# ("Madder", "Groove Armada", "English", 5.25, 5.27, "electronic", ["https://www.youtube.com/watch?v=vEYG9jGztIc"]),
# ("Blood On Our Hands (Justice Remix)", "Death from Above 1979", "English", 7.57, 6.23, "electronic", ["https://www.youtube.com/watch?v=pZNQNELfd0g"]),
# ("Superheroes", "You Love Her Coz She's Dead", "English", 6.97, 5.78, "electronic", ["https://www.youtube.com/watch?v=mhuP_RiLwIg"]),
# ("Nothing Really Matters (feat. will.i.am)", "David Guetta", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=vbGF44nsD7o"]),
# ("Down on Life", "Elliphant", "English", 6.37, 4.67, "electronic", ["https://www.youtube.com/watch?v=jrolhm3CES0"]),
# ("Restless", "Evil Nine", "English", 6.17, 5.03, "electronic", ["https://www.youtube.com/watch?v=LfcZlRYj2IE"]),
# ("Gangster Trippin'", "Fatboy Slim", "English", 6.54, 4.51, "electronic", ["https://www.youtube.com/watch?v=3k1comdW1Ig"]),
# ("Ratio Shmatio", "Infected Mushroom", "English", 6.05, 4.62, "electronic", ["https://www.youtube.com/watch?v=R3z5h7WoKr4"]),
# ("Horus The Chorus", "Infected Mushroom", "English", 5.55, 4.42, "electronic", ["https://www.youtube.com/watch?v=TtWjLaBtMFQ"]),
# ("Express Yourself (Feat. Nicky Da B)", "Diplo", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=BKaL7WL-onI"]),
# ("Silly Boy (Richard Vission RMX)", "Eva Simons", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=pVqg6eOjikk"]),
# ("Give It Up", "Datarock", "English", 5.99, 5.91, "electronic", ["https://www.youtube.com/watch?v=SmNR21tY-VM"]),
# ("Masters Of The Universe", "Juno Reactor", "English", 5.39, 4.15, "electronic", ["https://www.youtube.com/watch?v=7lV36CrVWJ8"]),
# ("Romborama (feat. All Leather)", "The Bloody Beetroots", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=_MK-IEc7uyY"]),
# ("Believe (feat Kele Okereke)", "The Chemical Brothers", "English", 6.09, 5.38, "electronic", ["https://www.youtube.com/watch?v=7f2wg1pqQDs"]),
# ("Down River", "M.I.A.", "English", 7.0, 5.22, "electronic", ["https://www.youtube.com/watch?v=MPsaIE6w1BQ"]),
# ("Bad Karma", "Axel Thesleff", "English", 6.33, 5.09, "electronic", ["https://www.youtube.com/watch?v=-sNWKbnaFkg"]),
# ("Insomnia (Monster Mix)", "Faithless", "English", 5.52, 4.28, "electronic", ["https://www.youtube.com/watch?v=P8JEm4d6Wu4"]),
# ("Escape Me", "Tiësto", "English", 5.49, 4.22, "electronic", ["https://www.youtube.com/watch?v=xOzhgu0fb4M"]),
# ("The Mojo Radio Gang (radio version)", "Parov Stelar", "English", 8.08, 5.89, "electronic", ["https://www.youtube.com/watch?v=OydP-Vb1NQE"]),
# ("Lions!", "Lights", "English", 7.88, 5.42, "electronic", ["https://www.youtube.com/watch?v=N9vgEtKEOdU"]),
# ("Shame On Me", "Amanda Blank", "English", 7.5, 6.45, "electronic", ["https://www.youtube.com/watch?v=UXSb-EVo3UY"]),
# ("Nightflight to Uranus", "Datarock", "English", 7.38, 5.4, "electronic", ["https://www.youtube.com/watch?v=ta0FQ8jaKMU"]),
# ("Shame of the Nation", "New Order", "English", 7.52, 5.67, "electronic", ["https://www.youtube.com/watch?v=WAL84JNBWWg"]),
# ("Gossip", "Breathe Carolina", "English", 7.4, 5.11, "electronic", ["https://www.youtube.com/watch?v=yg7aW07L6xU"]),
# ("Pleasures of Soho", "Sohodolls", "English", 6.86, 5.93, "electronic", ["https://www.youtube.com/watch?v=6XdPDkBriW0"]),
# ("Clash", "Junkie XL", "English", 7.26, 5.36, "electronic", ["https://www.youtube.com/watch?v=B0DrXxVg4OU"]),
# ("I Won't Let You Down", "Audio Bullys", "English", 7.68, 6.11, "electronic", ["https://www.youtube.com/watch?v=ia2BkiTBiEg"]),
# ("Segertåget", "Maskinen", "English", 6.49, 4.54, "electronic", ["https://www.youtube.com/watch?v=ndXCwPbg2Ag"]),
# ("Where's Your Head At?", "Basement Jaxx", "English", 7.84, 6.18, "electronic", ["https://www.youtube.com/watch?v=5rAOyh7YmEc"]),
# ("Bounce (feat. Kelis)", "Calvin Harris", "English", 7.64, 6.08, "electronic", ["https://www.youtube.com/watch?v=ooZwmeUfuXg"]),
# ("It's the Way You Love Me (feat. Kelly Rowland)", "David Guetta", "English", 7.66, 6.09, "electronic", ["https://www.youtube.com/watch?v=9h20PHSKWr8"]),
# ("Soulbitch", "Yonderboi", "English", 6.37, 4.67, "electronic", ["https://www.youtube.com/watch?v=9hPv77VkYwE"]),
# ("Kimosabe", "BT", "English", 5.16, 3.99, "electronic", ["https://www.youtube.com/watch?v=llnLnP5LxJc"]),
# ("Chelsea", "Stefy", "English", 7.71, 6.13, "electronic", ["https://www.youtube.com/watch?v=N1RMCuIUB9c"]),
# ("Bloscid", "Boxcutter", "English", 6.06, 5.03, "electronic", ["https://www.youtube.com/watch?v=bq6TZaEJS20"]),
# ("Slipping Away (Axwell Vocal Mix)", "Moby", "English", 6.59, 5.24, "electronic", ["https://www.youtube.com/watch?v=6w7dpp_SAuc"]),
# ("End of Line (Boys Noize remix)", "Daft Punk", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=ui63KWIooq8"]),
# ("This Boy's In Love (Lifelike Remix)", "The Presets", "English", 4.85, 3.56, "electronic", ["https://www.youtube.com/watch?v=3tl-pBQXJR4"]),
# ("Young Offender", "New Order", "English", 5.36, 5.6, "electronic", ["https://www.youtube.com/watch?v=FtA5MpA6pqo"]),
# ("'68 aka Only Time", "Lemon Jelly", "English", 6.37, 4.67, "electronic", ["https://www.youtube.com/watch?v=5cf_JFo-TYI"]),
# ("Move (You Make Me Feel So Good)", "Moby", "English", 6.26, 5.17, "electronic", ["https://www.youtube.com/watch?v=nc-o3elBPWc"]),
# ("Keine Melodien", "Jeans Team", "English", 7.05, 5.7, "electronic", ["https://www.youtube.com/watch?v=dhacVz343VE"]),
# ("Feeling So Real (live in London)", "Moby", "English", 5.77, 4.48, "electronic", ["https://www.youtube.com/watch?v=C0eN4Q80-ho"]),
# ("The Smell of Today Is Sweet Like Breastmilk in the Wind", "múm", "English", 7.57, 6.1, "electronic", ["https://www.youtube.com/watch?v=Nxj1gNpmf5c"]),
# ("Lassitude", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=SMny1jIfKU4"]),
# ("Puppy", "Netsky", "English", 8.3, 6.35, "electronic", ["https://www.youtube.com/watch?v=FU4cnelEdi4"]),
# ("Hypercaine", "DJ Fresh", "English", 7.57, 5.88, "electronic", ["https://www.youtube.com/watch?v=HKEzldCVPK4"]),
# ("Hot Right Now - Radio Edit", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=N7OPZOBJZyI"]),
# ("Dem Na Like Me", "The Qemists", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=x47buObuTxQ"]),
# ("Do You Hear?", "Mr. Scruff", "English", 6.6, 4.34, "electronic", ["https://www.youtube.com/watch?v=1lnU0z_Zysc"]),
# ("Hot Right Now", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=N7OPZOBJZyI"]),
# ("Gravity - Radio Edit", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=ujlPY31ZAJQ"]),
# ("Three Little Birdies Down Beat", "The Chemical Brothers", "English", 6.02, 4.8, "electronic", ["https://www.youtube.com/watch?v=kJqSFi0ukEY"]),
# ("Compression", "Everything But the Girl", "English", 7.38, 5.19, "electronic", ["https://www.youtube.com/watch?v=O6mZyFbkyQ0"]),
# ("Funk Academy", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=RlK9dalm5JQ"]),
# ("Kofea", "Agoria", "English", 6.95, 5.24, "electronic", ["https://www.youtube.com/watch?v=-I5Uy-zt-XE"]),
# ("Verbal (Kid 606 Dancehall Devastation mix)", "Amon Tobin", "English", 5.63, 5.79, "electronic", [""]),
# ("Hule Lam", "Juno Reactor", "English", 5.22, 6.79, "electronic", ["https://www.youtube.com/watch?v=aY2y6g76L4k"]),
# ("Pyramid", "Photek", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=0ITmq_KS9aM"]),
# ("Peter Street", "Engineers", "English", 8.21, 6.5, "electronic", ["https://www.youtube.com/watch?v=Gs6UbV1cGto"]),
# ("Memoria (Sutekh's Trisagion mix)", "Murcof", "English", 7.07, 5.35, "electronic", ["https://www.youtube.com/watch?v=QVgIuhytAzU"]),
# ("Sciew Spoc", "Gescom", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=A5qddHi4eqI"]),
# ("Like Clockwork", "Spor", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=IVZ5_GY2utg"]),
# ("Different Strokes", "State Of Mind", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=KU0D80dQgew"]),
# ("HEAVEN ON EARTH", "中島美嘉", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=75ySR7O_W5A"]),
# ("Arms House", "Spor", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=atApsdrMUUw"]),
# ("Always Right Never Left", "Spor", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=MZwywCJAc7A"]),
# ("Madness (I Prefer This Mix)", "Bart Claessen & Dave Schiemann", "English", 5.8, 4.39, "electronic", ["https://www.youtube.com/watch?v=2KPFDSXu5Wc"]),
# ("Show Me Love (Extended Mix)", "Robin S", "English", 6.91, 6.26, "electronic", ["https://www.youtube.com/watch?v=Ps2Jc28tQrw"]),
# ("Freak Me", "Total Science", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=dshNDJM-BAs"]),
# ("Swan Lake", "Total Science", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=rYSGiPKiOm0"]),
# ("The Hiding Place", "Psyche", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=X9ia3Kb9X9c"]),
# ("翡翠 -HISUI-", "川田まみ", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=FNl1ud7KxtI"]),
# ("Gravity (feat. Ella Eyre)", "DJ Fresh", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=ujlPY31ZAJQ"]),
# ("CEI3C - Sentient", "Dieselboy", "English", 7.95, 6.95, "electronic", [""]),
# ("Candy Man (Instrumental)", "Brown Eyed Girls", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=UIsSHydUw78"]),
# ("Hold you Colour", "Pendulum", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=7UIJYWSZXPI"]),
# ("Street Pilot", "Human Blue", "English", 6.59, 5.47, "electronic", ["https://www.youtube.com/watch?v=e9XVLpDJrmA"]),
# ("The Random Hustle", "Anthony Shakir", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=23XgqjYj4DY"]),
# ("To Victory [Philip Steir's Sacrifice For Sparta Remix]", "Tyler Bates", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=6XKfwKphCzQ"]),
# ("Turmoil", "Xenoc", "English", 5.03, 5.22, "electronic", ["https://www.youtube.com/watch?v=BZsKsrLQuUg"]),
# ("Digital Prayer", "Neology", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=fwWNLPHn6lI"]),
# ("Bad Sandwich", "Hello Potato!", "English", 3.98, 3.48, "electronic", ["https://www.youtube.com/watch?v=l8N8yNJFA-w"]),
# ("Vasne & Naivu", "KX", "English", 5.44, 6.54, "electronic", ["https://www.youtube.com/watch?v=UGbxs2I2tjo"]),
# ("Plimsoul VIP", "Facs & B-Key", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=Gg_cn7rYM-U"]),
# ("رسالة ● •", "0lд", "English", 5.78, 5.56, "electronic", [""]),
# ("Cheezy as Cheez", "Deporitaz", "English", 7.96, 6.46, "electronic", ["https://www.youtube.com/watch?v=GqnQVQUOnNA"]),
# ("The Ending", "Graham Gold", "English", 7.38, 5.19, "electronic", ["https://www.youtube.com/watch?v=0vIEkzT_HrE"]),
# ("You & I (DJ Amaya's Angel at Dawn Remix)", "박봄", "English", 6.92, 5.48, "electronic", ["https://www.youtube.com/watch?v=VZadsYJ1KIs"]),
# ("Die fabelhafte Welt der Männer", "Galli", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=CY7i38I-I-4"]),
# ("۞", "0lд", "English", 5.78, 5.56, "electronic", [""]),
# ("The Scepter4", "遠藤幹雄", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=tTou7x48TB0"]),
# ("ÇΣῥhΣŪŠ", "0lд", "English", 5.78, 5.56, "electronic", [""]),
# ("ℒĨℒ ℑØ₦ / ℛᗩᏔ", "0lд", "English", 5.78, 5.56, "electronic", [""]),
# ("Nanu", "Halima Ahkdar", "English", 8.26, 6.44, "electronic", [""]),
# ("Stop The Clock", "Lisa Abbot, Dougal & Gammer", "English", 7.21, 6.18, "electronic", ["https://www.youtube.com/watch?v=gb6XPGcAYew"]),
# ("Act 4: Pt.1 - Resolution", "Joe Dulay", "English", 5.6, 4.94, "electronic", [""]),
# ("Act 3:  Empower (Darkest before Dawn)", "Joe Dulay", "English", 5.47, 4.96, "electronic", [""]),
# ("This Truth (An Original Piece)", "Joe Dulay", "English", 5.47, 4.96, "electronic", ["https://www.youtube.com/watch?v=7KOulpyzpA4"]),
# ("Web of Trust", "Zhang Li", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=zSoyn87TcNI"]),
# ("This Truth (Piano Solo)", "Joe Dulay", "English", 5.55, 5.01, "electronic", ["https://www.youtube.com/watch?v=7KOulpyzpA4"]),
# ("╇®Ap ►", "0lд", "English", 5.78, 5.56, "electronic", [""]),
# ("Nacht", "Phorsicht", "English", 6.26, 5.96, "electronic", ["https://www.youtube.com/watch?v=sEOYKJau0Qg"]),
# ("anORGANik", "Phorsicht", "English", 6.96, 5.92, "electronic", ["https://www.youtube.com/watch?v=nlVCtZmuG1w"]),
# ("Resolution Pt. 1 SFX", "Joe Dulay", "English", 5.33, 5.51, "electronic", [""]),
# ("Inspire", "Marco", "English", 7.95, 6.3, "electronic", ["https://www.youtube.com/watch?v=2sw4dXM1KnY"]),
# ("Miss Tortilla De Patata", "Burbuja", "English", 7.95, 6.95, "electronic", ["https://www.youtube.com/watch?v=v8yk_-9vt6o"]),

# ("A Day In The Life Of Klaus", "Adam R. Beebe", "English", 7.95, 6.95, "electronic", [""]),
# ("Shivers (M.O.R.P.H. Red Light Dub)", "Armin van Buuren", "English", 6.17, 4.97, "electronic", ["https://www.youtube.com/watch?v=j8ztkxEu4qQ"]),
# ("Perpetual Motion Machine #6", "Adam R. Beebe", "English", 7.66, 5.58, "electronic", [""]),
# ("Do [ParaMaterial Deep Mix]", "Wan Light", "English", 6.18, 4.01, "electronic", [""]),
# ("Lucky Pressure", "Roni Size and Reprazent", "English", 7.7, 6.23, "electronic", ["https://www.youtube.com/watch?v=zkWBrbAyNCU"]),
# ("Aadime", "Juno Reactor", "English", 5.28, 5.01, "electronic", ["https://www.youtube.com/watch?v=M9YGUkKphsg"]),
# ("Homogen", "Justus Köhncke", "English", 4.98, 4.01, "electronic", ["https://www.youtube.com/watch?v=rrnCLFgnQmM"]),
# ("Take That", "Jürgen Paape", "English", 3.79, 3.05, "electronic", ["https://www.youtube.com/watch?v=uxsynR8nZqs"]),
# ("Romantic Rights (The Phone Lovers Remix) (album version)", "Death from Above 1979", "English", 4.5, 2.08, "electronic", ["https://www.youtube.com/watch?v=6Wnl9PpnTXI"]),
# ("I Can't Loose", "The Phantom's Revenge", "English", 5.75, 4.21, "electronic", ["https://www.youtube.com/watch?v=v9pieCBgnIQ"]),
# ("Dread-O", "eucalyptus.", "English", 6.66, 5.1, "electronic", ["https://www.youtube.com/watch?v=H0loSZFsyoQ"]),
# ("with a heavy heart(i regret to inform you)", "Does It Offend You, Yeah?", "English", 6.05, 4.5, "electronic", ["https://www.youtube.com/watch?v=GmPTAKOi82Q"]),
# ("Circuit Breaker", "Mark Ronson & The Business Intl", "English", 5.11, 4.95, "electronic", ["https://www.youtube.com/watch?v=5NM4tyu82c0"]),
# ("Freak (feat. Steve Bays)", "Steve Aoki, Diplo & Deorro", "English", 6.74, 5.63, "electronic", ["https://www.youtube.com/watch?v=-X6qF7sF9eo"]),
# ("Bullitproof", "Breakbeat Era", "English", 5.11, 4.95, "electronic", ["https://www.youtube.com/watch?v=dSD0aVQkcwo"]),
# ("Kinetic", "Golden Girls", "English", 6.12, 4.77, "electronic", ["https://www.youtube.com/watch?v=KvmY2x0w8PE"]),
# ("8-Bit Felon (Original Mix) 120BPM", "Groovebox", "English", 4.29, 3.1, "electronic", [""]),
# ("Left Behind", "Cansei de Ser Sexy", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=vIhJC2-UNZA"]),
# ("As Stars We Belong", "Thermostatic", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=dp-cFS0fhcc"]),
# ("Commodore C=64 (C=64)", "Welle:Erdball", "English", 7.37, 6.33, "electronic", ["https://www.youtube.com/watch?v=RPgm6XCokNw"]),
# ("Holding Hands", "Ryan Farish", "English", 4.91, 3.05, "electronic", ["https://www.youtube.com/watch?v=l4yfvsjkWIY"]),
# ("Zipper", "Tujiko Noriko", "English", 6.13, 4.8, "electronic", ["https://www.youtube.com/watch?v=nIrZmcNuJU8"]),
# ("Sun Gate", "Tangerine Dream", "English", 7.1, 5.41, "electronic", ["https://www.youtube.com/watch?v=jrqK7-uEMS8"]),
# ("Bionic Commando (Tune 5)", "Romeo Knight", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=gkbF231kk6I"]),
# ("Shining (Koki Dub)", "Kiko Navarro", "English", 6.18, 5.05, "electronic", ["https://www.youtube.com/watch?v=pKk64NV3v18"]),
# ("Green Grass", "Kin", "English", 6.54, 4.13, "electronic", ["https://www.youtube.com/watch?v=CtuFaOqiots"]),
# ("My Spine (Feat. Evelyn Glennie)", "Björk", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=UUeECOEJIP8"]),
# ("Horny Mutant Jazz", "T Power vs MK Ultra", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=ufRj7KuPcXM"]),
# ("North Pole Express", "Jon Schmidt", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=rUgYGhil2zg"]),
# ("Carnival (Clutter's Bombay Mix)", "Adrian Carter", "English", 4.65, 3.49, "electronic", [""]),
# ("Tout Est Bleu (original mix)", "Jean Michel Jarre", "English", 7.79, 6.08, "electronic", ["https://www.youtube.com/watch?v=6xOUYcBxkYU"]),
# ("Last Ninja - Palace Gardens Ld", "Tonka", "English", 7.34, 6.1, "electronic", ["https://www.youtube.com/watch?v=np1bVvu6a3g"]),
# ("In celebration of physical education teachers", "Anym3n", "English", 6.99, 5.11, "electronic", [""]),
# ("The Great Giana Sisters", "Dr.Future", "English", 7.79, 6.08, "electronic", ["https://www.youtube.com/watch?v=XKkmyMKbPKE"]),
# ("Tout Est Bleu (Eiffel 65 parade mix)", "Jean Michel Jarre", "English", 7.35, 5.85, "electronic", ["https://www.youtube.com/watch?v=oWRZH-NYPPg"]),
# ("Lightforce 2000", "Chris Abbott", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=sHCQ4HKIcuI"]),
# ("Dann werden wir uns wiedersehen (Rückkehr nach Frankfurt)", "psalmenundlieder", "English", 7.12, 6.1, "electronic", [""]),
# ("Summer Lovin", "Daniel Washburn", "English", 6.18, 5.05, "electronic", ["https://www.youtube.com/watch?v=tQ0yjYUFKAE"]),
# ("Undoubted", "Unmorph", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=jaF8cewBogo"]),
# ("Moonlight Densetsu", "ももいろクローバーZ", "English", 7.25, 4.51, "electronic", ["https://www.youtube.com/watch?v=LOYctgvp75s"]),
# ("Candor Original Cue Mix", "Cue Dj", "English", 7.34, 6.1, "electronic", ["https://www.youtube.com/watch?v=rzg8OPPKVdM"]),
# ("What's Left [Refuel]", "Unmorph", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=7Lxvc0e-Rvk"]),
# ("Energize", "DJ AJA Inc.", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=n2bBQY-TTLo"]),
# ("Wake Up To A Sunset", "Unmorph", "English", 6.94, 5.38, "electronic", ["https://www.youtube.com/watch?v=fnwMyn5hQdw"]),
# ("04 - Gert Emmens - Dawn", "Gert Emmens", "English", 7.12, 6.1, "electronic", ["https://www.youtube.com/watch?v=VHSUsqD2xjM"]),
# ("Precipice", "Badadam", "English", 7.69, 6.17, "electronic", ["https://www.youtube.com/watch?v=mEoBwBACmXs"]),
# ("Comic_bakery.mp3", "Instant Remedy", "English", 7.34, 6.1, "electronic", ["https://www.youtube.com/watch?v=FmCMOCQ8W6k"]),
# ("Descent into the Depths", "Midnight Syndicate", "English", 3.6, 2.73, "electronic", ["https://www.youtube.com/watch?v=smzTRJrznW4"]),
# ("Street Fighters", "Ananda Shake", "English", 2.62, 2.0, "electronic", ["https://www.youtube.com/watch?v=afFpVJ6GTNQ"]),
# ("Jack In The Box", "Logic Bomb", "English", 3.49, 2.67, "electronic", ["https://www.youtube.com/watch?v=Kg_MzEOfnv8"]),
# ("Lost In a Dream Yell", "Asia Minor", "English", 1.05, 1.75, "electronic", ["https://www.youtube.com/watch?v=w5zlIqeRN7c"]),
# ("Counting Back To 1", "Beautiful Small Machines", "English", 7.96, 6.21, "electronic", ["https://www.youtube.com/watch?v=BCzsMAHCt6Y"]),
# ("Triumph of a Heart (Radio Edit)", "Björk", "English", 5.49, 4.52, "electronic", ["https://www.youtube.com/watch?v=0z-rhM-dcO8"]),
# ("Pow Pow", "LCD Soundsystem", "English", 6.86, 4.55, "electronic", ["https://www.youtube.com/watch?v=6Zw3Hk7V1mE"]),
# ("Innocence", "Björk", "English", 7.09, 5.3, "electronic", ["https://www.youtube.com/watch?v=zSGPz_en90A"]),
# ("Hand Me Down Your Love", "Hot Chip", "English", 6.88, 4.43, "electronic", ["https://www.youtube.com/watch?v=t_v7yRXd3zY"]),
# ("Summer Crane", "The Avalanches", "English", 6.08, 4.09, "electronic", ["https://www.youtube.com/watch?v=k5r5FLII3c4"]),
# ("Just Saying", "Jamie xx", "English", 5.79, 3.8, "electronic", ["https://www.youtube.com/watch?v=kVx2jSFuUQU"]),
# ("Love Underlined", "Metronomy", "English", 5.79, 3.8, "electronic", ["https://www.youtube.com/watch?v=V2kVa7zNl38"]),
# ("You're My Excuse to Travel", "Baths", "English", 6.06, 4.61, "electronic", ["https://www.youtube.com/watch?v=mcAlMhfY0UI"]),
# ("Bad Body Double", "Imogen Heap", "English", 6.29, 4.26, "electronic", ["https://www.youtube.com/watch?v=fRUfYslxt3I"]),
# ("Nights Off", "Siriusmo", "English", 7.19, 4.87, "electronic", ["https://www.youtube.com/watch?v=XcNpg8eEyMU"]),
# ("Ever Falling", "Amon Tobin", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=JUSVf5UtxI8"]),
# ("Drumheller", "Caribou", "English", 6.14, 3.7, "electronic", ["https://www.youtube.com/watch?v=R5tjo6cBDRA"]),
# ("Garbage", "Chairlift", "English", 7.24, 6.13, "electronic", ["https://www.youtube.com/watch?v=64Js6P0VYrI"]),
# ("Sweetness", "Fischerspooner", "English", 5.94, 4.45, "electronic", ["https://www.youtube.com/watch?v=eNMtj2ZmfyU"]),
# ("Ghostship", "Menomena", "English", 6.59, 3.33, "electronic", ["https://www.youtube.com/watch?v=GAbcK1FvrMw"]),
# ("High Together", "Siriusmo", "English", 7.21, 4.86, "electronic", ["https://www.youtube.com/watch?v=NdmwPadDlgQ"]),
# ("Pendulum", "Broadcast", "English", 5.86, 3.89, "electronic", ["https://www.youtube.com/watch?v=yS4t-Wxmz-M"]),
# ("Breathe . Something/Stellar STar", "Flying Lotus", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=By8xW_HK9GU"]),
# ("Seven Hours With a Backseat Driver", "Gotye", "English", 7.09, 5.05, "electronic", ["https://www.youtube.com/watch?v=ixzD-wzxUEQ"]),
# ("Mic Check", "Cornelius", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=L5xwwpNwQ70"]),
# ("Here We Go", "Mr. Scruff", "English", 7.24, 4.61, "electronic", ["https://www.youtube.com/watch?v=7lCANLyM02o"]),
# ("Franks", "Infected Mushroom", "English", 6.06, 4.33, "electronic", ["https://www.youtube.com/watch?v=1peKEvok74o"]),
# ("The Blue Wrath", "I Monster", "English", 5.76, 3.88, "electronic", ["https://www.youtube.com/watch?v=sOKwewc9dns"]),
# ("Word Up (Feat. Ghostface Killah)", "MSTRKRFT", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=uHjO2tunW5Q"]),
# ("Lunch Hour Pops", "Broadcast", "English", 6.59, 4.26, "electronic", ["https://www.youtube.com/watch?v=AyIf8Uk_fZU"]),
# ("The Kids Quartet", "MGMT", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=oLL44k5SpnA"]),
# ("Money Power Respect", "Diplo", "English", 5.76, 3.23, "electronic", ["https://www.youtube.com/watch?v=CA-r3wHRVqk"]),
# ("Dance Face 2000", "Starfucker", "English", 6.55, 3.58, "electronic", ["https://www.youtube.com/watch?v=4fJPoChYOT4"]),
# ("Sorry", "Moloko", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=_6Gi_gpjKIU"]),
# ("Pom Pom", "Matthew Dear", "English", 6.61, 4.67, "electronic", ["https://www.youtube.com/watch?v=LoYdxbMu3RU"]),
# ("I Can't Feel", "Matthew Dear", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=Nzd28gxnfWg"]),
# ("Roulette Thrift Run", "Clark", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=CYuDeaZjxFk"]),
# ("Eat Your Heart", "Micachu", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=rkwiByJjwOA"]),
# ("Newme", "Jamie Lidell", "English", 6.79, 4.0, "electronic", ["https://www.youtube.com/watch?v=erQnJH27zNA"]),
# ("The Number One Song In Heaven", "Sparks", "English", 7.4, 5.33, "electronic", ["https://www.youtube.com/watch?v=P6I6yr7WDeg"]),
# ("Music", "Cornelius", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=u76XEmLdWz0"]),
# ("I Played Organ While Colter Played Guitar", "Devendra Banhart", "English", 6.8, 2.21, "electronic", ["https://www.youtube.com/watch?v=A4Te9Vo2Zrc"]),
# ("I My Me Mine", "POLYSICS", "English", 7.57, 5.77, "electronic", ["https://www.youtube.com/watch?v=X2Er1654hlo"]),
# ("Nothing Is Wrong", "FC/Kahuna", "English", 6.46, 4.48, "electronic", ["https://www.youtube.com/watch?v=Mm1OcnKDLqY"]),
# ("Pioneers (M83 Remix)", "Bloc Party", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=52jWkrfjakk"]),
# ("Idiologie", "Siriusmo", "English", 5.41, 4.93, "electronic", ["https://www.youtube.com/watch?v=7mdpz5z8I4I"]),
# ("Baby's Got Her Gun Out", "Rediscover", "English", 7.6, 5.34, "electronic", ["https://www.youtube.com/watch?v=zivPSpCBRak"]),
# ("Toner", "Cornelius", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=NDXY5pN8LyA"]),
# ("Turret Wife Serenade", "Aperture Science Psychoacoustics Laboratory", "English", 6.84, 4.92, "electronic", ["https://www.youtube.com/watch?v=JxsMHvbTOqw"]),
# ("Micro Course In Russian", "DJ Vadim", "English", 7.11, 4.38, "electronic", ["https://www.youtube.com/watch?v=8JiGKDUwBJk"]),
# ("Casio Daisy", "Gold Panda", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=lGMfCCxFswA"]),
# ("Holoday", "Laurel Halo", "English", 6.43, 4.35, "electronic", ["https://www.youtube.com/watch?v=1yTIeANebtw"]),
# ("Fatman", "Pop Will Eat Itself", "English", 3.49, 2.67, "electronic", ["https://www.youtube.com/watch?v=SjVRLdYdcbQ"]),
# ("Katie W", "Fenix TX", "English", 2.62, 2.0, "electronic", ["https://www.youtube.com/watch?v=nArer1G6mtY"]),
# ("Wake Up! Time to Die...", "Pop Will Eat Itself", "English", 3.91, 2.99, "electronic", ["https://www.youtube.com/watch?v=j7OUeFShS6o"]),
# ("Adagio de Samuel Barber (string-choral-mix)", "Stidiek", "English", 5.45, 3.91, "electronic", [""]),
# ("Static Grate", "B. Fleischmann", "English", 6.6, 4.63, "electronic", ["https://www.youtube.com/watch?v=aDzgyncyYhk"]),
# ("Last Train to Trancentral (Live From the Lost Continent)", "The KLF", "English", 6.26, 4.82, "electronic", ["https://www.youtube.com/watch?v=pC_zffOenk8"]),
# ("Message", "Vangelis", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=SiBb3OwNDnw"]),
# ("Iron John", "Alphaville", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=aXPQMq4Y01k"]),
# ("Ivory Tower", "Alphaville", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=yDl-qOqizi0"]),
# ("The One Thing", "Alphaville", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=gvV7MbZVUng"]),
# ("Oh Patti", "Alphaville", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=mzqV6G8oRCk"]),
# ("Xpander Edit", "Sasha", "English", 6.08, 4.79, "electronic", ["https://www.youtube.com/watch?v=4EBK7qn-n1k"]),
# ("Meeting Point", "Jap Jap", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=LJ697FfGuzI"]),
# ("Atze de Boe/ Unsere Lieder", "MIA.", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=9PXG-ij4eH4"]),
# ("smiles of a summer night", "le visage", "English", 5.35, 5.83, "electronic", ["https://www.youtube.com/watch?v=4gomLBWhW5o"]),
# ("Think Alike (album version)", "Keshco", "English", 5.06, 5.92, "electronic", ["https://www.youtube.com/watch?v=gAcbylNOQuI"]),
# ("The Beauty of the Swarm", "Nicolas Dominique", "English", 5.29, 4.92, "electronic", ["https://www.youtube.com/watch?v=AeMc0cygHNw"]),
# ("Hey you", "Basement Jaxx", "English", 5.85, 5.91, "electronic", ["https://www.youtube.com/watch?v=TYWyZj285DA"]),
# ("Granite (Orginal Mix)", "Pendulum", "English", 6.37, 5.38, "electronic", ["https://www.youtube.com/watch?v=j6JhI4fVxPQ"]),
# ("Twist (Jaques Lu Cont's Conversion Perversion Mix)", "Goldfrapp", "English", 4.16, 4.88, "electronic", ["https://www.youtube.com/watch?v=Q_QiwMmKivY"]),
# ("Pa' Llegar a Tu Lado", "Lhasa", "English", 5.46, 5.14, "electronic", ["https://www.youtube.com/watch?v=qlbRLhOF5Gw"]),
# ("Nightheat", "Shiva in Exile", "English", 5.44, 4.48, "electronic", ["https://www.youtube.com/watch?v=sZJB2z4zEPQ"]),
# ("Mind Feeders", "Dom & Roland", "English", 6.42, 6.38, "electronic", ["https://www.youtube.com/watch?v=n5iM-EV-BY4"]),
# ("Gas", "Lucy", "English", 7.19, 6.16, "electronic", ["https://www.youtube.com/watch?v=sNC7AQi7wR8"]),
# ("Ready Steady Go (ft.Asher D) -", "Paul Oakenfold", "English", 6.21, 5.08, "electronic", ["https://www.youtube.com/watch?v=dHgGIVGPEyE"]),
# ("108", "Skincage", "English", 7.19, 6.16, "electronic", ["https://www.youtube.com/watch?v=BqYFq8sj4hI"]),
# ("Robibot (Live)", "Filewile", "English", 5.53, 4.46, "electronic", ["https://www.youtube.com/watch?v=BUAT-lrMmjQ"]),
# ("Jenerik", "Ezel", "English", 7.19, 6.16, "electronic", ["https://www.youtube.com/watch?v=MMH4GaSZpc0"]),
# ("afterall", "Delerium", "English", 5.94, 4.68, "electronic", ["https://www.youtube.com/watch?v=1gMxKBLftq4"]),
# ("Mutant Genius (New Genious)", "Gorillaz", "English", 5.19, 4.47, "electronic", ["https://www.youtube.com/watch?v=jJ0ldlGeLUs"]),
# ("Colours (DFA Remix)", "Hot Chip", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=sekxhcIUrJw"]),
# ("Uni C", "Alva Noto", "English", 4.32, 4.8, "electronic", ["https://www.youtube.com/watch?v=gh24zhRNnZI"]),
# ("Uni Fac", "Alva Noto", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=xzEqJMDL_cw"]),
# ("Uni Rec", "Alva Noto", "English", 4.32, 4.8, "electronic", ["https://www.youtube.com/watch?v=GBxyRh8PZt4"]),
# ("Uni Dia", "Alva Noto", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=tO6L1N63I04"]),
# ("Get On", "Yello", "English", 5.82, 6.07, "electronic", ["https://www.youtube.com/watch?v=lHz9g2LcCuk"]),
# ("Nervous", "Yello", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=WlBmIZlUgnk"]),
# ("Uni Syc", "Alva Noto", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=52-04SJfvI0"]),
# ("On Track", "Yello", "English", 4.63, 4.44, "electronic", ["https://www.youtube.com/watch?v=voYWP2wxnoo"]),
# ("Prisoner Of His Mind", "Yello", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=8lAJvldvYyY"]),
# ("Beyond The Continuum", "Dymons", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=n0_ETA7icCM"]),
# ("Never Never Land", "Lene Lovich", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=Bj2juTSwoZY"]),
# ("Fences", "Peas", "English", 6.07, 3.77, "electronic", ["https://www.youtube.com/watch?v=TnqsEZal60c"]),
# ("Mr. Hudd Refuses To Sit", "Silence", "English", 1.78, 2.75, "electronic", ["https://www.youtube.com/watch?v=RKtlRc-Hzpg"]),
# ("Drunken Mozart", "Edgar Froese", "English", 4.61, 6.05, "electronic", ["https://www.youtube.com/watch?v=BUSnl-bkVZE"]),
# ("Stalker (LP Version)", "Covenant", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=NhnGzIHtp5c"]),
# ("Nightlife (Zero DB Reconstruction)", "Bonobo", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=r6Agqjut1W0"]),
# ("Zombie'ites (Laughing Gravy Mix)", "Transglobal Underground", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=DIV0fsaEqiw"]),
# ("Sky Carillon", "Hangman alive", "English", 4.4, 4.75, "electronic", ["https://www.youtube.com/watch?v=alWhuZ4vOsU"]),
# ("Androids Have No Fun", "Flanicx", "English", 6.25, 5.09, "electronic", ["https://www.youtube.com/watch?v=hzwNPxv73iE"]),
# ("Bleu (Aor)", "Jean Michel Jarre", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=oEHivc-A5yI"]),
# ("Freya", "cleenmusic", "English", 4.32, 4.8, "electronic", [""]),
# ("Toasted BotBop", "Animals on Wheels", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=wFj3_1gcYrQ"]),
# ("To The Club", "Spankox", "English", 3.56, 5.51, "electronic", ["https://www.youtube.com/watch?v=3_mHpPuQ8g0"]),
# ("Canticle Drawl", "AFX", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=6UKvWbp6VIY"]),
# ("Manicure", "Biosphere", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=S3vNvSQBhjI"]),
# ("Total Paranoia", "Orbital", "English", 4.0, 4.66, "electronic", ["https://www.youtube.com/watch?v=BrgZRY-OYeQ"]),
# ("Fantasia", "Caustic Window", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=8Pu_IER4wtc"]),
# ("Thursday", "Modwheelmood", "English", 2.71, 5.49, "electronic", ["https://www.youtube.com/watch?v=Kcj3AYnnmgk"]),
# ("Web", "Brian Eno", "English", 4.49, 4.4, "electronic", ["https://www.youtube.com/watch?v=D3xaE9P4Co8"]),
# ("U.N. Owen Was Her?", "上海アリス幻樂団", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=o_fWvC40zBQ"]),
# ("Antibodies (Chateau Flight Remix)", "Poni Hoax", "English", 4.0, 4.53, "electronic", ["https://www.youtube.com/watch?v=-L4JSuouMKQ"]),
# ("Double Trouble", "동방신기", "English", 5.08, 6.06, "electronic", ["https://www.youtube.com/watch?v=CnSPVfZYz2o"]),
# ("Fueled", "Displacer", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=2SiNX3Lpr74"]),
# ("Gathering Clouds 2", "Deepspace", "English", 4.82, 3.5, "electronic", ["https://www.youtube.com/watch?v=Xu7mNLUU49U"]),
# ("Pressure Wounds", "Tetradin x Advance", "English", 4.24, 4.71, "electronic", ["https://www.youtube.com/watch?v=uiJ5jZ0OSX0"]),
# ("Nuances", "Unmorph", "English", 5.34, 4.86, "electronic", ["https://www.youtube.com/watch?v=nXq0yKH1XSc"]),
# ("Relent Less", "skin contact", "English", 4.68, 4.93, "electronic", ["https://www.youtube.com/watch?v=CaKEq_vqARE"]),
# ("Pterodactyl", "Tim Doyle", "English", 5.49, 4.54, "electronic", ["https://www.youtube.com/watch?v=JCiHX0sn4BQ"]),
# ("Stranger (To Stability) (Len Faki Podium Mix)", "Dustin Zahn", "English", 4.97, 5.06, "electronic", ["https://www.youtube.com/watch?v=TEtBg6od_mA"]),
# ("Northern Passing", "Unmorph", "English", 5.35, 4.18, "electronic", ["https://www.youtube.com/watch?v=7l-b_gXQ6nQ"]),
# ("Le Jeu des Pistes, Une Fois Contée - 06 - Ascension, NO alcohol music", "Le Jeu des Pistes", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=wtGa8VzPF48"]),
# ("40 - Confins, NO alcohol music", "Qui Dit", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=qQSfFHVzDG0"]),
# ("23 - Mouvement Focal, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=ntX3_PY-NXs"]),
# ("Enjeu Final, Martelante - 03 Martelante, NO alcohol music", "Enjeu Final", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=vljCKPltIvM"]),
# ("13 - Pour Cause, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.75, 4.39, "electronic", ["https://www.youtube.com/watch?v=MriPgS8_wwA"]),
# ("Wholesome", "Mezzanine Stairs", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=tiLW-K935rI"]),
# ("02 - Attracteur Limite, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("01 - Polarité, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("02 - Confusion, NO alcohol, alkolgabe musika", "Zuhurbelea", "English", 4.88, 4.3, "electronic", [""]),
# ("Sanity", "Unmorph", "English", 4.27, 4.92, "electronic", ["https://www.youtube.com/watch?v=JGX-q47XbQw"]),
# ("In Suspense", "Psychadelik Pedestrian", "English", 4.0, 5.58, "electronic", ["https://www.youtube.com/watch?v=zYQcf1DV89A"]),
# ("ThisFuckingSong", "JustLikeAmmy", "English", 3.08, 5.45, "electronic", ["https://www.youtube.com/watch?v=OoC8RTlNShk"]),
# ("Cyberstalker", "Psychadelik Pedestrian", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=EVmc5tOadPM"]),
# ("01 - Prémices, NO alcohol, alkolgabe musika", "Zuhurbelea", "English", 4.9, 4.42, "electronic", [""]),
# ("22 - Corridor, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("03 - Subir, NO alcohol, alkolgabe musika", "Zuhurbelea", "English", 4.9, 4.42, "electronic", [""]),
# ("Dolores Park", "Tim Doyle", "English", 4.84, 4.34, "electronic", ["https://www.youtube.com/watch?v=J4_wXPZ1Bnk"]),
# ("25 - Excellence, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("10 - Seconde Réflexion, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=MriPgS8_wwA"]),
# ("ThatBrokenDrum", "JustLikeAmmy", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=Z_4pWhUX-Vk"]),
# ("05 - Confusion, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("Cxscxdxs: Injured & Lost in a Pitch Black World", "krovo", "English", 3.96, 5.96, "electronic", [""]),
# ("09 - Après les Cimes, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("08 - Entre les Lignes, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=MriPgS8_wwA"]),
# ("06 - Silice, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("11 - Régénère, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", [""]),
# ("02 - Horizons Permanents, NO alcohol music", "Zeru ta Lur / Cérou ta Lourre", "English", 4.98, 4.44, "electronic", [""]),
# ("Mdk 5 Démo", "Zeru ta Lur / Cérou ta Lourre", "English", 2.75, 5.32, "electronic", [""]),
# ("08 - Entre les Lignes", "Zeru ta Lur / Cérou ta Lourre", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=MriPgS8_wwA"]),
# ("02 Rotation", "Garde Réelle", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=AiaOSGZTwtY"]),
# ("03 Martelante", "Enjeu Final", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=oGneAab3e88"]),
# ("03 - Tyran", "Zeru ta Lur", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=ikQcPzSLxGk"]),
# ("Cancel", "Hawk Duncan", "English", 3.85, 3.43, "electronic", ["https://www.youtube.com/watch?v=QZEe7fAhMAA"]),
# ("6 - Ascension", "Le Jeu des Pistes", "English", 4.9, 4.42, "electronic", ["https://www.youtube.com/watch?v=9r04cr_sxh4"]),
# ("November X Ray Mexico", "Richard H Kirk", "English", 2.75, 5.32, "electronic", ["https://www.youtube.com/watch?v=Qz8_-kgNcVk"]),
# ("Garde Réelle, Les Espaces Croisés - 02 Rotation Axiale, NO alcohol music", "Garde Réelle", "English", 4.9, 4.42, "electronic", [""]),
# ("03 - Mille Couleurs, NO alcohol music", "Cérou ta Loure", "English", 4.9, 4.42, "electronic", [""]),
# ("Sickness Unto Foolish Death", "Akira Yamaoka", "English", 5.13, 4.93, "electronic", ["https://www.youtube.com/watch?v=Mvle7cjUQzM"]),
# ("First Death In The Family", "The Future Sound of London", "English", 3.68, 4.18, "electronic", ["https://www.youtube.com/watch?v=to6VJcVfPy4"]),
# ("No Place For Us", "The Sight Below", "English", 5.29, 4.17, "electronic", ["https://www.youtube.com/watch?v=0yffB2Y_Z6c"]),
# ("repeat", "Mell", "English", 4.52, 5.1, "electronic", ["https://www.youtube.com/watch?v=kUlf-hBbpKU"]),
# ("Angels", "Nick Cave", "English", 3.8, 6.2, "electronic", ["https://www.youtube.com/watch?v=wc7mDChiiNU"]),
# ("Golgotha : Flies", "Chaos As Shelter", "English", 4.44, 5.14, "electronic", ["https://www.youtube.com/watch?v=NWaAOOFjxn0"]),
# ("Bengal Joy (Introitus)", "Datamatrix", "English", 5.77, 4.2, "electronic", [""]),
# ("Senkou", "小西香葉、近藤由紀夫", "English", 3.8, 6.2, "electronic", ["https://www.youtube.com/watch?v=EbgfJQ8yB10"]),
# ("Anxiety", "Temperament", "English", 4.54, 4.53, "electronic", ["https://www.youtube.com/watch?v=JqwMM1AAxVk"]),
# ("Anxiety", "Temperament-Music", "English", 4.68, 4.96, "electronic", ["https://www.youtube.com/watch?v=KhnEUbbMWUM"]),
# ("Loud Places", "Jamie xx", "English", 5.5, 3.19, "electronic", ["https://www.youtube.com/watch?v=TP9luRtEqjc"]),
# ("Made in the Dark", "Hot Chip", "English", 6.87, 3.27, "electronic", ["https://www.youtube.com/watch?v=I9fcaRNUpYA"]),
# ("In the Privacy of Our Love", "Hot Chip", "English", 6.63, 3.45, "electronic", ["https://www.youtube.com/watch?v=FLbNOgHdAw8"]),
# ("Great Escape", "Moby", "English", 6.92, 3.11, "electronic", ["https://www.youtube.com/watch?v=lHZZYEMpLtQ"]),
# ("Heaven's Light", "Air", "English", 6.03, 2.46, "electronic", ["https://www.youtube.com/watch?v=xAv4CERKQeI"]),
# ("Everything Merges With The Night", "Brian Eno", "English", 6.26, 2.63, "electronic", ["https://www.youtube.com/watch?v=WQjrTh33gUs"]),
# ("You Were There With Me", "Four Tet", "English", 5.63, 3.62, "electronic", ["https://www.youtube.com/watch?v=aIrqjpnPx50"]),
# ("Bad Luck", "Hot Chip", "English", 6.62, 3.19, "electronic", ["https://www.youtube.com/watch?v=YBgR832jofY"]),
# ("When No One Cares", "Junior Boys", "English", 6.19, 2.96, "electronic", ["https://www.youtube.com/watch?v=8BaPBvCJfrI"]),
# ("Nova Scotia Robots", "Boards of Canada", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=p8o7-GcztLI"]),
# ("Broken Mirrors", "Chromatics", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=sI4IL23M9H8"]),
# ("Bubble and Spike", "Telefon Tel Aviv", "English", 5.72, 3.11, "electronic", ["https://www.youtube.com/watch?v=wxGf39sxllY"]),
# ("The Weight Of My Words (Four Tet Remix)", "Kings of Convenience", "English", 6.23, 3.69, "electronic", ["https://www.youtube.com/watch?v=NFKGH-nTCNs"]),
# ("Sister Song", "Perfume Genius", "English", 5.47, 4.42, "electronic", ["https://www.youtube.com/watch?v=WFW2E8SpU74"]),
# ("Keep It Together", "How to Destroy Angels", "English", 6.37, 3.22, "electronic", ["https://www.youtube.com/watch?v=EoXSUOjGk6w"]),
# ("Solo Dancing", "Indiana", "English", 7.17, 3.04, "electronic", ["https://www.youtube.com/watch?v=WSrktmE963I"]),
# ("All I See", "Lydia", "English", 5.68, 3.57, "electronic", ["https://www.youtube.com/watch?v=V3bzil_2tv4"]),
# ("Beams", "The Presets", "English", 5.92, 3.42, "electronic", ["https://www.youtube.com/watch?v=EeYGD9tAsW8"]),
# ("Part 3", "Rhian Sheehan", "English", 6.53, 3.61, "electronic", ["https://www.youtube.com/watch?v=zcjD1A26z5s"]),
# ("Dust Clears", "Clean Bandit", "English", 7.06, 4.25, "electronic", ["https://www.youtube.com/watch?v=SdfL2nY-Xs8"]),
# ("P", "Labradford", "English", 7.08, 4.6, "electronic", ["https://www.youtube.com/watch?v=RGUktDaIKMI"]),
# ("Thank You", "Röyksopp", "English", 6.87, 2.36, "electronic", ["https://www.youtube.com/watch?v=ofpro72ku14"]),
# ("Seal Beach", "The Album Leaf", "English", 5.58, 2.53, "electronic", ["https://www.youtube.com/watch?v=LAmat2RRq68"]),
# ("A Ballad to Forget", "Soulwax", "English", 5.79, 3.32, "electronic", ["https://www.youtube.com/watch?v=NXYEfHnB47A"]),
# ("Trust Nobody (feat. Selena Gomez & Tory Lanez)", "Cashmere Cat", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=1Vn1BXfsd4Q"]),
# ("Therapy Car Noise", "Department of Eagles", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=qZrEKdB2Ss8"]),
# ("Remember", "Lali Puna", "English", 7.42, 3.13, "electronic", ["https://www.youtube.com/watch?v=71OB2kh8SL0"]),
# ("Galamb egyedül", "Venetian Snares", "English", 7.1, 3.51, "electronic", ["https://www.youtube.com/watch?v=wNWEmo59dGQ"]),
# ("I See Fire (Kygo Remix)", "Ed Sheeran", "English", 6.3, 2.88, "electronic", ["https://www.youtube.com/watch?v=oWYp1xRPH5g"]),
# ("Spirit", "Dead Can Dance", "English", 5.81, 3.05, "electronic", ["https://www.youtube.com/watch?v=dXo-XsdeXI4"]),
# ("I Like to Score", "Moby", "English", 6.75, 2.42, "electronic", ["https://www.youtube.com/watch?v=jGakY7PFobw"]),
# ("HyperParadise (Flume remix)", "Hermitude", "English", 4.59, 3.17, "electronic", ["https://www.youtube.com/watch?v=dIyKy4A4kBU"]),
# ("Droopy likes your face", "C418", "English", 6.03, 2.46, "electronic", ["https://www.youtube.com/watch?v=Hm51fAadWUI"]),
# ("Steal Away", "Harold Budd/Brian Eno", "English", 4.93, 2.14, "electronic", ["https://www.youtube.com/watch?v=0l1zL6YMZfw"]),
# ("Composure", "B. Fleischmann", "English", 6.79, 3.34, "electronic", ["https://www.youtube.com/watch?v=vHE_2QgBfEw"]),
# ("LIGHT POWERED", "Deastro", "English", 7.86, 4.73, "electronic", ["https://www.youtube.com/watch?v=Tzkr-z7o2cI"]),
# ("Stave Peak", "Loscil", "English", 6.78, 3.82, "electronic", ["https://www.youtube.com/watch?v=v5ke9itwVIU"]),
# ("Falling Horses", "Efterklang", "English", 6.82, 3.6, "electronic", ["https://www.youtube.com/watch?v=ppnAvCoDPns"]),
# ("Beautiful Son", "Peaking Lights", "English", 5.61, 2.74, "electronic", ["https://www.youtube.com/watch?v=HINIs3Sp5Lk"]),
# ("Silberweidenpark", "Deichkind", "English", 6.46, 3.24, "electronic", ["https://www.youtube.com/watch?v=XxX-dmbB4t8"]),
# ("Song of the Dispossessed", "Dead Can Dance", "English", 3.28, 1.95, "electronic", ["https://www.youtube.com/watch?v=ps-t4W2T_rc"]),
# ("Weeping Willow", "Sébastien Schuller", "English", 4.83, 2.38, "electronic", ["https://www.youtube.com/watch?v=JuzKgOIe4c4"]),
# ("Confines Of Gravity", "PlayRadioPlay!", "English", 7.68, 4.58, "electronic", ["https://www.youtube.com/watch?v=_GvDw6Rw_fM"]),
# ("Some Crap About the Future", "Electric President", "English", 6.97, 3.12, "electronic", ["https://www.youtube.com/watch?v=XNA9Dp0BIZU"]),
# ("Ode to Simplicity", "Secret Garden", "English", 6.0, 3.05, "electronic", ["https://www.youtube.com/watch?v=XB1QGshIFfM"]),
# ("Prometheus", "Covenant", "English", 4.66, 2.85, "electronic", ["https://www.youtube.com/watch?v=4Z-QCDyL2q4"]),
# ("Fruend", "Braids", "English", 7.19, 3.28, "electronic", ["https://www.youtube.com/watch?v=726T6mylEZU"]),
# ("Wall of Memories", "Gesaffelstein", "English", 5.3, 3.33, "electronic", ["https://www.youtube.com/watch?v=Juz7WS4pegU"]),
# ("Calmer", "Zoot Woman", "English", 6.27, 2.96, "electronic", ["https://www.youtube.com/watch?v=VmkRYHzl3vk"]),
# ("Lucidity", "Soen", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=FRH9ADDqLIM"]),
# ("Until It Comes Again", "The Black Ghosts", "English", 5.7, 2.65, "electronic", ["https://www.youtube.com/watch?v=OYJDp-k68pA"]),
# ("Secret Circles", "Hybrid", "English", 5.49, 3.63, "electronic", ["https://www.youtube.com/watch?v=mKIpY43e4YM"]),
# ("Belladonna", "Red Snapper", "English", 6.4, 3.79, "electronic", ["https://www.youtube.com/watch?v=IJPt_KeVck8"]),
# ("Stagger", "Underworld", "English", 6.0, 3.22, "electronic", ["https://www.youtube.com/watch?v=1TzArpPHlR8"]),
# ("Forty-two", "Sonic Adventure Project", "English", 6.62, 3.68, "electronic", ["https://www.youtube.com/watch?v=2cJErMcMtdE"]),
# ("In Our Eyes", "Niki & the Dove", "English", 7.04, 3.23, "electronic", ["https://www.youtube.com/watch?v=5f6qQN-cGug"]),
# ("Remando Al Viento", "Schwarz & Funk", "English", 6.67, 2.9, "electronic", ["https://www.youtube.com/watch?v=HafJKr9UwWE"]),
# ("The Whispering Wind", "Moby", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=XKw6MewhYz8"]),
# ("Solid Ground", "Ms. John Soda", "English", 5.87, 2.76, "electronic", ["https://www.youtube.com/watch?v=rPdE8Mxq9xo"]),
# ("Sensuous", "Cornelius", "English", 6.31, 3.95, "electronic", ["https://www.youtube.com/watch?v=WK-71-b7NSo"]),
# ("Never Seen You Get So Low", "Aquilo", "English", 6.68, 1.81, "electronic", ["https://www.youtube.com/watch?v=ARkWXukMvok"]),
# ("Let's Roll (Feat. Blaktroniks)", "Parov Stelar", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=LSuwZR8S1NQ"]),
# ("Blue Sky And Yellow Sunflower", "Susumu Yokota", "English", 6.8, 3.77, "electronic", ["https://www.youtube.com/watch?v=rYH7gknIrj8"]),
# ("Alma", "Teddybears", "English", 5.92, 2.9, "electronic", ["https://www.youtube.com/watch?v=RU3V0AiunoM"]),
# ("Three Sisters", "Hammock", "English", 6.89, 1.67, "electronic", ["https://www.youtube.com/watch?v=wmrKo_-PKPs"]),
# ("Feather", "Devin Townsend Project", "English", 7.45, 3.02, "electronic", ["https://www.youtube.com/watch?v=yy4y3FUUPgs"]),
# ("Emerald and Stone", "Brian Eno", "English", 6.47, 3.68, "electronic", ["https://www.youtube.com/watch?v=gGaDk0dZIrU"]),
# ("Z Twig", "Aphex Twin", "English", 6.96, 4.09, "electronic", ["https://www.youtube.com/watch?v=yzY8OMxo99s"]),
# ("Space fertilizer", "M83", "English", 6.86, 4.24, "electronic", ["https://www.youtube.com/watch?v=MVRvrblZ3YE"]),
# ("Want U", "Lo-Fi-Fnk", "English", 7.63, 4.8, "electronic", ["https://www.youtube.com/watch?v=AIU7-_pD3rk"]),
# ("Betula Pendula", "Carbon Based Lifeforms", "English", 6.62, 3.89, "electronic", ["https://www.youtube.com/watch?v=NTajW1JsnKE"]),
# ("City in the Dust on My Window", "Hammock", "English", 4.03, 2.19, "electronic", ["https://www.youtube.com/watch?v=_pIbBTGMbGo"]),
# ("Satellite", "BT", "English", 6.25, 4.26, "electronic", ["https://www.youtube.com/watch?v=Z_hZUt1BbAs"]),
# ("And Then So Clear", "Brian Eno", "English", 5.94, 3.85, "electronic", ["https://www.youtube.com/watch?v=lcK8_kKCsq8"]),
# ("Gold for the Price of Silver (Erot Collaboration)", "Kings of Convenience", "English", 6.67, 3.34, "electronic", ["https://www.youtube.com/watch?v=iRutwPv87kY"]),
# ("Wondering", "Buckethead", "English", 4.39, 2.54, "electronic", ["https://www.youtube.com/watch?v=0kdL8NKGYco"]),
# ("Grace", "Moby", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=FJsEfihqAjM"]),
# ("Passage to Nagoya", "Arovane", "English", 5.51, 3.42, "electronic", ["https://www.youtube.com/watch?v=OwWtsUr5hWM"]),
# ("What Else Is There ? (Live)", "Röyksopp", "English", 6.56, 4.31, "electronic", ["https://www.youtube.com/watch?v=ADBKdSCbmiM"]),
# ("Lodge", "Nest", "English", 4.84, 2.99, "electronic", ["https://www.youtube.com/watch?v=r69TiaIE4uo"]),
# ("More Dead Than Alive (Get Away from the Medicine)", "Hammock", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=CMVno22E0K8"]),
# ("Rain", "Alphawezen", "English", 6.44, 4.47, "electronic", ["https://www.youtube.com/watch?v=kqbw2AirVJ0"]),
# ("Go Dreaming", "Dido", "English", 7.93, 4.01, "electronic", ["https://www.youtube.com/watch?v=x_S8HGIZdS4"]),
# ("Parallel Minds", "Oneohtrix Point Never", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=PsAWm0y-YM4"]),
# ("Filmosonic XL", "Efterklang", "English", 7.34, 3.6, "electronic", ["https://www.youtube.com/watch?v=6wld3Fw-_Xg"]),
# ("The Faith", "Leonard Cohen", "English", 7.68, 4.85, "electronic", ["https://www.youtube.com/watch?v=DmihIriFBPM"]),
# ("Hisen", "Susumu Yokota", "English", 6.25, 3.65, "electronic", ["https://www.youtube.com/watch?v=0EWhzdcidoA"]),
# ("I Love My Parents", "Buckethead", "English", 5.27, 3.39, "electronic", ["https://www.youtube.com/watch?v=ODzhJHS1aeE"]),
# ("Crystalline", "Younger Brother", "English", 6.64, 3.73, "electronic", ["https://www.youtube.com/watch?v=Hn_Nf1tUnFM"]),
# ("Humming Chorus", "William Orbit", "English", 5.27, 3.21, "electronic", ["https://www.youtube.com/watch?v=k-mnK622-6M"]),
# ("Almost Home (with Damien Jurado)", "Moby", "English", 7.88, 3.67, "electronic", ["https://www.youtube.com/watch?v=QVbD9lKVszE"]),
# ("Fern and Robin", "Loscil", "English", 6.86, 3.39, "electronic", ["https://www.youtube.com/watch?v=IRgedi_OV5c"]),
# ("The Sky At Night", "DJ Food", "English", 5.49, 3.91, "electronic", ["https://www.youtube.com/watch?v=sRCXBmV3tp8"]),
# ("The Mysteries", "David Bowie", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=wWg2wiNsfNM"]),
# ("Night Village", "Deep Forest", "English", 6.13, 3.64, "electronic", ["https://www.youtube.com/watch?v=lIF5EEneWEU"]),
# ("Die Dinge des Lebens", "To Rococo Rot", "English", 6.09, 4.07, "electronic", ["https://www.youtube.com/watch?v=QRdVeE8uErg"]),
# ("I Believed In God", "The Flashbulb", "English", 7.86, 5.21, "electronic", ["https://www.youtube.com/watch?v=9fQGTo4A-eE"]),
# ("Still, Still, Still", "Mannheim Steamroller", "English", 7.29, 3.67, "electronic", ["https://www.youtube.com/watch?v=xpnGgXL5B3g"]),
# ("Attends", "Swod", "English", 5.87, 4.25, "electronic", ["https://www.youtube.com/watch?v=XMAqOp0LznU"]),
# ("Cider Time", "Lifeformed", "English", 7.86, 5.21, "electronic", ["https://www.youtube.com/watch?v=KXH5G_bqrlc"]),
# ("Without Motion", "The Sight Below", "English", 4.15, 3.16, "electronic", ["https://www.youtube.com/watch?v=3ES4aIOpkds"]),
# ("Walking On A Dream (Treasure Fingers Remix)", "Empire of the Sun", "English", 7.54, 4.53, "electronic", ["https://www.youtube.com/watch?v=JHNG1wL1B7Q"]),
# ("Life's Fading Light", "The Sight Below", "English", 5.02, 3.72, "electronic", ["https://www.youtube.com/watch?v=vNJMI-yDDMU"]),
# ("Suddenly Yours", "2002", "English", 6.21, 3.13, "electronic", ["https://www.youtube.com/watch?v=2tjBnFs2TMI"]),
# ("Breathe Between Sleep", "Christ.", "English", 6.61, 3.8, "electronic", ["https://www.youtube.com/watch?v=oLVkXC2sJDA"]),
# ("Disempowered", "Olan Mill", "English", 3.88, 2.94, "electronic", ["https://www.youtube.com/watch?v=Fp1iUPvbjP0"]),
# ("Fifteen White", "World's End Girlfriend", "English", 6.87, 4.01, "electronic", ["https://www.youtube.com/watch?v=hnGYl_saGEI"]),
# ("Twenty Four Hours In Lake of Ice", "Alaska In Winter", "English", 4.0, 2.19, "electronic", ["https://www.youtube.com/watch?v=NeDoN8aE3-0"]),
# ("Close Your Eyes - We Are Blind", "Alaska In Winter", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=Jb8npK14wL0"]),
# ("The Most Gorgeous Sunset", "Kuba", "English", 4.72, 2.93, "electronic", ["https://www.youtube.com/watch?v=twvOT5juekI"]),
# ("Awakening", "Liquid Mind", "English", 7.24, 4.08, "electronic", ["https://www.youtube.com/watch?v=NCa4dpKyzwo"]),
# ("Whenusleep", "Salem", "English", 6.58, 3.81, "electronic", ["https://www.youtube.com/watch?v=c9zo7v0BiOs"]),
# ("K-half Noise", "Mum", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=CCdYkFeedZU"]),
# ("Oceanic Glow", "Obfusc", "English", 7.69, 3.21, "electronic", ["https://www.youtube.com/watch?v=fpi7z8ZQPPw"]),
# ("Further Away", "The Sight Below", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=aKmqHll9UwQ"]),
# ("Honest Love", "Voodoo Child", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=PK13knRiJY0"]),
# ("Son", "Juana Molina", "English", 5.34, 2.0, "electronic", ["https://www.youtube.com/watch?v=nCiTws-CBms"]),
# ("Eternal Bliss of the Grateful Souls", "Shulman", "English", 7.45, 3.02, "electronic", ["https://www.youtube.com/watch?v=u94w_jOy_mI"]),
# ("Storm (Radio Mix)", "CJ Stone", "English", 6.51, 4.2, "electronic", ["https://www.youtube.com/watch?v=PUPbAVkGBmc"]),
# ("Coyote", "Harold Budd", "English", 5.2, 2.56, "electronic", ["https://www.youtube.com/watch?v=Piji-AhqQiY"]),
# ("Avada Kedavra", "The Age of Rockets", "English", 6.92, 4.54, "electronic", ["https://www.youtube.com/watch?v=kOgKKN73u4g"]),
# ("Chemistry", "DJ Encore", "English", 5.95, 4.45, "electronic", ["https://www.youtube.com/watch?v=B0iqtPcPV3k"]),
# ("Air For Life (Mirco De Govia Remix)", "Above & Beyond vs. Andy Moor", "English", 5.55, 3.92, "electronic", ["https://www.youtube.com/watch?v=JcrA5kQYNvA"]),
# ("Kim & Jessie (Montag Remix)", "M83", "English", 6.89, 4.18, "electronic", ["https://www.youtube.com/watch?v=uOq0OKKpCBk"]),
# ("Nautilus", "2002", "English", 7.62, 4.53, "electronic", ["https://www.youtube.com/watch?v=N77LgSnmejk"]),
# ("The Sunset Passage", "The Sight Below", "English", 5.06, 2.46, "electronic", ["https://www.youtube.com/watch?v=SWY54cMqM4I"]),
# ("Kathleen's Song", "Ray Lynch", "English", 7.33, 4.07, "electronic", ["https://www.youtube.com/watch?v=dbV7B0Avpeg"]),
# ("Ustad Sultan Khan - Tanara", "Thievery Corporation", "English", 6.33, 3.62, "electronic", ["https://www.youtube.com/watch?v=LapFu0DCiiE"]),
# ("Kreuzung", "Hauschka", "English", 7.12, 3.33, "electronic", ["https://www.youtube.com/watch?v=LjhpMGrv6zY"]),
# ("Monorail", "Port Blue", "English", 7.24, 3.17, "electronic", ["https://www.youtube.com/watch?v=lCLRsVd4OB4"]),
# ("01-Jan", "Bang On A Can", "English", 6.72, 3.66, "electronic", ["https://www.youtube.com/watch?v=3GvtvdgnfzM"]),
# ("Lux Aeterna", "Paul Schwartz", "English", 4.68, 3.55, "electronic", ["https://www.youtube.com/watch?v=1K0xUoes-Bw"]),
# ("Breathing", "D'Alt Vila", "English", 6.64, 4.23, "electronic", ["https://www.youtube.com/watch?v=Kpt9rypA6Lk"]),
# ("Mystic River", "Celestial Aeon Project", "English", 6.48, 3.71, "electronic", ["https://www.youtube.com/watch?v=Lca26kGxnRM"]),
# ("Ian Fish, U.K. Heir", "David Bowie", "English", 5.82, 4.2, "electronic", ["https://www.youtube.com/watch?v=BiWFSyPdzy8"]),
# ("Surreal", "Ryan Stewart", "English", 7.68, 4.72, "electronic", ["https://www.youtube.com/watch?v=vfC1VcSbl0Q"]),
# ("Love Here", "Mr. Projectile", "English", 6.9, 4.4, "electronic", ["https://www.youtube.com/watch?v=VmYPH3yNl7M"]),
# ("Wind And Water", "Kitaro", "English", 6.11, 3.63, "electronic", ["https://www.youtube.com/watch?v=tHj34UEbQvQ"]),
# ("Olympic Airways (diskJokke remix)", "Foals", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=jm3u-hE2i2s"]),
# ("Balaam", "Christ.", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=R6h4XtzNWuc"]),
# ("Full Sail", "Ryan Farish", "English", 6.75, 4.03, "electronic", ["https://www.youtube.com/watch?v=f8l3aIJsupc"]),
# ("Out There", "DJ Encore", "English", 5.7, 4.06, "electronic", ["https://www.youtube.com/watch?v=MOPnOA2kfqs"]),
# ("sleepy quest for coffee", "Plastik Joy", "English", 7.54, 4.12, "electronic", ["https://www.youtube.com/watch?v=YZASHXctEKc"]),
# ("Lifegiving", "Australis", "English", 6.92, 3.79, "electronic", ["https://www.youtube.com/watch?v=Um6pNg00yjU"]),
# ("Mont St. Joseph", "Celestial Aeon Project", "English", 6.39, 3.66, "electronic", ["https://www.youtube.com/watch?v=E_6zHza-roo"]),
# ("Release To The System", "Mark Franklin", "English", 6.29, 3.91, "electronic", ["https://www.youtube.com/watch?v=-N7JyGaDNCw"]),
# ("Super Metroid One Girl in all the World OC ReMix", "The Wingless", "English", 6.78, 4.14, "electronic", ["https://www.youtube.com/watch?v=lbUOSL69iXQ"]),
# ("Chrono Trigger At the End of All Things OC ReMix", "Abadoss", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=kRM1IEKDipU"]),
# ("The Keeper (Grasscut Bitter Peace Remix)", "Bonobo", "English", 8.0, 4.38, "electronic", ["https://www.youtube.com/watch?v=pUP7YK9rRLE"]),
# ("Hear You Now (Grand Chillas mix)", "DJ Hooligan", "English", 6.23, 4.27, "electronic", ["https://www.youtube.com/watch?v=ibGpkhicUfE"]),
# ("Solace", "Akira Kosemura", "English", 5.33, 3.25, "electronic", ["https://www.youtube.com/watch?v=ea4aidGFgJo"]),
# ("Dreamdance", "Lisa Lynne", "English", 7.56, 4.37, "electronic", ["https://www.youtube.com/watch?v=lw9vTEMgdwM"]),
# ("Camel", "Flying Lotus", "English", 3.0, 1.67, "electronic", ["https://www.youtube.com/watch?v=XFuIvzxQK40"]),
# ("Golden Diva", "Flying Lotus", "English", 4.25, 3.44, "electronic", ["https://www.youtube.com/watch?v=sr7QHFK-ZAw"]),
# ("Maundy Thursday", "Air France", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=zhdrvuH_cu0"]),
# ("FSOSF", "Bassnectar", "English", 4.25, 2.33, "electronic", ["https://www.youtube.com/watch?v=Yb1QOCmUC0E"]),
# ("Auntie's Lock/Infinitum (feat Laura Darlington)", "Flying Lotus", "English", 3.71, 3.4, "electronic", ["https://www.youtube.com/watch?v=sYo4__B4qfc"]),
# ("The Bells", "Fluke", "English", 4.48, 2.79, "electronic", ["https://www.youtube.com/watch?v=FtvQlfRBPSg"]),
# ("Where its At", "Beck", "English", 3.89, 2.42, "electronic", ["https://www.youtube.com/watch?v=EPfmNxKLDG4"]),
# ("Nowism", "Freeland", "English", 5.04, 2.99, "electronic", ["https://www.youtube.com/watch?v=D9WoEzcKNng"]),
# ("Slipping away (single version)", "Moby", "English", 3.23, 0.97, "electronic", ["https://www.youtube.com/watch?v=kXCHdpaT_Zk"]),
# ("Higher State of Consciousness (Tweekin Acid Funk)", "Wink", "English", 2.62, 2.0, "electronic", ["https://www.youtube.com/watch?v=d3hAnAnJwyU"]),
# ("Looking So", "CFCF", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=77CPpYAmq4A"]),
# ("Instrumental", "LL Cool J", "English", 5.26, 3.9, "electronic", ["https://www.youtube.com/watch?v=WXfYBScEC34"]),
# ("Into The Storm", "Luke Chable", "English", 3.43, 2.45, "electronic", ["https://www.youtube.com/watch?v=tvU1ENZ_I8g"]),
# ("Tiki Mix", "The Gentle People", "English", 4.13, 2.3, "electronic", ["https://www.youtube.com/watch?v=c_PX5lEjny4"]),
# ("Layla In Dub (Last.FM exclusive)", "Instant Wilkie", "English", 4.45, 2.63, "electronic", [""]),
# ("Starling", "The Echoing Green", "English", 3.76, 2.45, "electronic", ["https://www.youtube.com/watch?v=U6j1ZaS5KFE"]),
# ("Parallel Bows", "Lagomorpha", "English", 3.62, 1.25, "electronic", ["https://www.youtube.com/watch?v=BROLGXwdiMo"]),
# ("Something About Us", "Daft Punk", "English", 6.35, 4.03, "electronic", ["https://www.youtube.com/watch?v=sOS9aOIXPEk"]),
# ("Nightvision", "Daft Punk", "English", 6.06, 3.49, "electronic", ["https://www.youtube.com/watch?v=xBTqRd09y3E"]),
# ("Alone in Kyoto", "Air", "English", 6.21, 3.54, "electronic", ["https://www.youtube.com/watch?v=I0SVd_Q5wIg"]),
# ("Ce Matin-là", "Air", "English", 6.58, 3.77, "electronic", ["https://www.youtube.com/watch?v=tRoBfcPxJdA"]),
# ("You Make It Easy", "Air", "English", 6.44, 3.93, "electronic", ["https://www.youtube.com/watch?v=mHjuYDloDtE"]),
# ("Venus", "Air", "English", 6.44, 3.67, "electronic", ["https://www.youtube.com/watch?v=Zrpne2Qr0z8"]),
# ("Make Love", "Daft Punk", "English", 6.8, 4.24, "electronic", ["https://www.youtube.com/watch?v=W0R-qPfb2bE"]),
# ("Finally Moving", "Pretty Lights", "English", 6.54, 3.09, "electronic", ["https://www.youtube.com/watch?v=Sk9XYQMRiLY"]),
# ("Headlock", "Imogen Heap", "English", 6.01, 3.7, "electronic", ["https://www.youtube.com/watch?v=roPiy2JydwA"]),
# ("Surfing on a rocket", "Air", "English", 6.62, 3.86, "electronic", ["https://www.youtube.com/watch?v=m3uHuWQrpMQ"]),
# ("Með Blóðnasir", "Sigur Rós", "English", 6.38, 4.24, "electronic", ["https://www.youtube.com/watch?v=nlVA_e6WQhw"]),
# ("Breathe In", "Frou Frou", "English", 6.52, 3.79, "electronic", ["https://www.youtube.com/watch?v=oet5L-gqctk"]),
# ("Just For Now", "Imogen Heap", "English", 5.97, 3.57, "electronic", ["https://www.youtube.com/watch?v=5yBBaK1vGh4"]),
# ("Ritual Union", "Little Dragon", "English", 5.98, 2.86, "electronic", ["https://www.youtube.com/watch?v=0Yeb3q5nqWA"]),
# ("A&E", "Goldfrapp", "English", 6.12, 4.15, "electronic", ["https://www.youtube.com/watch?v=p7Ptai9I6eo"]),
# ("Biological", "Air", "English", 6.36, 3.83, "electronic", ["https://www.youtube.com/watch?v=6iWcfaEcnCs"]),
# ("Olson", "Boards of Canada", "English", 6.36, 3.92, "electronic", ["https://www.youtube.com/watch?v=WcMFeMSIzu4"]),
# ("Must Be Dreaming", "Frou Frou", "English", 6.58, 3.9, "electronic", ["https://www.youtube.com/watch?v=1BT6_TL67Wk"]),
# ("Have You Got It In You?", "Imogen Heap", "English", 6.02, 3.88, "electronic", ["https://www.youtube.com/watch?v=CndVNF1kGXE"]),
# ("The Warning", "Hot Chip", "English", 6.56, 3.87, "electronic", ["https://www.youtube.com/watch?v=ykqDDSBcBKY"]),
# ("Country", "Empire of the Sun", "English", 5.98, 3.12, "electronic", ["https://www.youtube.com/watch?v=zyirFLQMng4"]),
# ("It's Good To Be In Love", "Frou Frou", "English", 6.33, 3.99, "electronic", ["https://www.youtube.com/watch?v=DavJoemJNBk"]),
# ("Space Maker", "Air", "English", 5.96, 3.38, "electronic", ["https://www.youtube.com/watch?v=GFXXkRLQy1Y"]),
# ("EMI", "Aus", "English", 6.52, 3.8, "electronic", ["https://www.youtube.com/watch?v=YA2P-0D2yds"]),
# ("Etherlands-4", "Wolfram Spyra", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=5JUXhXv75rc"]),
# ("Etherlands-9", "Wolfram Spyra", "English", 2.86, 2.04, "electronic", ["https://www.youtube.com/watch?v=1hyd0gIaTdo"]),
# ("Loving Arm", "Metronomy", "English", 7.13, 3.04, "electronic", ["https://www.youtube.com/watch?v=PW3s0_ISv-U"]),
# ("Crap Kraft Dinner", "Hot Chip", "English", 5.62, 3.81, "electronic", ["https://www.youtube.com/watch?v=HGnzj4l34QA"]),
# ("Wink", "Little Dragon", "English", 6.03, 4.51, "electronic", ["https://www.youtube.com/watch?v=siUOFWj-5YQ"]),
# ("Glory", "Wye Oak", "English", 7.57, 4.2, "electronic", ["https://www.youtube.com/watch?v=tMOcpGKTh84"]),
# ("Simple Girl", "IAMX", "English", 5.87, 4.11, "electronic", ["https://www.youtube.com/watch?v=Ty2jOvhJwoo"]),
# ("The Sailor", "The Album Leaf", "English", 6.38, 3.6, "electronic", ["https://www.youtube.com/watch?v=2GJY4bi1NUw"]),
# ("Teach Me How to Fight", "Junior Boys", "English", 6.02, 3.63, "electronic", ["https://www.youtube.com/watch?v=8PhSpJppLu4"]),
# ("Soft", "Lemon Jelly", "English", 6.12, 3.25, "electronic", ["https://www.youtube.com/watch?v=q0sSspVG35g"]),
# ("Maliblue", "Darius", "English", 6.14, 3.15, "electronic", ["https://www.youtube.com/watch?v=ntoMKM0ToP0"]),
# ("Mullholland", "Stars of the Lid", "English", 6.11, 3.39, "electronic", ["https://www.youtube.com/watch?v=9fSYKdTLeZk"]),
# ("Soft Atlas", "13 & God", "English", 6.4, 4.08, "electronic", ["https://www.youtube.com/watch?v=THTLAAz8lsY"]),
# ("'75 aka Stay With You", "Lemon Jelly", "English", 6.6, 3.73, "electronic", ["https://www.youtube.com/watch?v=96CYNeihFwg"]),
# ("Games", "Peter Broderick", "English", 5.04, 2.8, "electronic", ["https://www.youtube.com/watch?v=rY_CiNo0PPk"]),
# ("Shine", "Ulrich Schnauss", "English", 5.77, 3.44, "electronic", ["https://www.youtube.com/watch?v=V5nMGoIlXQo"]),
# ("Bugs Don’t Buzz", "Majical Cloudz", "English", 7.13, 3.04, "electronic", ["https://www.youtube.com/watch?v=7CLrDPtFjPI"]),
# ("I Was Lost Without You", "Sam Hulick", "English", 6.6, 4.34, "electronic", ["https://www.youtube.com/watch?v=uCwgu5W0ZaU"]),
# ("We’re Looking For A Lot Of Love", "Hot Chip", "English", 7.19, 3.92, "electronic", ["https://www.youtube.com/watch?v=WV4CQFD5eY0"]),
# ("Overlook", "Tycho", "English", 5.83, 3.18, "electronic", ["https://www.youtube.com/watch?v=l6pWIFIdz90"]),
# ("I Saw The Bright Shinies", "The Octopus Project", "English", 5.56, 4.05, "electronic", ["https://www.youtube.com/watch?v=Cf9Q-svKyXs"]),
# ("Cosmic Country Noir", "Stereolab", "English", 4.52, 1.93, "electronic", ["https://www.youtube.com/watch?v=sTt7Hdo25qk"]),
# ("Blueberry Tree Part I", "Husky Rescue", "English", 6.89, 4.02, "electronic", ["https://www.youtube.com/watch?v=_7g-YGj3za4"]),
# ("Jacknuggeted", "Manitoba", "English", 6.14, 3.15, "electronic", ["https://www.youtube.com/watch?v=h9xr802t10o"]),
# ("Sex Dreams and Denim Jeans", "Uffie", "English", 7.13, 3.04, "electronic", ["https://www.youtube.com/watch?v=eSeWsDSbmu8"]),
# ("Fireworks", "Moby", "English", 6.15, 2.85, "electronic", ["https://www.youtube.com/watch?v=9A2QyswKAmI"]),
# ("Streamside", "The Album Leaf", "English", 5.84, 3.04, "electronic", ["https://www.youtube.com/watch?v=1SsvQP-xADg"]),
# ("Tracking Aeroplanes", "The Echelon Effect", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=RZ_Pv1K62TM"]),
# ("So Good to Me", "Chris Malinchak", "English", 6.11, 3.6, "electronic", ["https://www.youtube.com/watch?v=KrMl32cuC2A"]),
# ("You and Me in Time", "Broadcast", "English", 6.34, 3.81, "electronic", ["https://www.youtube.com/watch?v=DVX1kogrFNM"]),
# ("Pauvre Simon", "Sylvain Chauveau", "English", 4.3, 2.81, "electronic", ["https://www.youtube.com/watch?v=BNxuS7B9NUM"]),
# ("Lava", "Air", "English", 6.25, 2.71, "electronic", ["https://www.youtube.com/watch?v=kER_84PSxIM"]),
# ("Song", "Max Richter", "English", 5.21, 3.37, "electronic", ["https://www.youtube.com/watch?v=b_YHE4Sx-08"]),
# ("The 6 Million Dollar Sandwich", "The Dead Texan", "English", 5.94, 3.19, "electronic", ["https://www.youtube.com/watch?v=u6BpUCQuEIo"]),
# ("Contradiction", "Apparat", "English", 3.23, 0.97, "electronic", ["https://www.youtube.com/watch?v=qEHm8o6J2yk"]),
# ("Bi-Pet", "Lali Puna", "English", 6.02, 3.06, "electronic", ["https://www.youtube.com/watch?v=04zE0xa653o"]),
# ("We Played Some Open Chords and Rejoiced, for the Earth Had Circled the Sun Yet Another Year", "A Winged Victory for the Sullen", "English", 3.4, 2.03, "electronic", ["https://www.youtube.com/watch?v=XDvtuEQ1LvE"]),
# ("All the Way...", "Ladytron", "English", 5.17, 2.42, "electronic", ["https://www.youtube.com/watch?v=J-7dQOi_LfM"]),
# ("Rebirth of Slick (Cool Like Dat) - 2005 Digital Remaster", "Digable Planets", "English", 6.04, 3.64, "electronic", ["https://www.youtube.com/watch?v=ni4xrKQQxGY"]),
# ("Flying Over the Dateline", "Moby", "English", 5.08, 3.2, "electronic", ["https://www.youtube.com/watch?v=tTzA9h2B1tE"]),
# ("Cherry Blossom Road", "Bibio", "English", 6.09, 3.02, "electronic", ["https://www.youtube.com/watch?v=sCNkTorxz10"]),
# ("Two Rapid Formations", "Brian Eno", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=9X7kuFqAcFM"]),
# ("A Rising Wind", "Helios", "English", 6.15, 3.43, "electronic", ["https://www.youtube.com/watch?v=RmE7vo3aBko"]),
# ("Drowning In You", "Pascäal", "English", 5.82, 2.6, "electronic", ["https://www.youtube.com/watch?v=lkfn2liuSS0"]),
# ("Remove the Inside", "Belong", "English", 6.09, 3.02, "electronic", ["https://www.youtube.com/watch?v=S4OXfk9QPMk"]),
# ("Notes to You", "Sleep Party People", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=POAC_8bFlII"]),
# ("Einsteigen", "Ellen Allien", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=OtopNp29rao"]),
# ("Shoulder to Hand", "Helios", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=X2aD0gJ5gs4"]),
# ("Caught Between", "Brian Eno", "English", 6.84, 3.12, "electronic", ["https://www.youtube.com/watch?v=XM3igl1Nsz0"]),
# ("Signed I Wish You Well", "Helios", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=lzLCraUwJiU"]),
# ("Sumi-e", "Goldmund", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=Segx00u94Fc"]),
# ("One Hell of a Party (feat. Jarvis Cocker)", "Air", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=IGVkY9a4Dqk"]),
# ("In Heaven", "Helios", "English", 5.75, 3.85, "electronic", ["https://www.youtube.com/watch?v=Em9eY7WhRuY"]),
# ("Strange Light", "Brian Eno", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=ZK1P5P6qzEc"]),
# ("The Making of Grief Point", "Loscil", "English", 4.83, 3.75, "electronic", ["https://www.youtube.com/watch?v=y6u9Vzc-H5M"]),
# ("Degradation", "Hilmar Örn Hilmarsson & Sigur Rós", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=GlHv6PML3G8"]),
# ("Your Eyes Only", "Styrofoam", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=9FexX96lO14"]),
# ("Save Your Neck, Save Your Brother", "I Am Robot and Proud", "English", 6.1, 2.67, "electronic", ["https://www.youtube.com/watch?v=VwRH3a_M__I"]),
# ("Flood", "The American Dollar", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=128DAsCqu3k"]),
# ("Sailor", "Hem", "English", 6.82, 2.91, "electronic", ["https://www.youtube.com/watch?v=xGG3d1oIzjg"]),
# ("Sleepless", "Marconi Union", "English", 5.82, 2.6, "electronic", ["https://www.youtube.com/watch?v=UfcAVejslrU"]),
# ("Repeat It", "Dúné", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=CV9U1nFQA-4"]),
# ("To", "Goldmund", "English", 3.23, 0.97, "electronic", ["https://www.youtube.com/watch?v=-rDJfevwaxU"]),
# ("Oh My Love", "Martin L. Gore", "English", 7.52, 4.53, "electronic", ["https://www.youtube.com/watch?v=s51g-nCM4u8"]),
# ("Beautiful Night", "The Go Find", "English", 5.0, 2.73, "electronic", ["https://www.youtube.com/watch?v=bqIxCtEveG8"]),
# ("Slow Motion Suicide", "Voodoo Child", "English", 6.7, 2.98, "electronic", ["https://www.youtube.com/watch?v=WVaj178g4hg"]),
# ("Breather", "Laika", "English", 6.57, 3.65, "electronic", ["https://www.youtube.com/watch?v=b4hQ8L5qSUg"]),
# ("Btwn You + Me", "Windy & Carl", "English", 6.09, 3.02, "electronic", ["https://www.youtube.com/watch?v=i1VxTGw0gSY"]),
# ("Swim", "Message To Bears", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=Hspn219v0rs"]),
# ("Drum Machines and Glockenspiels", "Fridge", "English", 6.09, 3.02, "electronic", ["https://www.youtube.com/watch?v=6Issj3ji-2E"]),
# ("In The Dolphin Tank", "Port Blue", "English", 3.23, 0.97, "electronic", ["https://www.youtube.com/watch?v=NR5YFts96W4"]),
# ("This Place in Time", "Colleen", "English", 7.32, 4.1, "electronic", ["https://www.youtube.com/watch?v=fhgK4FuPyn0"]),
# ("Living Room", "Emeralds", "English", 5.96, 3.01, "electronic", ["https://www.youtube.com/watch?v=dHn0UVK02IM"]),
# ("Risky Nap Under Blue Tree", "Cell", "English", 6.03, 3.23, "electronic", ["https://www.youtube.com/watch?v=LPCYWWLWcPw"]),
# ("5.00 AM", "Kwamie Liv", "English", 6.68, 1.81, "electronic", [""]),
# ("The Axial Catwalk", "Port Blue", "English", 7.48, 4.2, "electronic", ["https://www.youtube.com/watch?v=637mKc54lUc"]),
# ("Tungsten", "Esmerine", "English", 4.75, 2.03, "electronic", ["https://www.youtube.com/watch?v=yuKtfIclS9c"]),
# ("On the Farm", "Panda Bear", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=bUTjxoKVsN8"]),
# ("For Those Who Hesitate", "Brian McBride", "English", 6.47, 1.95, "electronic", ["https://www.youtube.com/watch?v=SqSVw7zVbD8"]),
# ("I Love To Hurt (You Love To Be Hurt)", "Primal Scream", "English", 4.53, 4.94, "electronic", ["https://www.youtube.com/watch?v=MwL730ipovo"]),
# ("Selfish Girls Stay Thin", "FakeSensations", "English", 5.77, 4.04, "electronic", ["https://www.youtube.com/watch?v=dmJ2nhhvovo"]),
# ("Amplifier", "Male Or Female", "English", 4.19, 4.43, "electronic", ["https://www.youtube.com/watch?v=_iireQvRJHo"]),
# ("Tanner Than Me", "Nicola Foti", "English", 3.3, 4.78, "electronic", ["https://www.youtube.com/watch?v=-BupZVOpRpw"]),
# ("y", "Takahiro Kido", "English", 6.44, 4.12, "electronic", ["https://www.youtube.com/watch?v=O4joCkz07Ig"]),
# ("Ne connissons", "Kazumasa Hashimoto", "English", 7.16, 4.14, "electronic", ["https://www.youtube.com/watch?v=-Pv3qVRtCD0"]),
# ("I used to talk to you all the time (even though I was alone)", "Doc & Lena Selyanina", "English", 4.48, 2.36, "electronic", ["https://www.youtube.com/watch?v=hP5gzEVUW0s"]),
# ("And if I have no love (I will be nothing)", "Doc & Lena Selyanina", "English", 3.99, 2.16, "electronic", ["https://www.youtube.com/watch?v=uY-tRrsoKrw"]),
# ("That fine line from your ear to your chin (is not so obvious anymore)", "Doc & Lena Selyanina", "English", 4.48, 2.36, "electronic", ["https://www.youtube.com/watch?v=dRUxBVgP98g"]),
# ("As soon as I close my eyes (I cant recall what my face is like)", "Doc & Lena Selyanina", "English", 3.99, 3.06, "electronic", ["https://www.youtube.com/watch?v=tj7Z_ongVTg"]),
# ("Stoneship Age - Sirrus' Theme", "Robyn Miller", "English", 6.92, 4.25, "electronic", ["https://www.youtube.com/watch?v=Jn83y0DTgG0"]),
# ("The Song of Songs - Song No. 2", "Alexander Goldscheider", "English", 6.49, 4.48, "electronic", ["https://www.youtube.com/watch?v=h-4jkCIdm50"]),
# ("Rainslaight", "Bola", "English", 5.97, 3.61, "electronic", ["https://www.youtube.com/watch?v=2T-8ORBZFPY"]),
# ("Last Inhale", "Tapage", "English", 4.3, 3.34, "electronic", ["https://www.youtube.com/watch?v=zrp18b1nxKk"]),
# ("EU8: 02 Transparent", "Ethereal Universe", "English", 5.37, 3.45, "electronic", ["https://www.youtube.com/watch?v=zqmwUIuJ8hE"]),
# ("Unfold", "Message To Bears", "English", 3.34, 1.41, "electronic", ["https://www.youtube.com/watch?v=pVW0LrnU2hw"]),


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
