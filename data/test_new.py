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

("Shook Ones Pt. II", "Mobb Deep", "English", 0.48, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=yoYZf-lBF_U"]),
("Survival of the Fittest", "Mobb Deep", "English", 0.46, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=Dz5VzLz67WA"]),
("Quiet Storm", "Mobb Deep", "English", 0.52, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=WAPEjqGbxdE"]),
("Hell on Earth", "Mobb Deep", "English", 0.44, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=-lonWMzBKdU"]),
("Temperature's Rising", "Mobb Deep", "English", 0.5, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=51qX7C3KAXE"]),
("Respiration", "Black Star", "English", 0.58, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=eeTnog5RRQo"]),
("Definition", "Black Star", "English", 0.66, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=EuJaStSL0xM"]),
("Thieves in the Night", "Black Star", "English", 0.52, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=GjxtRehIz2Y"]),
("Brown Skin Lady", "Black Star", "English", 0.72, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=I1zX4zz03Qg"]),
("Re:Definition", "Black Star", "English", 0.7, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=EuJaStSL0xM"]),
("Ms. Fat Booty", "Mos Def", "English", 0.74, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=01yUzXQctcM"]),
("Mathematics", "Mos Def", "English", 0.6, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=8Ir-zFC9nFE"]),
("Hip Hop", "Mos Def", "English", 0.68, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=uq3syNck274"]),
("Umi Says", "Mos Def", "English", 0.8, 0.7, "Hip-Hop", ["https://www.youtube.com/watch?v=vntLKOd9saI"]),
("Auditorium", "Mos Def", "English", 0.58, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=kSQEGE8gKj4"]),
("They Reminisce Over You", "Pete Rock & CL Smooth", "English", 0.78, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=k6mdRv0ZdR8"]),
("Straighten It Out", "Pete Rock & CL Smooth", "English", 0.74, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=xhDb8LoScpI"]),
("Lots of Lovin", "Pete Rock & CL Smooth", "English", 0.82, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=EM-h3OIbbpQ"]),
("Take You There", "Pete Rock & CL Smooth", "English", 0.76, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=ifX1xY1tC3M"]),
("All Souled Out", "Pete Rock & CL Smooth", "English", 0.7, 0.79, "Hip-Hop", ["https://www.youtube.com/watch?v=S9w-_0sbTww"]),
("Runnin", "The Pharcyde", "English", 0.72, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=jQ-RrGCSa2M"]),
("Passin Me By", "The Pharcyde", "English", 0.8, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=a-mAK3uB2_0"]),
("Drop", "The Pharcyde", "English", 0.68, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=wqVsfGQ_1SU"]),
("Otha Fish", "The Pharcyde", "English", 0.78, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=SZk4HK2fY-8"]),
("Ya Mama", "The Pharcyde", "English", 0.74, 0.85, "Hip-Hop", ["https://www.youtube.com/watch?v=lnCeZY6nxjQ"]),
("93 'til Infinity", "Souls of Mischief", "English", 0.84, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=fXJc2NYwHjw"]),
("That's When Ya Lost", "Souls of Mischief", "English", 0.72, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=RVcesMcLHSA"]),
("Never No More", "Souls of Mischief", "English", 0.7, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=ZpI9BQwOvoo"]),
("Live and Let Live", "Souls of Mischief", "English", 0.76, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=72-J-kRR1v4"]),
("Make Your Mind Up", "Souls of Mischief", "English", 0.74, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=zVT2ZIntDDU"]),
("Award Tour", "A Tribe Called Quest", "English", 0.86, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=P800UWoE9xs"]),
("Electric Relaxation", "A Tribe Called Quest", "English", 0.88, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=WHRnvjCkTsw"]),
("Can I Kick It?", "A Tribe Called Quest", "English", 0.92, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=O3pyCGnZzYA"]),
("Check the Rhime", "A Tribe Called Quest", "English", 0.84, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=1QWEPdgS3As"]),
("Bonita Applebum", "A Tribe Called Quest", "English", 0.9, 0.7, "Hip-Hop", ["https://www.youtube.com/watch?v=6xE6ZWwJezg"]),
("Scenario", "A Tribe Called Quest", "English", 0.82, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=Q6TLWqn82J4"]),
("Jazz (We've Got)", "A Tribe Called Quest", "English", 0.86, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=cxN4nKk2cfk"]),
("Buggin Out", "A Tribe Called Quest", "English", 0.78, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=cxN4nKk2cfk"]),
("Oh My God", "A Tribe Called Quest", "English", 0.74, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=OIah18jcJko"]),
("Find a Way", "A Tribe Called Quest", "English", 0.88, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=Ki7EoH31UkM"]),
("The Choice Is Yours", "Black Sheep", "English", 0.82, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=K9F5xcpjDMU"]),
("Flavor of the Month", "Black Sheep", "English", 0.78, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=F01fzPwBwc4"]),
("Strobelite Honey", "Black Sheep", "English", 0.8, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=A_JtkSmw808"]),
("Similak Child", "Black Sheep", "English", 0.74, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=GHy5F0vayl8"]),
("Try Counting Sheep", "Black Sheep", "English", 0.72, 0.83, "Hip-Hop", ["https://www.youtube.com/watch?v=RYmliqe6PqU"]),
("Step into a World", "KRS-One", "English", 0.76, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=xbJxcFyaCpI"]),
("MC's Act Like They Don't Know", "KRS-One", "English", 0.7, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=xnI8JEW7Ty4"]),
("Sound of da Police", "KRS-One", "English", 0.6, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=9ZrAYxWPN6c"]),
("My Philosophy", "Boogie Down Productions", "English", 0.68, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=h1vKOchATXs"]),
("South Bronx", "Boogie Down Productions", "English", 0.64, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=4jPWe8DCLlI"]),
("Eric B. Is President", "Eric B. & Rakim", "English", 0.7, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=0pwoIFyWE1M"]),
("Paid in Full", "Eric B. & Rakim", "English", 0.74, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=E7t8eoA_1jQ"]),
("I Ain't No Joke", "Eric B. & Rakim", "English", 0.68, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=2TN-kDEKxF0"]),
("Follow the Leader", "Eric B. & Rakim", "English", 0.72, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=95gP3m-uBHA"]),
("Don't Sweat the Technique", "Eric B. & Rakim", "English", 0.78, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=6Y1Emb7Jyks"]),
("Just a Friend", "Biz Markie", "English", 0.92, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=9aofoBrFNdg"]),
("Vapors", "Biz Markie", "English", 0.84, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=WpQjN7yQ_8c"]),
("Nobody Beats the Biz", "Biz Markie", "English", 0.86, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=3hbxvsrFLaw"]),
("Make the Music With Your Mouth", "Biz Markie", "English", 0.88, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=ExDae9Hmfr4"]),
("Pickin' Boogers", "Biz Markie", "English", 0.8, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=1LVlRnQhX0c"]),
("La Di Da Di", "Doug E. Fresh & Slick Rick", "English", 0.94, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=taBFnWMSeAc"]),
("The Show", "Doug E. Fresh", "English", 0.92, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=6sGw9GSCiYU"]),
("Children's Story", "Slick Rick", "English", 0.82, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=HjNTu8jdukA"]),
("Mona Lisa", "Slick Rick", "English", 0.88, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=hcYWVCM2RgY"]),
("Teenage Love", "Slick Rick", "English", 0.86, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=5iZasCzxIX8"]),
("I Used to Love H.E.R.", "Common", "English", 0.7, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=TrUERC2Zk64"]),
("The Light", "Common", "English", 0.88, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=OjHX7jf-znA"]),
("Go!", "Common", "English", 0.82, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=YCe1gC5VaW4"]),
("Testify", "Common", "English", 0.74, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=CZRH68Ib1Ko"]),
("Universal Mind Control", "Common", "English", 0.86, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=EC4hkYgdikI"]),
("Kick, Push", "Lupe Fiasco", "English", 0.84, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=Gl83mI69nX4"]),
("Daydreamin'", "Lupe Fiasco", "English", 0.8, 0.74, "Hip-Hop", ["https://www.youtube.com/watch?v=7XOAStfv-v0"]),
("Superstar", "Lupe Fiasco", "English", 0.9, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=hVkBlsgthLg"]),
("The Show Goes On", "Lupe Fiasco", "English", 0.92, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=Rmp6zIr5y4U"]),
("Battle Scars", "Lupe Fiasco", "English", 0.76, 0.72, "Hip-Hop", ["https://www.youtube.com/watch?v=4ka1Lgd3SAI"]),
("Reagan", "Killer Mike", "English", 0.62, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=6lIqNjC1RKU"]),

("Untitled", "Killer Mike", "English", 0.66, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=lNsAfGDkUtk"]),
("Ric Flair", "Killer Mike", "English", 0.78, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=GAN_O9xRkf4"]),
("Burn", "Killer Mike", "English", 0.6, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=wr4v7sA6Wto"]),
("Close Your Eyes", "Run the Jewels", "English", 0.68, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=PkGwI7nGehA"]),
("Legend Has It", "Run the Jewels", "English", 0.72, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=vWaljXUiCaE"]),
("Blockbuster Night Part 1", "Run the Jewels", "English", 0.7, 0.95, "Hip-Hop", ["https://www.youtube.com/watch?v=uuWQyfGa1yI"]),
("Talk to Me", "Run the Jewels", "English", 0.74, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=1b9n0Amr9RI"]),
("Ooh La La", "Run the Jewels", "English", 0.82, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=Sff7Kc77QAY"]),
("Nobody Speak", "DJ Shadow", "English", 0.78, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=NUC2EQvdzmY"]),
("Six Days", "DJ Shadow", "English", 0.76, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=eY-eyZuW_Uk"]),
("Organ Donor", "DJ Shadow", "English", 0.68, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=bfwXxRNVqi4"]),
("Building Steam With a Grain of Salt", "DJ Shadow", "English", 0.72, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=HORLJvUMs08"]),
("Midnight in a Perfect World", "DJ Shadow", "English", 0.74, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=InFbBlpDTfQ"]),
("Protect Ya Neck", "Wu-Tang Clan", "English", 0.64, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=R0IUR4gkPIE"]),
("Method Man", "Wu-Tang Clan", "English", 0.7, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=PEnwXYJcSZc"]),
("C.R.E.A.M.", "Wu-Tang Clan", "English", 0.6, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=PBwAxmrE194"]),
("Triumph", "Wu-Tang Clan", "English", 0.68, 0.96, "Hip-Hop", ["https://www.youtube.com/watch?v=cPRKsKwEdUQ"]),
("Wu-Tang Clan Ain't Nuthing ta F' Wit", "Wu-Tang Clan", "English", 0.72, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=HnOZea4Zgbc"]),
("Shimmy Shimmy Ya", "Ol' Dirty Bastard", "English", 0.78, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=h2zgB93KANE"]),
("Got Your Money", "Ol' Dirty Bastard", "English", 0.82, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=XXTzm5GtqVo"]),
("Brooklyn Zoo", "Ol' Dirty Bastard", "English", 0.74, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=GRblYfKwa88"]),
("Raw Hide", "Ol' Dirty Bastard", "English", 0.68, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=O0bxyJGQVjo"]),
("Return to the 36 Chambers", "Ol' Dirty Bastard", "English", 0.66, 0.91, "Hip-Hop", ["https://www.youtube.com/watch?v=GRblYfKwa88"]),
("Liquid Swords", "GZA", "English", 0.58, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=1YOOxeFr1b8"]),
("Shadowboxin'", "GZA", "English", 0.6, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=5qDhaWqeNMc"]),
("4th Chamber", "GZA", "English", 0.56, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=8vJRnNXOJEk"]),
("Cold World", "GZA", "English", 0.54, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=s48GO6tBm3A"]),
("Duel of the Iron Mic", "GZA", "English", 0.62, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=PJqou7PFTk0"]),
("Glaciers of Ice", "Raekwon", "English", 0.58, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=SuZ-pkIwH8s"]),
("Criminology", "Raekwon", "English", 0.62, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=l9oS0BFhA30"]),
("Knowledge God", "Raekwon", "English", 0.6, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=MzH5_AWqbio"]),
("Ice Water", "Raekwon", "English", 0.64, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=Gj0S3RDzU8g"]),
("Knuckleheadz", "Raekwon", "English", 0.66, 0.89, "Hip-Hop", ["https://www.youtube.com/watch?v=VXuwljCWZMU"]),
("Mighty Healthy", "Ghostface Killah", "English", 0.7, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=KBWXgVdAJiY"]),
("Nutmeg", "Ghostface Killah", "English", 0.68, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=nn_Wy-_hP48"]),
("Ghost Deini", "Ghostface Killah", "English", 0.72, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=7Gewv9-Qot0"]),
("Malcolm", "Ghostface Killah", "English", 0.66, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=OZz8zZ_0iRQ"]),
("Be Easy", "Ghostface Killah", "English", 0.74, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=I3zeGGK0Diw"]),
("B.O.B.", "OutKast", "English", 0.78, 0.98, "Hip-Hop", ["https://www.youtube.com/watch?v=lVehcuJXe6I"]),
("ATLiens", "OutKast", "English", 0.7, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=ktc9hsSfUck"]),
("Elevators (Me & You)", "OutKast", "English", 0.82, 0.74, "Hip-Hop", ["https://www.youtube.com/watch?v=uqB_UVlhlPA"]),
("SpottieOttieDopaliscious", "OutKast", "English", 0.84, 0.72, "Hip-Hop", ["https://www.youtube.com/watch?v=KwhBreZic8I"]),
("Rosa Parks", "OutKast", "English", 0.88, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=drsQLEU0N1Y"]),
("Hey Ya! Remix", "OutKast", "English", 0.92, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=PWgvGjAhvIw"]),
("Player's Ball", "OutKast", "English", 0.86, 0.78, "Hip-Hop", ["https://www.youtube.com/watch?v=vFofKGKlWo4"]),
("Da Art of Storytellin' Pt. 1", "OutKast", "English", 0.8, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=F0tamp9XdqU"]),
("Ms. Jackson Remix", "OutKast", "English", 0.88, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=MYxAiK6VnXw"]),
("So Fresh, So Clean Remix", "OutKast", "English", 0.9, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=-JfEJq56IwI"]),
("Southernplayalisticadillacmuzik", "OutKast", "English", 0.84, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=0yPagRrAgIU"]),
("Git Up, Git Out", "OutKast", "English", 0.76, 0.8, "Hip-Hop", ["https://www.youtube.com/watch?v=CssC-DY4lO8"]),
("Wheelz of Steel", "OutKast", "English", 0.74, 0.82, "Hip-Hop", ["https://www.youtube.com/watch?v=DYjhEJC3F70"]),
("Return of the 'G'", "OutKast", "English", 0.78, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=jhS1nvpjeXk"]),
("Hootie Hoo", "OutKast", "English", 0.86, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=cc5Po252d_U"]),
("Kryptonite", "Big Boi", "English", 0.82, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=_2wsx1onlOE"]),
("Shutterbugg", "Big Boi", "English", 0.88, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=rWsvkW6rKkQ"]),
("Follow Us", "Big Boi", "English", 0.8, 0.88, "Hip-Hop", ["https://www.youtube.com/watch?v=UryzYsFPWu4"]),
("General Patton", "Big Boi", "English", 0.74, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=h_1ivdW8Yj4"]),
("In the A", "Big Boi", "English", 0.78, 0.86, "Hip-Hop", ["https://www.youtube.com/watch?v=ROlKx7PG6sY"]),
("The Way You Move", "Big Boi", "English", 0.92, 0.84, "Hip-Hop", ["https://www.youtube.com/watch?v=xI5NQ-0Ubfs"]),
("Apple of My Eye", "Big Boi", "English", 0.88, 0.76, "Hip-Hop", ["https://www.youtube.com/watch?v=swWT2UcDv2c"]),
("War", "Kendrick Lamar", "English", 0.6, 0.92, "Hip-Hop", ["https://www.youtube.com/watch?v=lQEHOLdWx3I"]),
("Ronald Reagan Era", "Kendrick Lamar", "English", 0.64, 0.9, "Hip-Hop", ["https://www.youtube.com/watch?v=bk-_RTrUEkw"]),
("Rigamortus", "Kendrick Lamar", "English", 0.68, 0.94, "Hip-Hop", ["https://www.youtube.com/watch?v=yh6QxtRpSH8"]),



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
