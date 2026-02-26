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

("Power Of Love", "Mahavishnu Orchestra", "English", 6.68, 4.37, "Jazz", ["https://www.youtube.com/watch?v=lpeakqSdKd8"]),
("Beyond the Sea (La Mer)", "Django Reinhardt", "English", 7.21, 4.22, "Jazz", ["https://www.youtube.com/watch?v=oRViHTHk33s"]),
("Bei Mir Bist Du Schon", "The Puppini Sisters", "English", 6.68, 4.37, "Jazz", ["https://www.youtube.com/watch?v=N4uWVcq629Q"]),
("Betty et Zorg", "Gabriel Yared", "English", 6.39, 4.69, "Jazz", ["https://www.youtube.com/watch?v=VZye9RX-7lA"]),
("黄昏泣き", "東京事変", "English", 3.34, 1.41, "Jazz", ["https://www.youtube.com/watch?v=9lDBZGHs_3w"]),
("Baltimore", "Nina Simone", "English", 5.07, 3.18, "Jazz", ["https://www.youtube.com/watch?v=ztCgNQg9FCQ"]),
("Together", "Matthew Halsall", "English", 4.49, 2.45, "Jazz", ["https://www.youtube.com/watch?v=kDA-VwBcjC4"]),
("Being Me", "Abbey Lincoln", "English", 5.57, 3.38, "Jazz", ["https://www.youtube.com/watch?v=eHZpA55AFcM"]),
("I Told Jesus", "Roberta Flack", "English", 5.61, 4.28, "Jazz", ["https://www.youtube.com/watch?v=C39ltA1nvUc"]),
("Ending Image", "Arve Henriksen", "English", 4.64, 4.19, "Jazz", ["https://www.youtube.com/watch?v=ycr2BIx6R-s"]),
("Bye Bye Blackbird", "Rahsaan Roland Kirk", "English", 5.99, 3.43, "Jazz", ["https://www.youtube.com/watch?v=Rz9wtsQqAd0"]),
("Piosenka dla Stasia", "Anna Maria Jopek & Friends with Pat Metheny", "English", 5.57, 3.38, "Jazz", ["https://www.youtube.com/watch?v=cO2ju60nMS0"]),
("Sometimes", "Duncan Parsons", "English", 2.79, 1.89, "Jazz", ["https://www.youtube.com/watch?v=CI8G5ppmHVo"]),
("Black City Skyline", "Bohren & der Club of Gore", "English", 3.39, 1.77, "Jazz", ["https://www.youtube.com/watch?v=PtDq2IaObbY"]),
("I'll be seeing you", "Frank Sinatra", "English", 7.34, 4.5, "Jazz", ["https://www.youtube.com/watch?v=f89FStHEoZA"]),
("I'm Old Fashioned", "Chet Baker", "English", 6.54, 3.19, "Jazz", ["https://www.youtube.com/watch?v=xhf5KnzRy1U"]),
("Everybody Loves Somebody", "Frank Sinatra", "English", 7.13, 4.16, "Jazz", ["https://www.youtube.com/watch?v=bl2DzNG-haU"]),
("The Man I Love", "Dinah Shore", "English", 6.94, 4.04, "Jazz", ["https://www.youtube.com/watch?v=Hkwcq896wz4"]),
("Autumn in New York", "Jo Stafford", "English", 6.64, 3.27, "Jazz", ["https://www.youtube.com/watch?v=WpL8bQDY_eE"]),
("La Rosita", "Coleman Hawkins", "English", 6.94, 4.04, "Jazz", ["https://www.youtube.com/watch?v=zLK2lbRi828"]),
("Bless You", "The Ink Spots", "English", 7.05, 4.67, "Jazz", ["https://www.youtube.com/watch?v=xTcD5JPTd64"]),
("Att angöra en brygga", "Monica Zetterlund", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=wySjem8_AA8"]),
("Any Old Time", "Artie Shaw", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=iTPzazpdjFU"]),
("Ev'ry Time We Say Goodbye", "John Coltrane", "English", 6.98, 3.71, "Jazz", ["https://www.youtube.com/watch?v=By88wMU1pIQ"]),
("Don't Sit Under the Apple Tree (With Anyone Else but Me)", "The Andrews Sisters", "English", 7.02, 3.54, "Jazz", ["https://www.youtube.com/watch?v=YcyiC79l910"]),
("Nightbird", "Eva Cassidy", "English", 5.28, 2.16, "Jazz", ["https://www.youtube.com/watch?v=K-X1g-aEeNc"]),
("Mad About the Boy", "Lena Horne", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=VDiMpPiYFfE"]),
("Killing Me Softly With Her Song", "Perry Como", "English", 7.59, 4.97, "Jazz", ["https://www.youtube.com/watch?v=V4iTdMyL_9s"]),
("Le Cinéma", "Claude Nougaro", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=mmv16X-nT7k"]),
("How Much Is That Doggie In The Window", "Patti Page", "English", 7.57, 5.04, "Jazz", ["https://www.youtube.com/watch?v=Aqwq4AgMiik"]),
("Everybody Loves My Baby", "The Boswell Sisters", "English", 7.32, 4.63, "Jazz", ["https://www.youtube.com/watch?v=ltUzHJJ3aAo"]),
("Medley: Jack & Neal/California, Here I Come", "Tom Waits", "English", 4.55, 3.56, "Jazz", ["https://www.youtube.com/watch?v=oJMV5OmDJNM"]),
("Night and Day", "Frank Sinatra & Tommy Dorsey", "English", 7.58, 4.83, "Jazz", ["https://www.youtube.com/watch?v=lGC4JEal8JU"]),
("Blue Moon", "Billy Eckstine", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=aSsYV930O10"]),
("新不了情", "萬芳", "English", 7.42, 4.56, "Jazz", ["https://www.youtube.com/watch?v=JBg21ECQe6Y"]),
("My Blue Heaven", "Artie Shaw", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=FavAwYgnxTE"]),
("How Deep Is The Ocean", "John Coltrane", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=91DGhON6C6Q"]),
("Night And Day", "Karrin Allyson", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=IpUrWXocyFI"]),
("Easy to Remember", "Billie Holiday", "English", 5.09, 3.24, "Jazz", ["https://www.youtube.com/watch?v=t7nQOMGFjYQ"]),
("Harvest Time", "Herbie Hancock", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=dZQCSE2E9AA"]),
("I'm nobody's baby", "Mildred Bailey", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=ldPFLD1VuhI"]),
("Where Or When", "George Michael", "English", 7.21, 4.03, "Jazz", ["https://www.youtube.com/watch?v=eFkq36Tnerk"]),
("After You've Gone", "Marion Harris", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=VW33oH_EkW4"]),
("Farewell Blues", "Paul Whiteman", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=uxJHcI4NXVU"]),
("Come Rain Or Come Shine", "Art Pepper", "English", 5.51, 3.33, "Jazz", ["https://www.youtube.com/watch?v=XhEm1inEzOs"]),
("And I Love Her", "Harry Connick, Jr.", "English", 6.27, 2.95, "Jazz", ["https://www.youtube.com/watch?v=tKkur4ULk4g"]),
("A Tree in the Meadow", "Margaret Whiting", "English", 5.49, 3.78, "Jazz", ["https://www.youtube.com/watch?v=w7zn1r_bvaU"]),
("Crazy She Calls Me", "Chet Baker", "English", 6.59, 3.06, "Jazz", ["https://www.youtube.com/watch?v=ukKk1p2jDSg"]),
("I Like The Sunrise", "Amel Larrieux", "English", 5.93, 3.05, "Jazz", ["https://www.youtube.com/watch?v=EmgcVXM0rTo"]),
("Zeigefinger", "Bohren & der Club of Gore", "English", 4.3, 4.72, "Jazz", ["https://www.youtube.com/watch?v=dtP0cWaZgE8"]),
("Body and Soul (Live)", "Carmen McRae", "English", 2.81, 4.42, "Jazz", ["https://www.youtube.com/watch?v=Kcidz8Vgz-0"]),
("Willow Weep for Me", "Nina Simone", "English", 2.49, 2.75, "Jazz", ["https://www.youtube.com/watch?v=o3-tbry6kiM"]),
("Low Down Man", "Squirrel Nut Zippers", "English", 3.14, 3.18, "Jazz", ["https://www.youtube.com/watch?v=6L_DYlfRsOk"]),
("Can I Keep Him?", "Anja Garbarek", "English", 6.12, 2.59, "Jazz", ["https://www.youtube.com/watch?v=mFnN9xbgWPg"]),
("Days of Wine and Roses", "Henry Mancini", "English", 4.39, 1.85, "Jazz", ["https://www.youtube.com/watch?v=bY55VayUW9I"]),
("Trav'lin Light", "Billie Holiday", "English", 3.34, 1.41, "Jazz", ["https://www.youtube.com/watch?v=CnYINIgIvH0"]),
("A Great Effect", "Simon & Garfunkel", "English", 3.71, 3.4, "Jazz", ["https://www.youtube.com/watch?v=qQ7ZeoD4s_4"]),
("Turiya and Ramakrishna", "Alice Coltrane", "English", 4.31, 2.2, "Jazz", ["https://www.youtube.com/watch?v=QUMuDWDVd20"]),
("Ten Cents A Dance", "Ruth Etting", "English", 1.05, 1.75, "Jazz", ["https://www.youtube.com/watch?v=6k4E9bPpMXE"]),
("Lonelytown", "Paula Cole", "English", 1.18, 2.04, "Jazz", ["https://www.youtube.com/watch?v=fkXpiZ4U3uI"]),
("The Waltz", "Silje Nergaard", "English", 3.68, 2.24, "Jazz", ["https://www.youtube.com/watch?v=aX7Ken5H_CQ"]),
("Blues", "Django Reinhardt", "English", 1.72, 1.8, "Jazz", ["https://www.youtube.com/watch?v=aZ308aOOX04"]),
("Love Where Did You Go?", "Dayna Kurtz", "English", 3.05, 2.79, "Jazz", ["https://www.youtube.com/watch?v=yUyViahT2Hs"]),
("Three Coins in a Fountain", "Frank Sinatra", "English", 4.36, 3.19, "Jazz", ["https://www.youtube.com/watch?v=ENUo8K-8X40"]),
("Helsinki-Vantaa", "Ultra Bra", "English", 5.11, 2.86, "Jazz", ["https://www.youtube.com/watch?v=yJSNsNTDh5Q"]),
("Yöllä", "Ultra Bra", "English", 3.56, 1.52, "Jazz", ["https://www.youtube.com/watch?v=L9AE7y87MV0"]),
("Smoke Gets in Your Eyes", "Roxy Music", "English", 5.3, 3.59, "Jazz", ["https://www.youtube.com/watch?v=OqnSVkkgjzw"]),
("These Foolish Things", "Roxy Music", "English", 5.33, 3.47, "Jazz", ["https://www.youtube.com/watch?v=NbSp_xEa3PI"]),
("Inside Each Other", "Emma Salokoski Ensemble", "English", 4.0, 2.68, "Jazz", ["https://www.youtube.com/watch?v=yXdmHpA8BWk"]),
("I Got It Bad And That Ain't Good (+ quote)", "Nina Simone", "English", 3.6, 2.93, "Jazz", ["https://www.youtube.com/watch?v=l3SS4r1HG4g"]),
("the left hand of god", "Charlie Haden Quartet West", "English", 4.23, 2.96, "Jazz", ["https://www.youtube.com/watch?v=lq8AOfYtJks"]),
("God Give Me Strength", "Elvis Costello live with the Metropole Orkest", "English", 4.4, 4.11, "Jazz", ["https://www.youtube.com/watch?v=5LtrjVsrK58"]),
("Journey Of The Heart", "Nnenna Freelon", "English", 3.77, 2.17, "Jazz", ["https://www.youtube.com/watch?v=uMQ2Bmt_bXU"]),
("Casting", "Deep Tree Mantra", "English", 4.84, 2.74, "Jazz", ["https://www.youtube.com/watch?v=ubOpAV2tKo4"]),
("Chasing Pavements (Live)", "Adele", "English", 5.0, 4.17, "Jazz", ["https://www.youtube.com/watch?v=08DjMT-qR9g"]),
("Morning Broadway", "Keith Mansfield", "English", 5.45, 5.37, "Jazz", ["https://www.youtube.com/watch?v=jDZA_-3xfhc"]),
("Chaotica", "The Bambi Molesters", "English", 5.0, 4.17, "Jazz", ["https://www.youtube.com/watch?v=9KDnfMpt4-Q"]),
("Sugar Rum Cherry (Dance of the Sugar-Plum Fairy)", "Duke Ellington", "English", 6.33, 4.86, "Jazz", ["https://www.youtube.com/watch?v=ONknTGUckKc"]),
("Misty Canyon", "Sven Libaek", "English", 5.12, 4.06, "Jazz", ["https://www.youtube.com/watch?v=6iniYD3sryE"]),
("Little Angel", "Bobby Hutcherson", "English", 6.24, 4.0, "Jazz", ["https://www.youtube.com/watch?v=1rhhEmUGeFo"]),
("Quasars", "Sven Libaek", "English", 4.29, 3.15, "Jazz", ["https://www.youtube.com/watch?v=2DFBFaD7quo"]),
("The Bell That Couldn't Jingle", "Burt Bacharach", "English", 4.77, 3.08, "Jazz", ["https://www.youtube.com/watch?v=9_JSkN4mm1s"]),
("And You Know That", "Yellowjackets", "English", 5.0, 4.17, "Jazz", ["https://www.youtube.com/watch?v=Uh8qriMqu2k"]),
("Swingset (Album Version)", "Bob James", "English", 6.32, 2.99, "Jazz", ["https://www.youtube.com/watch?v=4_HQfeobtVg"]),
("Nuova Oprachina", "Oprachina", "English", 5.91, 3.8, "Jazz", ["https://www.youtube.com/watch?v=LG4VvPw5eB4"]),
("Tão Lonely (So Lonely)", "Zuco 103", "English", 5.0, 4.17, "Jazz", [""]),
("Getting Nowhere", "Roy Budd", "English", 5.22, 4.18, "Jazz", ["https://www.youtube.com/watch?v=jrxk3kSGh3g"]),
("Baby, It's Cold Outside", "Al Hirt; Margret Ann", "English", 5.54, 4.18, "Jazz", ["https://www.youtube.com/watch?v=wVfrtmzpOuY"]),
("Wild Blues", "Wild Bill Davis", "English", 5.0, 4.17, "Jazz", ["https://www.youtube.com/watch?v=K5i_08i8MD0"]),
("Guns & Roses", "Paradise Lunch", "English", 6.06, 4.78, "Jazz", ["https://www.youtube.com/watch?v=TGUfqigItDY"]),
("You've Come This Way", "Nancy Priddy", "English", 6.21, 3.68, "Jazz", ["https://www.youtube.com/watch?v=tMel-vIuoqI"]),
("My Children, My Angels", "Dr.John", "English", 4.1, 3.85, "Jazz", ["https://www.youtube.com/watch?v=4678yz18F_o"]),
("Help Yourself", "Amy Winehouse", "English", 6.01, 4.16, "Jazz", ["https://www.youtube.com/watch?v=SJAnQ_Fk7nQ"]),
("You Ain't Gotta Lie (Momma Said)", "Kendrick Lamar", "English", 5.0, 3.21, "Jazz", ["https://www.youtube.com/watch?v=78bSAKX_4vs"]),
("Child Song", "The Cinematic Orchestra", "English", 5.99, 3.16, "Jazz", ["https://www.youtube.com/watch?v=HbBGsmd0sjE"]),
("Moondance", "Michael Bublé", "English", 7.37, 4.74, "Jazz", ["https://www.youtube.com/watch?v=PBCJWJXeFzk"]),
("If The Stars Were Mine", "Melody Gardot", "English", 6.21, 3.17, "Jazz", ["https://www.youtube.com/watch?v=lpXqOrO7mEM"]),
("I've Got You Under My Skin", "Michael Bublé", "English", 6.96, 3.88, "Jazz", ["https://www.youtube.com/watch?v=SMjex7AfTsg"]),
("Didjital Vibrations", "Jamiroquai", "English", 6.23, 3.29, "Jazz", ["https://www.youtube.com/watch?v=TXah7y6XVxg"]),
("It's Gonna Be", "Norah Jones", "English", 4.57, 3.46, "Jazz", ["https://www.youtube.com/watch?v=Oxmn-o-VLbI"]),
("Quando, Quando, Quando", "Michael Bublé", "English", 7.11, 4.12, "Jazz", ["https://www.youtube.com/watch?v=6_q_CyD4X7c"]),
("Blame It on the Moon", "Katie Melua", "English", 4.49, 3.45, "Jazz", ["https://www.youtube.com/watch?v=zkbwNAGh9J8"]),
("Let's Never Stop Falling In Love", "Pink Martini", "English", 6.77, 4.0, "Jazz", ["https://www.youtube.com/watch?v=Ldxn6aq2GCc"]),
("Lover Undercover", "Melody Gardot", "English", 6.46, 2.77, "Jazz", ["https://www.youtube.com/watch?v=yfF2ajflWzU"]),
("Central Park West", "John Coltrane", "English", 6.36, 3.38, "Jazz", ["https://www.youtube.com/watch?v=euE0wyRmyZs"]),
("Roxanne", "George Michael", "English", 6.18, 4.06, "Jazz", ["https://www.youtube.com/watch?v=dC_XfM6JjXc"]),
("How Can You Mend a Broken Heart", "Michael Bublé", "English", 5.85, 3.39, "Jazz", ["https://www.youtube.com/watch?v=oTchQP8su24"]),
("Love Ain't Gonna Let You Down", "Jamie Cullum", "English", 6.95, 3.64, "Jazz", ["https://www.youtube.com/watch?v=Q0gBFvCPi98"]),
("Autumn Nocturne", "Lou Donaldson", "English", 6.77, 3.11, "Jazz", ["https://www.youtube.com/watch?v=CGpMhyP4JrI"]),
("Be Your Girl", "Teedra Moses", "English", 6.44, 3.4, "Jazz", ["https://www.youtube.com/watch?v=1DlXyONrkjc"]),
("Sway (Quien Sera)", "Dean Martin", "English", 6.62, 3.1, "Jazz", ["https://www.youtube.com/watch?v=J76ExFgg0Vs"]),
("Ruby Baby", "Donald Fagen", "English", 6.5, 3.32, "Jazz", ["https://www.youtube.com/watch?v=kbVdRGev6r8"]),
("Maxine", "Donald Fagen", "English", 5.42, 3.48, "Jazz", ["https://www.youtube.com/watch?v=3saecjSNTlM"]),
("Afro (Freestyle skit)", "Erykah Badu", "English", 6.89, 4.33, "Jazz", ["https://www.youtube.com/watch?v=Vw9XqA_BOtA"]),
("Here's To Life", "Shirley Horn", "English", 6.47, 3.49, "Jazz", ["https://www.youtube.com/watch?v=Nksj0BK_MyA"]),
("One Shine", "The Roots", "English", 6.26, 2.97, "Jazz", ["https://www.youtube.com/watch?v=6LVMblknDHc"]),
("All Through The Night", "Ella Fitzgerald", "English", 7.14, 3.88, "Jazz", ["https://www.youtube.com/watch?v=tCTD6wQ45VU"]),
("Miss Riddle", "Boz Scaggs", "English", 6.05, 3.43, "Jazz", ["https://www.youtube.com/watch?v=sKB-3Rugnic"]),
("The Goodbye Look", "Donald Fagen", "English", 6.35, 3.35, "Jazz", ["https://www.youtube.com/watch?v=vATLlSPnGxE"]),
("Bewitched, Bothered, And Bewildered", "Ella Fitzgerald", "English", 6.69, 3.44, "Jazz", ["https://www.youtube.com/watch?v=1fzZ4l2H5-w"]),
("Alone and I", "Herbie Hancock", "English", 6.5, 3.66, "Jazz", ["https://www.youtube.com/watch?v=qbd0oDkSqVQ"]),
("Mr Magic (Through the Smoke)", "Amy Winehouse", "English", 5.87, 3.31, "Jazz", ["https://www.youtube.com/watch?v=ZTKypbCDheU"]),
("The Gardens Of Sampson & Beasley", "Pink Martini", "English", 6.57, 3.17, "Jazz", ["https://www.youtube.com/watch?v=Ln3GQntM0Zc"]),
("Red Baron", "Billy Cobham", "English", 6.4, 4.48, "Jazz", ["https://www.youtube.com/watch?v=bN9Vaml0dZE"]),
("It'll Come", "Belleruche", "English", 6.36, 3.29, "Jazz", ["https://www.youtube.com/watch?v=eSBs44AuKr4"]),
("Walk Between Raindrops", "Donald Fagen", "English", 6.77, 3.22, "Jazz", ["https://www.youtube.com/watch?v=CpJRc4RZPqE"]),
("Abidan", "Electric Masada", "English", 6.28, 4.37, "Jazz", ["https://www.youtube.com/watch?v=swpvDV6Qvik"]),
("Crab Apple Jam", "David Snell", "English", 7.29, 5.3, "Jazz", ["https://www.youtube.com/watch?v=mknnYsVaG20"]),
("Beyond the Stars", "Jimi Tenor", "English", 5.5, 5.04, "Jazz", ["https://www.youtube.com/watch?v=PY7rk1ci2VI"]),
("Nine Feet Underground/Nigel Blows a Tune/Love's a Friend/Make It 76/Dan", "Caravan", "English", 3.34, 1.41, "Jazz", ["https://www.youtube.com/watch?v=shLH4CzM7mY"]),
("This Will Be", "Natalie Cole", "English", 4.31, 3.29, "Jazz", ["https://www.youtube.com/watch?v=sR85q5jmu20"]),
("Midget", "Jaga Jazzist", "English", 4.18, 3.16, "Jazz", ["https://www.youtube.com/watch?v=VSz42s-v59I"]),
("Fake", "The Brand New Heavies", "English", 6.61, 6.04, "Jazz", ["https://www.youtube.com/watch?v=1xHw099a7hM"]),
("Surrender", "The Brand New Heavies", "English", 6.13, 6.29, "Jazz", ["https://www.youtube.com/watch?v=WwaWawamfq0"]),
("Mr. Melody", "Natalie Cole", "English", 4.85, 5.77, "Jazz", ["https://www.youtube.com/watch?v=nQW6YW9ECjk"]),
("Waste My Time", "The Brand New Heavies", "English", 4.85, 5.77, "Jazz", ["https://www.youtube.com/watch?v=_MsiwmTB13k"]),
("How Do You Think", "The Brand New Heavies", "English", 5.83, 4.6, "Jazz", ["https://www.youtube.com/watch?v=DjKqqPLFEvQ"]),
("How We Do This", "The Brand New Heavies", "English", 4.85, 5.77, "Jazz", ["https://www.youtube.com/watch?v=hcJvY2i1Sd8"]),
("It Could Be Me", "The Brand New Heavies", "English", 6.61, 6.04, "Jazz", ["https://www.youtube.com/watch?v=tuowBkeTRU8"]),
("Show Me the Way", "Zap Mama", "English", 4.85, 5.77, "Jazz", ["https://www.youtube.com/watch?v=jDPVXz0x-CE"]),
("I Believe It To My Soul", "Joss Stone", "English", 4.85, 5.77, "Jazz", ["https://www.youtube.com/watch?v=lbKKyMZddgQ"]),
("Jerkoff", "Bourbon Princess", "English", 3.19, 4.08, "Jazz", ["https://www.youtube.com/watch?v=VYckrPBVO8U"]),
("Don't Sleep In The Subway", "Frank Sinatra", "English", 6.6, 4.74, "Jazz", ["https://www.youtube.com/watch?v=l6chOXVl_LQ"]),
("Ex ego", "Możdżer Danielsson Fresco", "English", 6.45, 5.43, "Jazz", ["https://www.youtube.com/watch?v=Ji6J9mdya-s"]),
("Panama", "The Cat Empire", "English", 6.8, 4.23, "Jazz", ["https://www.youtube.com/watch?v=PnGLS0ci61E"]),
("Jive Samba", "Cannonball Adderley", "English", 6.46, 5.41, "Jazz", ["https://www.youtube.com/watch?v=mpkN8nR0EEg"]),
("Nothing", "Nikka Costa", "English", 6.57, 4.11, "Jazz", ["https://www.youtube.com/watch?v=GXxLTNBDVmI"]),
("Heart Shaped Rock", "Berry Weight", "English", 6.83, 5.11, "Jazz", ["https://www.youtube.com/watch?v=ZfQueCLmTsg"]),
("Govinda Jai Jai", "Alice Coltrane", "English", 6.5, 4.23, "Jazz", ["https://www.youtube.com/watch?v=As8vGWnfqBU"]),
("Holding Me Down", "Toby Lightman", "English", 5.69, 4.55, "Jazz", ["https://www.youtube.com/watch?v=MxpFcvIwul4"]),
("Don't Worry 'Bout Me (LP Version)", "John Parr", "English", 6.81, 5.14, "Jazz", ["https://www.youtube.com/watch?v=ayIqaBHmfp4"]),
("This Little Bird", "Allison Crowe", "English", 6.81, 5.14, "Jazz", ["https://www.youtube.com/watch?v=tlLUrfOAxeU"]),
("Jump Did-Le Ba (1994 Remastered)", "Dizzy Gillespie", "English", 7.91, 5.68, "Jazz", ["https://www.youtube.com/watch?v=eo4zqizakLA"]),
("MoFo Heat - Part 4 - Sweltering", "FOWL", "English", 7.45, 5.05, "Jazz", [""]),
("Asta II", "Możdżer Danielsson Fresco", "English", 3.95, 2.54, "Jazz", ["https://www.youtube.com/watch?v=x-tj0lLho0g"]),
("Me jedyne niebo", "A. M. Jopek & Pat Metheny", "English", 5.33, 3.57, "Jazz", ["https://www.youtube.com/watch?v=tC1g7MZmr44"]),
("Summer Sun (feat. Yukimi Nagano)", "Koop", "English", 6.18, 4.56, "Jazz", ["https://www.youtube.com/watch?v=V9A1JLlwUaM"]),
("Bright Nights (feat. Yukimi Nagano)", "Koop", "English", 2.58, 1.62, "Jazz", ["https://www.youtube.com/watch?v=00kRNF8MjLM"]),
("Canela", "Santana", "English", 6.1, 3.09, "Jazz", ["https://www.youtube.com/watch?v=i8yB3fvSpo8"]),
("Che Cossè L'amor", "Vinicio Capossela", "English", 4.24, 3.02, "Jazz", ["https://www.youtube.com/watch?v=OWwqSj_VULQ"]),
("Summer Sun", "Koop (Featuring Yukimi Nagano)", "English", 4.79, 2.71, "Jazz", ["https://www.youtube.com/watch?v=V9A1JLlwUaM"]),
("Bubblebath", "Simply Put", "English", 2.29, 2.41, "Jazz", ["https://www.youtube.com/watch?v=4wfZT74dFEw"]),
("Algiers Stomp", "Preservation Hall Jazz Band", "English", 5.61, 4.12, "Jazz", ["https://www.youtube.com/watch?v=zsLEhmJNIF8"]),
("thoughts of you", "Freddie Fox", "English", 3.41, 1.72, "Jazz", ["https://www.youtube.com/watch?v=uv4oUX16Igk"]),
("This Way", "Acoustic Alchemy", "English", 5.41, 3.48, "Jazz", ["https://www.youtube.com/watch?v=gQBbCWIJ2NY"]),
("That Easy", "Clara Hill", "English", 3.86, 2.0, "Jazz", ["https://www.youtube.com/watch?v=TqovxEQ7bww"]),
("Drummin' Man", "Gene Krupa", "English", 6.84, 5.0, "Jazz", ["https://www.youtube.com/watch?v=B0KidPoeQ_c"]),
("She Could Be Mine", "Dave Grusin", "English", 6.63, 3.88, "Jazz", ["https://www.youtube.com/watch?v=tukBLyPkKvU"]),
("Where Love Shines", "Incognito", "English", 7.66, 5.53, "Jazz", ["https://www.youtube.com/watch?v=lRaNw9ku9q0"]),
("Tobacco Auctioneer", "Raymond Scott", "English", 7.61, 5.66, "Jazz", ["https://www.youtube.com/watch?v=o8HPJkkOq2c"]),
("What A Little Moonlight Can Do", "Emilie-Claire Barlow", "English", 7.61, 5.66, "Jazz", ["https://www.youtube.com/watch?v=vkRUdLeyt1U"]),
("Shadows in the Rain", "Sting and Gil Evans", "English", 6.84, 5.0, "Jazz", ["https://www.youtube.com/watch?v=ho7VsV4ONgY"]),
("I Am A Missile", "The Kingsize Five", "English", 7.61, 5.66, "Jazz", ["https://www.youtube.com/watch?v=9wL-Z71kAvA"]),
("Some Of These Days", "Ted Lewis & His Band, featuring Sophie Tucker", "English", 6.84, 5.0, "Jazz", ["https://www.youtube.com/watch?v=R4eqYlOB3Lg"]),
("Dizzy's Dream", "The Dave Brubeck Quartet", "English", 6.84, 5.0, "Jazz", ["https://www.youtube.com/watch?v=9HcoGqBKC9M"]),
("Rangoon [#][*]", "Cannonball Adderley", "English", 6.84, 5.0, "Jazz", [""]),
("Oomgar", "FOWL", "English", 7.42, 5.18, "Jazz", [""]),
("Completely Consumed (I'm Not There)", "Heroes of Heartache", "English", 6.21, 4.63, "Jazz", ["https://www.youtube.com/watch?v=ikfNYC5hi00"]),
("Is you or is you aint my baby", "Renee Olstead", "English", 5.97, 4.65, "Jazz", ["https://www.youtube.com/watch?v=5nR83LXs828"]),
("Beautiful People", "Mathilde Santing", "English", 6.95, 4.0, "Jazz", ["https://www.youtube.com/watch?v=WBtkC_ImZ7U"]),
("Thando's Groove", "Sibongile Khumalo", "English", 6.95, 4.0, "Jazz", ["https://www.youtube.com/watch?v=OvAJ5_TizHs"]),
("Tropidelico", "The Quantic Soul Orchestra", "English", 6.95, 4.0, "Jazz", ["https://www.youtube.com/watch?v=WK-VI6o6FY8"]),
("Innocence", "Keith Jarrett", "English", 6.55, 4.4, "Jazz", ["https://www.youtube.com/watch?v=WK-oQPa8GMI"]),
("test3", "The Courageous Egg", "English", 6.55, 4.4, "Jazz", ["https://www.youtube.com/watch?v=h6fcK_fRYaI"]),
("Trubbel", "Monica Zetterlund", "English", 6.64, 3.05, "Jazz", ["https://www.youtube.com/watch?v=qhEdtCJjQ6I"]),
("Humpty Dumpty", "Placebo", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=ZE5R4XeW3JU"]),
("Too Young To Go Steady", "Karrin Allyson", "English", 5.55, 4.04, "Jazz", ["https://www.youtube.com/watch?v=KkQMMivWiAA"]),
("You for Me", "Blossom Dearie", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=AwMsA0xyYfA"]),
("I'm Hip", "Blossom Dearie", "English", 7.2, 4.22, "Jazz", ["https://www.youtube.com/watch?v=EMB5CzzWXMQ"]),
("There's More to Life", "Tiago Iorc", "English", 7.08, 4.12, "Jazz", ["https://www.youtube.com/watch?v=1s1VY0YOgiE"]),
("Heart and Soul", "Mel Tormé", "English", 7.63, 4.89, "Jazz", ["https://www.youtube.com/watch?v=dhp6uwj-URg"]),
("That Fascinating Thing", "Squirrel Nut Zippers", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=Wmz5U1IeGUI"]),
("By the Numbers", "John Coltrane", "English", 6.62, 3.17, "Jazz", ["https://www.youtube.com/watch?v=tdGV83fw0QM"]),
("Lose & Win", "Beady Belle", "English", 4.99, 3.56, "Jazz", ["https://www.youtube.com/watch?v=mWePW0Qlm0U"]),
("I Walk a Little Faster", "Blossom Dearie", "English", 6.72, 2.6, "Jazz", ["https://www.youtube.com/watch?v=licSPVntq8g"]),
("Kool Kat Walk", "Julee Cruise", "English", 6.65, 3.75, "Jazz", ["https://www.youtube.com/watch?v=n40kCSYuatg"]),
("Ice", "Jazzanova", "English", 5.86, 3.38, "Jazz", ["https://www.youtube.com/watch?v=LV5TjBbfWd4"]),
("Last Tango for Astor", "Al Di Meola", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=bwJkI69Rwb0"]),
("I've Got A Great Idea", "Harry Connick, Jr.", "English", 6.51, 2.73, "Jazz", ["https://www.youtube.com/watch?v=6dDy-nX5q-c"]),
("All Of You", "Helen Merrill", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=xL7oZJofnU4"]),
("Final Fantasy 7 Fighting (7/8 jazz spiritual) OC ReMix", "Lau", "English", 6.62, 3.17, "Jazz", ["https://www.youtube.com/watch?v=TTLihuIUA30"]),
("Let it be me", "Inger Marie Gundersen", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=Mx2OIzK2fkM"]),
("Song for the Journey", "Tish Hinojosa", "English", 6.92, 3.89, "Jazz", ["https://www.youtube.com/watch?v=oJMiCWqqxXo"]),
("Soiree", "Bill Evans", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=CbxNj9q_jWo"]),
("I See A Different You (Vocals : Yukimi Nagano)", "Koop", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=9zfKCavdE0I"]),
("I'll Wait and Pray (Alternate Take)", "John Coltrane", "English", 5.1, 3.52, "Jazz", ["https://www.youtube.com/watch?v=oX0QYSLPYek"]),
("A Little Bird Told Me", "Evelyn Knight", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=fiKrclcBPXY"]),
("Miles of Blue (Blue Miles)", "Joe Sample", "English", 5.88, 3.12, "Jazz", ["https://www.youtube.com/watch?v=jANsh-yd-nw"]),
("If Only", "Patrice Rushen", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=IQeWSxHVCxg"]),
("En La Tierra Que No Olvida", "Pat Metheny & Brad Mehldau", "English", 6.55, 3.52, "Jazz", ["https://www.youtube.com/watch?v=DFWKYd-BXRg"]),
("Seven Days In Sunny June (Album Version)", "Jamiroquai", "English", 6.83, 3.21, "Jazz", ["https://www.youtube.com/watch?v=FRSH-egVyzk"]),
("Looking At My Feet", "Room Eleven", "English", 5.42, 3.46, "Jazz", ["https://www.youtube.com/watch?v=lhV4QD6bCuE"]),
("The Dance of J.B.Purvis (Pt.1)", "The Grizzly Folk", "English", 4.22, 3.86, "Jazz", [""]),
("Looney Little Tooney", "Willie The Lion Smith", "English", 4.22, 3.86, "Jazz", ["https://www.youtube.com/watch?v=dTjk4EpNEXQ"]),
("The Sun feat. Graham Candy", "Parov Stelar", "English", 7.7, 4.79, "Jazz", ["https://www.youtube.com/watch?v=WTrNsAsjEmY"]),
("CAR 24", "The Seatbelts", "English", 7.96, 5.12, "Jazz", ["https://www.youtube.com/watch?v=aiJgkALci7E"]),
("I Have A Ghost Now What?", "Jaga Jazzist", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=HYeXRWVoh2Y"]),
("Ain't Got No, I Got Life (Nina Simone V Groovefinder Remix)", "Nina Simone", "English", 8.01, 5.35, "Jazz", ["https://www.youtube.com/watch?v=tK3Rxpaff0g"]),
("Fertile Field", "Bobby McFerrin", "English", 7.2, 4.45, "Jazz", ["https://www.youtube.com/watch?v=ahx5JLIbf4Q"]),
("The House I Live In", "Frank Sinatra", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=k2ImC32pXDw"]),
("Scrabbling for Purchase", "Charlie Hunter Trio", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=-9IxeqqtXqI"]),
("Çal Kapımı", "Birsen Tezer", "English", 7.32, 4.23, "Jazz", ["https://www.youtube.com/watch?v=kZF0UhzRm0A"]),
("Wack Wack", "Buddy Rich", "English", 7.83, 4.87, "Jazz", ["https://www.youtube.com/watch?v=llri1g28OuI"]),
("Dikalo", "Manu Dibango", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=VcTzwaxzEjk"]),
("What Game Shall We Play Today", "Chick Corea & Gary Burton", "English", 6.69, 4.52, "Jazz", ["https://www.youtube.com/watch?v=OzFKXDn858Q"]),
("All I'm Good For", "Benny Sings", "English", 7.2, 4.45, "Jazz", ["https://www.youtube.com/watch?v=-LE26RHPjSQ"]),
("What a Wonderful World", "Kenny G", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=CyFEs8e-P4Y"]),
("Metropolis", "Toshiyuki Honda", "English", 7.46, 4.56, "Jazz", ["https://www.youtube.com/watch?v=KdSV4vghJSo"]),
("Karen", "Le Maximum Kouette", "English", 5.6, 4.61, "Jazz", ["https://www.youtube.com/watch?v=QcwHBS8olps"]),
("Don´t Worry Be Happy", "Bobby McFerrin", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=d-diB65scQU"]),
("Pogoda Ducha", "Hanna Banaszak", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=JuNmgSVMuRE"]),
("Don`t Worry, Be Happy", "Bobby McFerrin", "English", 7.55, 4.34, "Jazz", ["https://www.youtube.com/watch?v=d-diB65scQU"]),
("Birdland / Weather Report", "Jaco Pastorius", "English", 7.45, 4.19, "Jazz", ["https://www.youtube.com/watch?v=pqashW66D7o"]),
("It Don't Cost Very Much (Live)", "Mahalia Jackson", "English", 6.92, 4.77, "Jazz", ["https://www.youtube.com/watch?v=lmv3QzdTShE"]),
("Over The Rainbow", "Art Tatum", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=8KnptwcrbcA"]),
("Christina", "Earl Klugh", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=khmPwOPH7TA"]),
("Consuelo's Love Theme", "Chuck Mangione", "English", 7.32, 4.83, "Jazz", ["https://www.youtube.com/watch?v=Hmbtp_UPWCo"]),
("In Her Family", "Pat Metheny", "English", 6.55, 4.24, "Jazz", ["https://www.youtube.com/watch?v=UH_3zAJPtRo"]),
("Passing Strangers", "Sarah Vaughan", "English", 3.99, 3.77, "Jazz", ["https://www.youtube.com/watch?v=Sk9NVZMNyiM"]),
("B'Bye", "Chuck Mangione", "English", 6.55, 4.24, "Jazz", ["https://www.youtube.com/watch?v=GylwcCEo7B0"]),
("An Affair to Remember", "Emile Pandolfi", "English", 7.16, 4.93, "Jazz", ["https://www.youtube.com/watch?v=_2kloZlspmE"]),
("A Song For You", "Christina Aguilera", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=OGuIkCl_CpY"]),
("Your Lonely Heart", "Natalie Cole", "English", 4.24, 3.07, "Jazz", ["https://www.youtube.com/watch?v=j54nrcu1-Qs"]),
("Who Can I Turn to", "Tony Bennett", "English", 5.31, 4.44, "Jazz", ["https://www.youtube.com/watch?v=wO2o8RhtqS0"]),
("C'est La Vie", "Andrzej Zaucha", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=tqXDC-aPTrk"]),
("Who Can I Turn To (When Nobody Needs Me)", "Wynton Marsalis", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=CUu7TDf3iKc"]),
("Misty (Instrumental)", "Erroll Garner", "English", 5.1, 4.09, "Jazz", ["https://www.youtube.com/watch?v=68WqHteqXqI"]),
("View from the Hill", "McCoy Tyner", "English", 3.6, 3.79, "Jazz", ["https://www.youtube.com/watch?v=yfkI03a367U"]),
("Sometimes I'm Happy", "Alberta Hunter", "English", 3.99, 3.77, "Jazz", ["https://www.youtube.com/watch?v=P4mPtKggiOg"]),
("But I Am a Good Girl", "Christina Aguilera", "English", 7.24, 5.27, "Jazz", ["https://www.youtube.com/watch?v=YDPR5EoYqOs"]),
("Pure", "Boney James", "English", 5.98, 3.65, "Jazz", ["https://www.youtube.com/watch?v=LvkxP06d0QM"]),
("De Zee", "Trijntje Oosterhuis", "English", 6.8, 4.05, "Jazz", ["https://www.youtube.com/watch?v=v0GhjmffnZo"]),
("Stone Groove", "Boney James", "English", 6.8, 4.05, "Jazz", ["https://www.youtube.com/watch?v=1adE3fKiOy0"]),
("Body And Soul", "Ruth Cameron", "English", 6.81, 3.74, "Jazz", ["https://www.youtube.com/watch?v=LJ_3CFEZK38"]),
("All About the Weather", "Phoebe Partridge", "English", 7.32, 4.12, "Jazz", ["https://www.youtube.com/watch?v=YykjpeuMNEk"]),
("Pure [Mig's Petalpusher Vocal]", "Blue Six", "English", 6.8, 4.05, "Jazz", ["https://www.youtube.com/watch?v=iP0GJu-T1qY"]),
("Strawberry Fingernails", "Ty Showers", "English", 8.05, 5.55, "Jazz", ["https://www.youtube.com/watch?v=BRUX6jPTZjQ"]),
("Misery( To Lady Day)", "Tony Scott", "English", 6.8, 4.05, "Jazz", ["https://www.youtube.com/watch?v=5UgeYOSa4Ag"]),
("Come Rain or Come Shine", "The Bill Evans Trio", "English", 6.8, 3.99, "Jazz", ["https://www.youtube.com/watch?v=YuuFffOrpGU"]),
("I'm Thru With Love", "Marilyn Monroe", "English", 7.52, 5.16, "Jazz", ["https://www.youtube.com/watch?v=pi2Pxg3TUDE"]),
("(This Is) A Fine Romance", "Marilyn Monroe", "English", 6.95, 5.0, "Jazz", ["https://www.youtube.com/watch?v=Qyz8nvXQjAY"]),
("Sacred", "Amel Larrieux", "English", 6.95, 5.0, "Jazz", ["https://www.youtube.com/watch?v=RbZ2ZRsNC44"]),
("Morgen Kinder wirds was geben", "Christmazz", "English", 6.95, 5.0, "Jazz", ["https://www.youtube.com/watch?v=rEKrqiDQJeE"]),
("The Tree´s Dreams-Revelation 2:7; 22:1,2,14", "Jan Rohrweg", "English", 4.7, 3.86, "Jazz", [""]),
("The Heather On The Hill [from Brigadoon]", "Gene Kelly", "English", 6.95, 5.0, "Jazz", ["https://www.youtube.com/watch?v=FB0kKAPYJPI"]),
("All I want for Christmazz", "Christmazz", "English", 6.95, 5.0, "Jazz", ["https://www.youtube.com/watch?v=yXQViqx6GMY"]),
("Lust For Life", "Vikter Duplaix", "English", 6.5, 4.16, "Jazz", ["https://www.youtube.com/watch?v=4EhNXexNzP0"]),
("Hang Gliding", "Maria Schneider Orchestra", "English", 6.81, 4.33, "Jazz", ["https://www.youtube.com/watch?v=IEJebihIJt0"]),
("Journey Home", "Maria Schneider Orchestra", "English", 6.53, 4.93, "Jazz", ["https://www.youtube.com/watch?v=IEJebihIJt0"]),
("Allégresse", "Maria Schneider Orchestra", "English", 6.98, 4.38, "Jazz", ["https://www.youtube.com/watch?v=iCbGp-eiMeU"]),
("Last Season", "Maria Schneider Orchestra", "English", 6.98, 4.38, "Jazz", ["https://www.youtube.com/watch?v=HaiVCVDd4rY"]),
("Gumba Blue", "Maria Schneider Orchestra", "English", 6.98, 4.38, "Jazz", ["https://www.youtube.com/watch?v=_oEV6ZItxSY"]),
("Wyrgly", "Maria Schneider Orchestra", "English", 6.81, 4.33, "Jazz", ["https://www.youtube.com/watch?v=zSVvxJwYdmQ"]),
("Dance You Monster to My Soft Song", "Maria Schneider Orchestra", "English", 5.26, 4.67, "Jazz", ["https://www.youtube.com/watch?v=9MFmGGak5lU"]),
("Evanescence", "Maria Schneider Orchestra", "English", 5.95, 4.47, "Jazz", ["https://www.youtube.com/watch?v=zSVvxJwYdmQ"]),
("Green Piece", "Maria Schneider Orchestra", "English", 6.98, 4.38, "Jazz", ["https://www.youtube.com/watch?v=FpQfpy_Kwzw"]),
("My Lament", "Maria Schneider Orchestra", "English", 5.58, 4.41, "Jazz", ["https://www.youtube.com/watch?v=-cSRjQfgmbU"]),
("Gush", "Maria Schneider Orchestra", "English", 6.17, 4.54, "Jazz", ["https://www.youtube.com/watch?v=IEJebihIJt0"]),
("El Viento", "Maria Schneider Orchestra", "English", 6.98, 4.38, "Jazz", ["https://www.youtube.com/watch?v=IEJebihIJt0"]),
("Waxwing", "Maria Schneider Orchestra", "English", 7.21, 4.72, "Jazz", ["https://www.youtube.com/watch?v=IEJebihIJt0"]),
("Coming About", "Maria Schneider Orchestra", "English", 6.5, 4.55, "Jazz", ["https://www.youtube.com/watch?v=wGXWkPbtmFI"]),
("Ev'ry Time We Say Goodbye", "Ella Fitzgerald", "English", 6.48, 4.16, "Jazz", ["https://www.youtube.com/watch?v=nP-8dzS1_rM"]),
("Man of the Hour", "Norah Jones", "English", 5.74, 4.28, "Jazz", ["https://www.youtube.com/watch?v=Z_F5xyP6cuE"]),
("Beyond the Son", "Koop", "English", 6.82, 4.1, "Jazz", ["https://www.youtube.com/watch?v=OZ86TR_q_AE"]),
("Hiperbole", "Skalpel", "English", 4.72, 2.62, "Jazz", ["https://www.youtube.com/watch?v=gx2o7tfHV5s"]),
("Something Stupid", "Frank Sinatra", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=0f48fpoSEPU"]),
("Do I Love You?", "Jane Monheit", "English", 6.84, 3.64, "Jazz", ["https://www.youtube.com/watch?v=A-bEjmjINWc"]),
("The Desperate Ones", "Nina Simone", "English", 5.09, 4.19, "Jazz", ["https://www.youtube.com/watch?v=rDyHVxD4IfI"]),
("Ninjazz", "Skalpel", "English", 3.23, 1.61, "Jazz", ["https://www.youtube.com/watch?v=VBbBPE50SeI"]),
("Laura", "Oscar Peterson", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=cCf4xmIhZ3M"]),
("What Would I Do", "Norah Jones", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=ZBseZ6y7hDQ"]),
("Body and Soul", "Lester Young", "English", 6.65, 2.63, "Jazz", ["https://www.youtube.com/watch?v=_WR1k430tB0"]),
("Moonlight in Vermont", "Gerry Mulligan", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=wyByNx-dzWI"]),
("Cherie, I Love You", "Nat King Cole", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=m1gjHBxzx3c"]),
("Colombia", "Anjulie", "English", 6.58, 3.02, "Jazz", ["https://www.youtube.com/watch?v=ive4UNyGSmQ"]),
("Entrance", "Keith Jarrett", "English", 3.23, 1.61, "Jazz", ["https://www.youtube.com/watch?v=As_Ms9F0Fvs"]),
("I Could Make You Care", "Frank Sinatra", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=OM9_15LuLcs"]),
("Stardust", "Glenn Miller & Benny Goodman", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=TplKR2PTZOs"]),
("Sweet Lorraine", "Coleman Hawkins", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=gNzUvOBDOOQ"]),
("Blue Angel", "Terje Rypdal", "English", 4.29, 3.35, "Jazz", ["https://www.youtube.com/watch?v=PnjDoKAyqBE"]),
("Mother's Whistler", "Moondog", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=5qUYwyyzLbA"]),
("I'll Be Home For Christmas", "Anita Baker", "English", 6.82, 3.6, "Jazz", ["https://www.youtube.com/watch?v=_dmRQXdQq1Q"]),
("Doloroso", "Ray Barretto", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=HY9GcidiV7A"]),
("Don't Explain (feat. Damien Rice & Lisa Hannigan)", "Herbie Hancock", "English", 6.67, 3.01, "Jazz", ["https://www.youtube.com/watch?v=52lQ-5I279I"]),
("The Star Spangled Banner (Marsalis/Hornsby) (LP Version)", "Branford Marsalis", "English", 6.82, 3.83, "Jazz", ["https://www.youtube.com/watch?v=EZbAwIj_yLw"]),
("Letter From India", "Al Di Meola", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=zafD98Rc320"]),
("All to Me", "Shoshana Bean", "English", 5.65, 4.34, "Jazz", ["https://www.youtube.com/watch?v=wUnCTaduDMw"]),
("Nova bossa", "Shakatak", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=sEZGCQtFmto"]),
("The Touch Of Your Lips", "Oscar Peterson & Ben Webster", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=IphIHPRn37I"]),
("Appelboompjes", "Denise Jannah", "English", 6.58, 3.35, "Jazz", ["https://www.youtube.com/watch?v=hCpDyPaUim0"]),
("He's My Guy", "marjorie barnes", "English", 6.47, 3.22, "Jazz", ["https://www.youtube.com/watch?v=s4DGqGxzmf4"]),
("Ode", "David Darling & Jan Garbarek", "English", 6.89, 3.9, "Jazz", ["https://www.youtube.com/watch?v=dQvHjuNtc8s"]),
("What Child Is This", "Mahalia Jackson", "English", 7.26, 2.55, "Jazz", ["https://www.youtube.com/watch?v=8VChQIHP_PI"]),
("In This Room", "Leslie Tucker", "English", 5.28, 2.66, "Jazz", ["https://www.youtube.com/watch?v=sKBuMZMl4BE"]),
("Wayfarer", "John Surman", "English", 3.63, 1.27, "Jazz", ["https://www.youtube.com/watch?v=pbvmyCD3BZE"]),
("Sunny Diego", "The Keith Burton Orchestra", "English", 4.6, 2.22, "Jazz", ["https://www.youtube.com/watch?v=k6sLCavzt6M"]),
("Sun", "Tom Rule", "English", 7.48, 3.11, "Jazz", ["https://www.youtube.com/watch?v=ap3j-UGBxSQ"]),
("Paranoid Eyes", "Alex Ryon", "English", 7.26, 2.55, "Jazz", ["https://www.youtube.com/watch?v=E0Pc5hxCOZc"]),
("Wayfearing Stranger", "Jimmy Kelly", "English", 6.98, 4.07, "Jazz", ["https://www.youtube.com/watch?v=uD7FwisOo80"]),
("Rudolph das kleine Renntier", "Michael Schanze", "English", 6.89, 4.58, "Jazz", ["https://www.youtube.com/watch?v=K-ipzlfOoKU"]),
("Wrap Your Troubles in Dreams", "Bing Crosby", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=Gcx-UiaIe00"]),
("International Flight", "David Snell", "English", 7.1, 4.97, "Jazz", ["https://www.youtube.com/watch?v=dKib5wlxm7Q"]),
("With Plenty of Money and You", "Count Basie", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=xKyIeyfJHLg"]),
("Sabre Dance", "Les Baxter", "English", 7.43, 5.15, "Jazz", ["https://www.youtube.com/watch?v=mUQHGpxrz-8"]),
("Bella Símamær", "Björk", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=oqlrSaNYXdk"]),
("(Still) Terminally Ambivalent Over You", "The Real Tuesday Weld", "English", 6.85, 3.78, "Jazz", ["https://www.youtube.com/watch?v=NgOQuLKeX7g"]),
("Button up Your Overcoat", "Ruth Etting", "English", 7.51, 5.54, "Jazz", ["https://www.youtube.com/watch?v=6UliCMEdTFE"]),
("Cement Mixer", "Slim Gaillard", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=ZKdrnTTDTqo"]),
("Trololo Song", "Trololo Man", "English", 7.55, 5.16, "Jazz", ["https://www.youtube.com/watch?v=oavMtUWDBTM"]),
("You're a Mean One, Mr. Grinch", "Asylum Street Spankers", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=vkN9pSiV558"]),
("Flat Foot Floogie", "Slim Gaillard", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=lFVeJ4wHWdQ"]),
("Body & Soul", "Art Tatum", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=b29Plopcr2s"]),
("Sabres", "Eldad Lidor", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=l9dzyK0u0qs"]),
("Van Lingle Mungo", "Dave Frishberg", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=nKzobTlF8fM"]),
("Yep Roc Heresay", "Slim Gaillard", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=sXJBTsTX53U"]),
("Black and Tan Fantasy (1999 Remastered)", "Duke Ellington", "English", 7.14, 5.33, "Jazz", ["https://www.youtube.com/watch?v=uNfKcYR8XNw"]),
("Stone Cold Dead In The Market", "Wilmoth Houdini", "English", 6.65, 4.77, "Jazz", ["https://www.youtube.com/watch?v=VaGJ-JKT_eA"]),
("One Day In My Garden", "Amon Tobin", "English", 5.07, 4.03, "Jazz", ["https://www.youtube.com/watch?v=cG5GFn9lbjM"]),
("Giant", "The Bad Plus", "English", 5.26, 3.3, "Jazz", ["https://www.youtube.com/watch?v=uG7W-OziAVo"]),
("Giffin", "Mr. Scruff", "English", 5.61, 3.5, "Jazz", ["https://www.youtube.com/watch?v=qgBEDKDfYCI"]),
("Bad Boy", "9 Lazy 9", "English", 5.76, 3.58, "Jazz", ["https://www.youtube.com/watch?v=vvoE1OxV4Rw"]),
("Hypnotic", "Boney James", "English", 5.79, 3.87, "Jazz", ["https://www.youtube.com/watch?v=dSluMSkL3dE"]),
("Sankofa", "Hypnotic Brass Ensemble", "English", 5.77, 3.63, "Jazz", ["https://www.youtube.com/watch?v=6PSvMDRq1Bo"]),
("God Must Be a Boogie Man", "Joni Mitchell", "English", 5.96, 3.27, "Jazz", ["https://www.youtube.com/watch?v=ExbC-JmccGA"]),
("The Mirror", "Nostalgia 77", "English", 6.31, 4.43, "Jazz", ["https://www.youtube.com/watch?v=nMJQKbrXyXg"]),
("Space Shipp", "Matthew Shipp", "English", 3.0, 4.61, "Jazz", ["https://www.youtube.com/watch?v=h8NwBt-618Y"]),
("Trying To Live With the Horrific Knowledge Your Hard Earned Tax Dollars Are Being Used to Slaughter Afghani Babies", "FOWL", "English", 3.0, 4.61, "Jazz", ["https://www.youtube.com/watch?v=kJbL5QcTSmI"]),
("New Country", "Jean-Luc Ponty", "English", 3.56, 3.05, "Jazz", ["https://www.youtube.com/watch?v=r2XbRK6a2ew"]),
("Journey Into Love", "Lonnie Liston Smith", "English", 4.45, 1.87, "Jazz", ["https://www.youtube.com/watch?v=rzSawfcKZcs"]),
("Vodka Cola", "Area", "English", 2.65, 3.45, "Jazz", ["https://www.youtube.com/watch?v=U_AHlVwqYSE"]),
("Time and Space", "Hiromi", "English", 4.75, 3.41, "Jazz", ["https://www.youtube.com/watch?v=R9sIVN0bKOU"]),
("Charade", "Kevin Spacey", "English", 4.24, 3.02, "Jazz", ["https://www.youtube.com/watch?v=5bgxv0ct9HM"]),
("Faking Jazz Together", "Connan Mockasin", "English", 3.86, 2.53, "Jazz", ["https://www.youtube.com/watch?v=NcSL-EWMkCo"]),
("The Turnaround", "The Herbaliser", "English", 4.18, 3.16, "Jazz", ["https://www.youtube.com/watch?v=2ehNKhPlkrM"]),
("Doppelganger", "Jaga Jazzist", "English", 2.58, 1.62, "Jazz", ["https://www.youtube.com/watch?v=xMsCjw5uWRA"]),
("The Human Abstract", "David Axelrod", "English", 4.85, 2.79, "Jazz", ["https://www.youtube.com/watch?v=kH20-FW-NAs"]),
("Erophone", "Pluto Project", "English", 4.47, 3.7, "Jazz", ["https://www.youtube.com/watch?v=oFQEa8GhomQ"]),
("Pacific(Grooverider remix)", "808 State", "English", 6.76, 5.0, "Jazz", ["https://www.youtube.com/watch?v=nCnPUqsT_4I"]),
("Stwisted", "Edie Brickell and New Bohemians", "English", 5.89, 4.7, "Jazz", ["https://www.youtube.com/watch?v=1KhAiR3gKl8"]),
("Black And Brown Cherries", "Abdullah Ibrahim", "English", 3.49, 1.96, "Jazz", ["https://www.youtube.com/watch?v=YpyITLiE5bs"]),
("Music Is Through", "Jamie Cullum", "English", 7.23, 5.74, "Jazz", ["https://www.youtube.com/watch?v=CgI6OEJ5NmY"]),
("Suomi Finland", "Jaga Jazzist", "English", 7.23, 5.2, "Jazz", ["https://www.youtube.com/watch?v=ojGtLHmqmHc"]),
("Rush", "The Seatbelts", "English", 7.43, 5.07, "Jazz", ["https://www.youtube.com/watch?v=XtxsYF4e3uM"]),
("Road Of Bones", "Acoustic Ladyland", "English", 6.21, 5.25, "Jazz", ["https://www.youtube.com/watch?v=wrv0OA9Y5Q0"]),
("Children Go `Round (Demissenw)", "Dee Dee Bridgewater", "English", 5.69, 5.28, "Jazz", ["https://www.youtube.com/watch?v=8bRALkDciqM"]),
("Mark One", "Tommy Flanagan", "English", 7.95, 6.95, "Jazz", ["https://www.youtube.com/watch?v=xIWdIoPedzA"]),
("Double Talk", "Fats Navarro", "English", 7.95, 6.95, "Jazz", ["https://www.youtube.com/watch?v=pHLgKHDgLR4"]),
("Clusterthing", "Flat Earth Society", "English", 7.95, 6.95, "Jazz", ["https://www.youtube.com/watch?v=J5aA0kULrfs"]),
("Hotter Than That", "Louis Armstrong and His Hot Five", "English", 7.95, 6.95, "Jazz", ["https://www.youtube.com/watch?v=UofL8pD69co"]),
("It's Crazy", "Eddie Harris", "English", 3.98, 3.48, "Jazz", ["https://www.youtube.com/watch?v=cUWyZo2c1fU"]),
("Karla's Changes", "David Sánchez", "English", 6.8, 6.78, "Jazz", ["https://www.youtube.com/watch?v=2x4K47gL3Rc"]),
("Jazz de Luxe", "Earl Fuller's Famous Jazz Band", "English", 7.57, 6.55, "Jazz", ["https://www.youtube.com/watch?v=cRdv7LRHzP4"]),
("Chevdo", "FOWL", "English", 7.95, 6.95, "Jazz", ["https://www.youtube.com/watch?v=u_Gb5Tt3Vck"]),
("Unsquare Dance [1961]", "Dave Brubeck", "English", 7.88, 6.73, "Jazz", ["https://www.youtube.com/watch?v=yLsqZfZ0ijg"]),
("Baby Elephant Walk Java", "Billy May, Al Caiola", "English", 3.34, 1.41, "Jazz", ["https://www.youtube.com/watch?v=Bs_t48E0DuI"]),
("Hambone", "Archie Shepp", "English", 5.7, 5.64, "Jazz", ["https://www.youtube.com/watch?v=Gzw7xTK15ms"]),
("Bettergit in Your Soul", "Charles Mingus", "English", 5.7, 5.64, "Jazz", ["https://www.youtube.com/watch?v=J0FcKOfRgvE"]),
("Classical Gas", "Tommy Emmanuel", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=S33tWZqXhnk"]),
("And She Sang", "The Puppini Sisters", "English", 7.56, 5.1, "Jazz", ["https://www.youtube.com/watch?v=-r_mL_q0tqs"]),
("Weed Whacker", "Béla Fleck and the Flecktones", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=UoLXtj0aCiM"]),
("Captain Caribe", "Earl Klugh", "English", 7.75, 6.21, "Jazz", ["https://www.youtube.com/watch?v=D9exrixG4wQ"]),
("Be-Bop", "Arturo Sandoval", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=317vXKAKdw0"]),
("Trickle Trickle", "The Manhattan Transfer", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=yek157w9VyI"]),
("Oceansong", "Rippingtons", "English", 6.83, 3.94, "Jazz", ["https://www.youtube.com/watch?v=dNc-tWYaWBo"]),
("Maynard Ferguson", "Arturo Sandoval", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=qvAn3z9Y_h4"]),
("The Fruit", "Bud Powell", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=etsKDDJYeMI"]),
("Spin", "Steve Cole", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=JO_mzmSl3R8"]),
("Off to the Races", "Donald Byrd", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=xMz0zmiZGn0"]),
("Tuba Tiger Rag", "Canadian Brass", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=o7IraO35qu8"]),
("Got It Goin' On", "Eric Darius", "English", 7.75, 6.21, "Jazz", ["https://www.youtube.com/watch?v=kPqye_6t4D0"]),
("The Thinker", "George Benson", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=y3q-qLChlDQ"]),
("Billy Boy", "Nelson Rangell", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=BAQmavhoIAQ"]),
("Salt Lick", "Tribal Tech", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=cweKIYnyARM"]),
("Cruise Control", "Special EFX", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=kYNvLHB010w"]),
("Today's Top Story", "Nelson Rangell", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=VWBh1h94m8Q"]),
("Ahma", "Maria Kalaniemi", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=cHgLFSzy2_I"]),
("La Salsa En Mi", "Bernie Williams", "English", 6.14, 4.67, "Jazz", ["https://www.youtube.com/watch?v=IkLqYN4Tb6s"]),
("Armageddon Man (Vocal Jazz feat. Karin Ojehagen)", "Mahoney", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=_nQuvjL9ELw"]),
("Mobimientos Del Alma", "Earl Klugh", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=SGfYepWQLv0"]),
("Saturday Night", "Nelson Rangell", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=UnAEwwlvPsc"]),
("(There's) Always Something There To Remind Me", "Stanley Turrentine", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=gTFIfV9d8bQ"]),
("Anym3n keeps up the conversation", "Anym3n", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=Y7DMyCz3mLE"]),
("Autumn Serenade", "Terry Pollard", "English", 7.12, 6.1, "Jazz", ["https://www.youtube.com/watch?v=S8gLokMLup4"]),
("Step Swing", "Lollo Meier", "English", 6.0, 6.57, "Jazz", ["https://www.youtube.com/watch?v=bFM8vZF73CA"]),
("Emotional Weather Report", "Tom Waits", "English", 3.94, 3.95, "Jazz", ["https://www.youtube.com/watch?v=6c5YNPDp8Ww"]),
("The Dog Song", "Nellie McKay", "English", 7.4, 5.16, "Jazz", ["https://www.youtube.com/watch?v=oIbpxHBmvYY"]),
("Protons, Neutrons, Electrons", "The Cat Empire", "English", 8.08, 5.74, "Jazz", ["https://www.youtube.com/watch?v=JXXpQAIPYOs"]),
("Not Nice", "9 Lazy 9", "English", 5.79, 3.8, "Jazz", ["https://www.youtube.com/watch?v=9MiObOpxemU"]),
("Luktar-Gvendur", "Björk", "English", 6.43, 4.35, "Jazz", ["https://www.youtube.com/watch?v=hTPtIS7HUYE"]),
("Tell It Like You Mean It", "Quantic", "English", 6.09, 3.47, "Jazz", ["https://www.youtube.com/watch?v=YL-gftjarcM"]),
("Ease", "Hanne Hukkelberg", "English", 6.71, 3.44, "Jazz", ["https://www.youtube.com/watch?v=iFsBSC4bHKQ"]),
("Tijuana Gift Shop", "Charles Mingus", "English", 5.56, 3.86, "Jazz", ["https://www.youtube.com/watch?v=QvJqjbZQVrM"]),
("Loveless Love", "Louis Armstrong", "English", 6.45, 4.61, "Jazz", ["https://www.youtube.com/watch?v=4TbibXhS2YE"]),
("Beatle Bones 'n' Smokin' Stones (Portsmouth 1975)", "Captain Beefheart & His Magic Band", "English", 1.72, 1.8, "Jazz", ["https://www.youtube.com/watch?v=QPJfj1BqJCg"]),
("Work It! (Man With a Movie Camera)", "The Cinematic Orchestra", "English", 5.35, 5.83, "Jazz", ["https://www.youtube.com/watch?v=3GyNB4-eN1E"]),
("Kozak", "Alex Bilo & Wit", "English", 5.35, 5.83, "Jazz", [""]),
("Midnight in Moscow", "New Orleans Syncopators", "English", 5.35, 5.83, "Jazz", ["https://www.youtube.com/watch?v=qAKZxEc_Cck"]),
("The Kraken", "Squirrel Nut Zippers", "English", 4.16, 4.88, "Jazz", ["https://www.youtube.com/watch?v=PS2aK_lAKzo"]),
("Melancholy Blues", "Louis Armstrong And His Hot Seven", "English", 3.64, 3.43, "Jazz", ["https://www.youtube.com/watch?v=cnHcdqoQDcY"]),
("Keyhole Blues", "Louis Armstrong And His Hot Seven", "English", 7.19, 6.16, "Jazz", ["https://www.youtube.com/watch?v=dyMk9Qvxrms"]),
("Once in a While", "Louis Armstrong and His Hot Five", "English", 7.19, 6.16, "Jazz", ["https://www.youtube.com/watch?v=3y-d1WN2IbM"]),
("Gommé", "Markus Stockhausen", "English", 7.19, 6.16, "Jazz", ["https://www.youtube.com/watch?v=4mIYMB60PPk"]),
("Thunder Showers", "Ty Showers", "English", 7.17, 4.58, "Jazz", ["https://www.youtube.com/watch?v=PB8yAVnD6MQ"]),
("Gras", "Kammerflimmer Kollektief", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=OBNvfFFYBQA"]),
("Konstant", "Kammerflimmer Kollektief", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=OBNvfFFYBQA"]),
("The World Is My Ashtray", "Deejay Punk-Roc", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=J1X4tYGtQVI"]),
("Jewel", "Jun Miyake", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=xFvp-uXo0V4"]),
("Speedball (Bonus Track)", "Charles Earland", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=qAnWRLtE6Z0"]),
("Darkroom", "Rez", "English", 4.32, 4.8, "Jazz", ["https://www.youtube.com/watch?v=HBxUYrCd7OE"]),
("Precipice", "Tribal Tech", "English", 3.56, 5.51, "Jazz", ["https://www.youtube.com/watch?v=Uqpcb9XN-6g"]),
("El Toro", "Chico Hamilton", "English", 4.0, 5.36, "Jazz", ["https://www.youtube.com/watch?v=Mfvd1n7F-cw"]),
("Silence", "Ty Showers", "English", 3.8, 6.2, "Jazz", ["https://www.youtube.com/watch?v=hkkv85o0l6A"]),
("Louco para chegar o carnaval", "Los Panchos? Sí!", "English", 5.48, 5.24, "Jazz", [""]),
("Ma Fleur", "The Cinematic Orchestra", "English", 6.31, 3.66, "Jazz", ["https://www.youtube.com/watch?v=oUFJJNQGwhk"]),
("December", "Norah Jones", "English", 5.12, 3.4, "Jazz", ["https://www.youtube.com/watch?v=7QLCrJkGP1E"]),
("Lilac Wine", "Katie Melua", "English", 5.31, 3.36, "Jazz", ["https://www.youtube.com/watch?v=r3F7-wCIR0Y"]),
("You Can't Always Get What You Want", "Ituana", "English", 7.43, 4.92, "Jazz", ["https://www.youtube.com/watch?v=Ef9QnZVpVd8"]),
("All My Tomorrows", "Frank Sinatra", "English", 6.29, 2.62, "Jazz", ["https://www.youtube.com/watch?v=L-5qMan4cj4"]),
("Sleepless Nights", "Norah Jones", "English", 6.57, 4.11, "Jazz", ["https://www.youtube.com/watch?v=Ld63xp1y-cI"]),
("I Can See Clearly Now", "Holly Cole", "English", 7.68, 4.61, "Jazz", ["https://www.youtube.com/watch?v=XyjzyHp3Wbk"]),
("I Don't Miss You Anymore", "Lisa Ekdahl", "English", 6.12, 3.4, "Jazz", ["https://www.youtube.com/watch?v=1AHqXSVdXXE"]),
("Dear Heart", "Jack Jones", "English", 7.33, 4.27, "Jazz", ["https://www.youtube.com/watch?v=3ZHsYCUc4t8"]),
("It's Raining Somewhere Else", "Toby Fox", "English", 7.34, 3.6, "Jazz", ["https://www.youtube.com/watch?v=KtC-pl9P3kE"]),
("Blue Cafe", "The Style Council", "English", 7.04, 3.58, "Jazz", ["https://www.youtube.com/watch?v=rO6EF9hezNI"]),
("Trail Blazer", "Acoustic Alchemy", "English", 7.48, 4.3, "Jazz", ["https://www.youtube.com/watch?v=lC5ukl6kz4o"]),
("Hide & Seek", "Count Basic", "English", 7.1, 4.03, "Jazz", ["https://www.youtube.com/watch?v=pjbiDbqjfU8"]),
("Perfect Day", "Povo", "English", 7.09, 4.64, "Jazz", ["https://www.youtube.com/watch?v=Js_m5mG8xkc"]),
("Do I Worry?", "The Ink Spots", "English", 4.72, 3.26, "Jazz", ["https://www.youtube.com/watch?v=Eo7_9rf4H2U"]),
("Sunny", "Wes Montgomery", "English", 2.35, 1.09, "Jazz", ["https://www.youtube.com/watch?v=h_SS_wI0seQ"]),
("Don't Try This At Home", "Metropolitan Jazz Affair", "English", 5.14, 2.21, "Jazz", ["https://www.youtube.com/watch?v=5U8__yVg79s"]),
("When the Moon Was Blue", "El-P", "English", 3.76, 2.45, "Jazz", ["https://www.youtube.com/watch?v=PaGnDOOzOGo"]),
("Sunday Evening (Sommermeyer)", "JS (Joerg Sommermeyer)", "English", 6.32, 3.18, "Jazz", ["https://www.youtube.com/watch?v=18s5VKtcN0M"]),
("Byzantine Excursion", "Glenn Mauchline", "English", 3.34, 1.41, "Jazz", [""]),
("Blue Water Blues", "Jasmine Brunch", "English", 3.49, 1.84, "Jazz", ["https://www.youtube.com/watch?v=TFaaOcmRzsg"]),
("Bossa Rocka", "The George Benson Quartett", "English", 1.72, 1.08, "Jazz", ["https://www.youtube.com/watch?v=CF1qQSryp2A"]),
("Don't Know Why", "Norah Jones", "English", 5.38, 3.5, "Jazz", ["https://www.youtube.com/watch?v=tO4dxvguQDk"]),
("Come Away with Me", "Norah Jones", "English", 5.63, 3.64, "Jazz", ["https://www.youtube.com/watch?v=lbjZPFBD6JU"]),
("Seven Years", "Norah Jones", "English", 5.02, 3.56, "Jazz", ["https://www.youtube.com/watch?v=ybiHOdO3Noc"]),
("Shoot the Moon", "Norah Jones", "English", 5.21, 3.62, "Jazz", ["https://www.youtube.com/watch?v=jlJX_9FMbp8"]),
("One Flight Down", "Norah Jones", "English", 5.21, 3.6, "Jazz", ["https://www.youtube.com/watch?v=TJBRpu4Z5kc"]),
("Nightingale", "Norah Jones", "English", 5.59, 3.27, "Jazz", ["https://www.youtube.com/watch?v=IP9m2pNFa60"]),
("Those Sweet Words", "Norah Jones", "English", 5.91, 3.72, "Jazz", ["https://www.youtube.com/watch?v=Ii2tVV1saZc"]),
("(There Is) No Greater Love", "Amy Winehouse", "English", 6.15, 4.07, "Jazz", ["https://www.youtube.com/watch?v=Y5PiXSdTXxg"]),
("Chasing Pirates", "Norah Jones", "English", 5.81, 3.74, "Jazz", ["https://www.youtube.com/watch?v=uTxythHY09k"]),
("In The Morning", "Norah Jones", "English", 5.05, 3.43, "Jazz", ["https://www.youtube.com/watch?v=mbxRw9rmL8M"]),
("Thinking About You", "Norah Jones", "English", 5.4, 3.8, "Jazz", ["https://www.youtube.com/watch?v=wE4lnC25fnU"]),
("Sister Moon", "Sting", "English", 6.15, 3.51, "Jazz", ["https://www.youtube.com/watch?v=QZLvDhTtONU"]),
("Tezeta", "Mulatu Astatke", "English", 5.46, 2.94, "Jazz", ["https://www.youtube.com/watch?v=Wy-v-FgiUD8"]),
("The Paris Match", "The Style Council", "English", 5.53, 3.42, "Jazz", ["https://www.youtube.com/watch?v=c6NYXNAcsj8"]),
("If The Stars Were Mine (Orchestral Version)", "Melody Gardot", "English", 6.0, 3.2, "Jazz", ["https://www.youtube.com/watch?v=924SAa5_-3Q"]),
("April in Paris", "Ella Fitzgerald", "English", 4.28, 3.41, "Jazz", ["https://www.youtube.com/watch?v=AZxrvslGt5w"]),
("Look What You're Doin' to Me", "Jazzanova", "English", 6.68, 3.6, "Jazz", ["https://www.youtube.com/watch?v=fSc6R7l4hJI"]),
("I Gotta Right To Sing The Blues", "Louis Armstrong", "English", 5.86, 3.23, "Jazz", ["https://www.youtube.com/watch?v=Rt_Yg__p30Y"]),
("Purple Rain", "Urselle", "English", 7.57, 4.2, "Jazz", ["https://www.youtube.com/watch?v=fp9X_R4TuNw"]),
("Nancy (With The Laughing Face)", "Frank Sinatra", "English", 6.67, 2.84, "Jazz", ["https://www.youtube.com/watch?v=hTdwFYxv_ro"]),
("Under the Bridges of Paris", "Dean Martin", "English", 7.05, 3.64, "Jazz", ["https://www.youtube.com/watch?v=KiAV3q41Ntc"]),
("Just One Of Those Things", "Sarah Vaughan", "English", 7.14, 3.5, "Jazz", ["https://www.youtube.com/watch?v=i2Ht71eln_Q"]),
("Skating in Central Park", "Bill Evans", "English", 6.47, 1.95, "Jazz", ["https://www.youtube.com/watch?v=iRGFx0EyWqM"]),
("Journey into Melody", "Stanley Turrentine", "English", 6.15, 2.68, "Jazz", ["https://www.youtube.com/watch?v=_RWl_3xrrwM"]),
("Knee-Deep in the North Sea", "Portico Quartet", "English", 5.5, 2.83, "Jazz", ["https://www.youtube.com/watch?v=V0RciGGGdaQ"]),
("O Morro Nao Tem Vez", "Stan Getz", "English", 6.52, 3.18, "Jazz", ["https://www.youtube.com/watch?v=yjaKvH-_hdk"]),
("I Hold No Grudge", "Nina Simone", "English", 5.7, 3.45, "Jazz", ["https://www.youtube.com/watch?v=R24tPPzX0l8"]),
("The River", "Keith Jarrett Trio", "English", 5.87, 2.38, "Jazz", ["https://www.youtube.com/watch?v=nC08aDl9C0A"]),
("Give Me That Slow Knowing Smile", "Lisa Ekdahl", "English", 4.95, 2.78, "Jazz", ["https://www.youtube.com/watch?v=p4NSoDfhhO0"]),
("You Take My Breath Away", "Eva Cassidy", "English", 6.92, 3.29, "Jazz", ["https://www.youtube.com/watch?v=QHfxMGEb9iE"]),
("Take Me Home", "Holly Cole", "English", 5.9, 2.15, "Jazz", ["https://www.youtube.com/watch?v=QzZUFEsMb6U"]),
("My Spanish Heart", "Chick Corea", "English", 6.47, 1.95, "Jazz", ["https://www.youtube.com/watch?v=xrGtuXCIBe0"]),
("Cinema Paradiso (Main Theme)", "Charlie Haden & Pat Metheny", "English", 6.65, 3.25, "Jazz", ["https://www.youtube.com/watch?v=6VwPYlQLtnI"]),
("Slow Dance", "John Coltrane", "English", 7.09, 4.0, "Jazz", ["https://www.youtube.com/watch?v=R4rNRkDgyZ4"]),
("Love You Madly", "Ella Fitzgerald", "English", 7.36, 4.14, "Jazz", ["https://www.youtube.com/watch?v=jCoq0AWLPxM"]),
("Where Have I Known You Before", "Return to Forever", "English", 6.47, 1.95, "Jazz", ["https://www.youtube.com/watch?v=u7xDFiyie0o"]),
("Bahia", "Stan Getz", "English", 6.58, 2.38, "Jazz", ["https://www.youtube.com/watch?v=yd0l1wS6VbA"]),
("Travels", "Pat Metheny", "English", 6.7, 2.12, "Jazz", ["https://www.youtube.com/watch?v=QkmP_rT0PPM"]),
("A Tale Begun", "Jan Garbarek", "English", 6.1, 2.55, "Jazz", ["https://www.youtube.com/watch?v=T4U20RqPcKw"]),
("I Will Find the Way", "Pat Metheny", "English", 5.55, 2.0, "Jazz", ["https://www.youtube.com/watch?v=ciCUBqok8hY"]),
("You Are Too Beautiful", "John Coltrane", "English", 6.47, 1.95, "Jazz", ["https://www.youtube.com/watch?v=E3RlMT-Xino"]),
("Everything I've Got", "Blossom Dearie", "English", 3.3, 4.78, "Jazz", ["https://www.youtube.com/watch?v=lvviA97AhAM"]),
("Cynical", "Rufus Jr", "English", 5.44, 3.42, "Jazz", ["https://www.youtube.com/watch?v=y7fudcFIlZs"]),
("Day is Done", "Brad Mehldau Trio", "English", 6.55, 3.48, "Jazz", ["https://www.youtube.com/watch?v=RP9XLElmYK8"]),
("Tribute to Tom", "Nick Elward", "English", 6.66, 4.35, "Jazz", ["https://www.youtube.com/watch?v=_2SUCoXrWpY"]),


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
