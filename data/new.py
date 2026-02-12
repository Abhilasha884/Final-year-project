from yt_dlp import YoutubeDL

songs = [

("Untitled","Killer Mike",0.66,0.86),
("Ric Flair","Killer Mike",0.78,0.88),
("Burn","Killer Mike",0.60,0.91),

("Close Your Eyes","Run the Jewels",0.68,0.94),
("Legend Has It","Run the Jewels",0.72,0.92),
("Blockbuster Night Part 1","Run the Jewels",0.70,0.95),
("Talk to Me","Run the Jewels",0.74,0.90),
("Ooh La La","Run the Jewels",0.82,0.88),

("Nobody Speak","DJ Shadow",0.78,0.89),
("Six Days","DJ Shadow",0.76,0.86),
("Organ Donor","DJ Shadow",0.68,0.84),
("Building Steam With a Grain of Salt","DJ Shadow",0.72,0.80),
("Midnight in a Perfect World","DJ Shadow",0.74,0.78),


("Protect Ya Neck","Wu-Tang Clan",0.64,0.94),
("Method Man","Wu-Tang Clan",0.70,0.90),
("C.R.E.A.M.","Wu-Tang Clan",0.60,0.82),
("Triumph","Wu-Tang Clan",0.68,0.96),
("Wu-Tang Clan Ain't Nuthing ta F' Wit","Wu-Tang Clan",0.72,0.92),

("Shimmy Shimmy Ya","Ol' Dirty Bastard",0.78,0.90),
("Got Your Money","Ol' Dirty Bastard",0.82,0.88),
("Brooklyn Zoo","Ol' Dirty Bastard",0.74,0.92),
("Raw Hide","Ol' Dirty Bastard",0.68,0.89),
("Return to the 36 Chambers","Ol' Dirty Bastard",0.66,0.91),

("Liquid Swords","GZA",0.58,0.90),
("Shadowboxin'","GZA",0.60,0.88),
("4th Chamber","GZA",0.56,0.94),
("Cold World","GZA",0.54,0.86),
("Duel of the Iron Mic","GZA",0.62,0.92),

("Glaciers of Ice","Raekwon",0.58,0.88),
("Criminology","Raekwon",0.62,0.90),
("Knowledge God","Raekwon",0.60,0.86),
("Ice Water","Raekwon",0.64,0.84),
("Knuckleheadz","Raekwon",0.66,0.89),

("Mighty Healthy","Ghostface Killah",0.70,0.92),
("Nutmeg","Ghostface Killah",0.68,0.88),
("Ghost Deini","Ghostface Killah",0.72,0.86),
("Malcolm","Ghostface Killah",0.66,0.84),
("Be Easy","Ghostface Killah",0.74,0.80),


("B.O.B.","OutKast",0.78,0.98),
("ATLiens","OutKast",0.70,0.86),
("Elevators (Me & You)","OutKast",0.82,0.74),
("SpottieOttieDopaliscious","OutKast",0.84,0.72),
("Rosa Parks","OutKast",0.88,0.80),
("Hey Ya! Remix","OutKast",0.92,0.90),
("Player's Ball","OutKast",0.86,0.78),
("Da Art of Storytellin' Pt. 1","OutKast",0.80,0.76),
("Ms. Jackson Remix","OutKast",0.88,0.84),
("So Fresh, So Clean Remix","OutKast",0.90,0.86),
("Southernplayalisticadillacmuzik","OutKast",0.84,0.82),
("Git Up, Git Out","OutKast",0.76,0.80),
("Wheelz of Steel","OutKast",0.74,0.82),
("Return of the 'G'","OutKast",0.78,0.86),
("Hootie Hoo","OutKast",0.86,0.90),

("Kryptonite","Big Boi",0.82,0.84),
("Shutterbugg","Big Boi",0.88,0.86),
("Follow Us","Big Boi",0.80,0.88),
("General Patton","Big Boi",0.74,0.90),
("In the A","Big Boi",0.78,0.86),
("The Way You Move","Big Boi",0.92,0.84),
("Apple of My Eye","Big Boi",0.88,0.76),

("War","Kendrick Lamar",0.60,0.92),
("Ronald Reagan Era","Kendrick Lamar",0.64,0.90),
("Rigamortus","Kendrick Lamar",0.68,0.94),


]

ydl_opts = {"quiet": True, "skip_download": True}

def get_link(song, artist):
    query = f"ytsearch1:{song} {artist} official video"
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        return f"https://www.youtube.com/watch?v={info['entries'][0]['id']}"

with open("hiphop_links_filled.py", "w", encoding="utf-8") as f:
    for song, artist, v, a in songs:
        link = get_link(song, artist)
        f.write(f'("{song}", "{artist}", "English", {v}, {a}, "Hip-Hop", ["{link}"]),\n')
