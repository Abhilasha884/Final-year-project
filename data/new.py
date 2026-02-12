from yt_dlp import YoutubeDL

songs = [

("Shook Ones Pt. II","Mobb Deep",0.48,0.92),
("Survival of the Fittest","Mobb Deep",0.46,0.90),
("Quiet Storm","Mobb Deep",0.52,0.86),
("Hell on Earth","Mobb Deep",0.44,0.91),
("Temperature's Rising","Mobb Deep",0.50,0.80),

("Respiration","Black Star",0.58,0.82),
("Definition","Black Star",0.66,0.88),
("Thieves in the Night","Black Star",0.52,0.84),
("Brown Skin Lady","Black Star",0.72,0.76),
("Re:Definition","Black Star",0.70,0.89),

("Ms. Fat Booty","Mos Def",0.74,0.82),
("Mathematics","Mos Def",0.60,0.88),
("Hip Hop","Mos Def",0.68,0.86),
("Umi Says","Mos Def",0.80,0.70),
("Auditorium","Mos Def",0.58,0.84),


("They Reminisce Over You","Pete Rock & CL Smooth",0.78,0.82),
("Straighten It Out","Pete Rock & CL Smooth",0.74,0.80),
("Lots of Lovin","Pete Rock & CL Smooth",0.82,0.78),
("Take You There","Pete Rock & CL Smooth",0.76,0.84),
("All Souled Out","Pete Rock & CL Smooth",0.70,0.79),

("Runnin","The Pharcyde",0.72,0.82),
("Passin Me By","The Pharcyde",0.80,0.76),
("Drop","The Pharcyde",0.68,0.88),
("Otha Fish","The Pharcyde",0.78,0.80),
("Ya Mama","The Pharcyde",0.74,0.85),

("93 'til Infinity","Souls of Mischief",0.84,0.82),
("That's When Ya Lost","Souls of Mischief",0.72,0.86),
("Never No More","Souls of Mischief",0.70,0.84),
("Live and Let Live","Souls of Mischief",0.76,0.80),
("Make Your Mind Up","Souls of Mischief",0.74,0.82),

("Award Tour","A Tribe Called Quest",0.86,0.84),
("Electric Relaxation","A Tribe Called Quest",0.88,0.78),
("Can I Kick It?","A Tribe Called Quest",0.92,0.76),
("Check the Rhime","A Tribe Called Quest",0.84,0.88),
("Bonita Applebum","A Tribe Called Quest",0.90,0.70),
("Scenario","A Tribe Called Quest",0.82,0.92),
("Jazz (We've Got)","A Tribe Called Quest",0.86,0.80),
("Buggin Out","A Tribe Called Quest",0.78,0.88),
("Oh My God","A Tribe Called Quest",0.74,0.86),
("Find a Way","A Tribe Called Quest",0.88,0.82),


("The Choice Is Yours","Black Sheep",0.82,0.86),
("Flavor of the Month","Black Sheep",0.78,0.84),
("Strobelite Honey","Black Sheep",0.80,0.82),
("Similak Child","Black Sheep",0.74,0.88),
("Try Counting Sheep","Black Sheep",0.72,0.83),

("Step into a World","KRS-One",0.76,0.90),
("MC's Act Like They Don't Know","KRS-One",0.70,0.92),
("Sound of da Police","KRS-One",0.60,0.94),

("My Philosophy","Boogie Down Productions",0.68,0.88),
("South Bronx","Boogie Down Productions",0.64,0.90),

("Eric B. Is President","Eric B. & Rakim",0.70,0.86),
("Paid in Full","Eric B. & Rakim",0.74,0.88),
("I Ain't No Joke","Eric B. & Rakim",0.68,0.90),
("Follow the Leader","Eric B. & Rakim",0.72,0.91),
("Don't Sweat the Technique","Eric B. & Rakim",0.78,0.92),

("Just a Friend","Biz Markie",0.92,0.80),
("Vapors","Biz Markie",0.84,0.78),
("Nobody Beats the Biz","Biz Markie",0.86,0.84),
("Make the Music With Your Mouth","Biz Markie",0.88,0.82),
("Pickin' Boogers","Biz Markie",0.80,0.86),

("La Di Da Di","Doug E. Fresh & Slick Rick",0.94,0.82),
("The Show","Doug E. Fresh",0.92,0.84),

("Children's Story","Slick Rick",0.82,0.80),
("Mona Lisa","Slick Rick",0.88,0.78),
("Teenage Love","Slick Rick",0.86,0.76),


("I Used to Love H.E.R.","Common",0.70,0.78),
("The Light","Common",0.88,0.76),
("Go!","Common",0.82,0.84),
("Testify","Common",0.74,0.88),
("Universal Mind Control","Common",0.86,0.82),

("Kick, Push","Lupe Fiasco",0.84,0.78),
("Daydreamin'","Lupe Fiasco",0.80,0.74),
("Superstar","Lupe Fiasco",0.90,0.82),
("The Show Goes On","Lupe Fiasco",0.92,0.88),
("Battle Scars","Lupe Fiasco",0.76,0.72),

("Reagan","Killer Mike",0.62,0.90),
("Big Beast","Killer Mike",0.70,0.92),
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
