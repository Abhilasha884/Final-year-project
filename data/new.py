from yt_dlp import YoutubeDL

songs = [

("Are You Experienced?","Jimi Hendrix",0.82,0.90),
("Purple Haze","Jimi Hendrix",0.80,0.96),
("Voodoo Child (Slight Return)","Jimi Hendrix",0.84,0.98),
("Hey Joe","Jimi Hendrix",0.86,0.82),
("Little Wing","Jimi Hendrix",0.88,0.76),

("Black Magic Woman","Santana",0.90,0.84),
("Smooth","Santana",0.94,0.88),
("Maria Maria","Santana",0.92,0.82),
("Oye Como Va","Santana",0.96,0.86),
("Europa","Santana",0.88,0.70),

("Born to Be Wild","Steppenwolf",0.92,0.96),
("Magic Carpet Ride","Steppenwolf",0.90,0.88),
("Rock Me","Steppenwolf",0.88,0.90),
("Monster","Steppenwolf",0.82,0.92),
("The Pusher","Steppenwolf",0.80,0.84),

("We Built This City","Starship",0.94,0.86),
("Nothing's Gonna Stop Us Now","Starship",0.96,0.82),
("Sara","Starship",0.88,0.74),
("It's Not Enough","Starship",0.86,0.80),
("Find Your Way Back","Starship",0.90,0.84),

("Centerfold","The J. Geils Band",0.94,0.88),
("Freeze Frame","The J. Geils Band",0.92,0.90),
("Love Stinks","The J. Geils Band",0.88,0.86),
("Give It to Me","The J. Geils Band",0.90,0.88),
("Musta Got Lost","The J. Geils Band",0.86,0.82),

("The Boys Are Back in Town","Thin Lizzy",0.92,0.90),
("Jailbreak","Thin Lizzy",0.88,0.92),
("Whiskey in the Jar","Thin Lizzy",0.90,0.84),
("Rosalie","Thin Lizzy",0.86,0.90),
("Still in Love With You","Thin Lizzy",0.88,0.76),

("Barracuda","Heart",0.90,0.94),
("Crazy on You","Heart",0.92,0.90),
("Alone","Heart",0.88,0.70),
("Magic Man","Heart",0.86,0.88),
("These Dreams","Heart",0.90,0.74),

("More Than Words","Extreme",0.92,0.68),
("Hole Hearted","Extreme",0.90,0.82),
("Get the Funk Out","Extreme",0.88,0.90),
("Decadence Dance","Extreme",0.86,0.92),
("Rest in Peace","Extreme",0.84,0.88),

("Plush (Acoustic)","Stone Temple Pilots",0.84,0.70),
("Creep (Acoustic)","Stone Temple Pilots",0.82,0.66),
("Interstate Love Song (Acoustic)","Stone Temple Pilots",0.86,0.72),
("Big Bang Baby","Stone Temple Pilots",0.88,0.90),
("Sour Girl","Stone Temple Pilots",0.90,0.80),


("Under the Bridge", "Red Hot Chili Peppers", 0.90, 0.74),
("Californication", "Red Hot Chili Peppers", 0.88, 0.78),
("Scar Tissue", "Red Hot Chili Peppers", 0.92, 0.70),
("Otherside", "Red Hot Chili Peppers", 0.84, 0.76),
("Dani California", "Red Hot Chili Peppers", 0.90, 0.88),

("Numb", "Linkin Park", 0.86, 0.82),
("In the End", "Linkin Park", 0.88, 0.90),
("Crawling", "Linkin Park", 0.78, 0.86),
("Somewhere I Belong", "Linkin Park", 0.82, 0.88),
("Breaking the Habit", "Linkin Park", 0.84, 0.80),

("Bring Me to Life (Acoustic)", "Evanescence", 0.80, 0.72),
("Lithium (Acoustic)", "Evanescence", 0.78, 0.68),
("Tourniquet", "Evanescence", 0.72, 0.86),
("Imaginary", "Evanescence", 0.84, 0.78),
("Everybody's Fool", "Evanescence", 0.80, 0.82),

("Take Me Out", "Franz Ferdinand", 0.92, 0.88),
("Do You Want To", "Franz Ferdinand", 0.94, 0.90),
("Walk Away", "Franz Ferdinand", 0.88, 0.82),
("No You Girls", "Franz Ferdinand", 0.90, 0.86),
("Love Illumination", "Franz Ferdinand", 0.92, 0.88),

("Seven Nation Army (Acoustic)", "The White Stripes", 0.86, 0.70),
("Hotel Yorba", "The White Stripes", 0.88, 0.82),
("We're Going to Be Friends", "The White Stripes", 0.90, 0.68),
("Dead Leaves and the Dirty Ground", "The White Stripes", 0.82, 0.90),
("The Denial Twist", "The White Stripes", 0.84, 0.88),

("Howlin' for You", "The Black Keys", 0.92, 0.88),
("Lonely Boy", "The Black Keys", 0.94, 0.90),
("Gold on the Ceiling", "The Black Keys", 0.90, 0.92),
("Tighten Up", "The Black Keys", 0.88, 0.84),
("Little Black Submarines", "The Black Keys", 0.86, 0.80),

("Use Me", "Hozier", 0.82, 0.76),
("Almost (Sweet Music)", "Hozier", 0.88, 0.80),
("From Eden", "Hozier", 0.84, 0.78),
("Angel of Small Death", "Hozier", 0.80, 0.82),
("Cherry Wine", "Hozier", 0.86, 0.66),

("Take Me to Church (Acoustic)", "Hozier", 0.82, 0.70),
("Someone New (Acoustic)", "Hozier", 0.84, 0.68),
("Work Song", "Hozier", 0.86, 0.72),
("Movement (Acoustic)", "Hozier", 0.80, 0.66),
("Dinner & Diatribes", "Hozier", 0.88, 0.86),

("Sex on Fire (Acoustic)", "Kings of Leon", 0.84, 0.70),
("Use Somebody (Live)", "Kings of Leon", 0.88, 0.76),
("Temple (Live)", "Kings of Leon", 0.82, 0.78),
("Pyro (Live)", "Kings of Leon", 0.80, 0.74),
("Closer (Live)", "Kings of Leon", 0.82, 0.76),




]

ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": True,
    "ignoreerrors": True
}

def get_link(song, artist):
    try:
        query = f"ytsearch1:{song} {artist} official video"
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

            if not info or "entries" not in info or not info["entries"]:
                return ""

            vid = info["entries"][0].get("id")
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
            return ""

    except Exception:
        return ""

# generate rock dataset with links
with open("rock_links_filled.py", "w", encoding="utf-8") as f:
    for song, artist, v, a in songs:
        link = get_link(song, artist)
        f.write(f'("{song}", "{artist}", "English", {v}, {a}, "Rock", ["{link}"]),\n')

print("✅ rock_links_filled.py created with YouTube links")
