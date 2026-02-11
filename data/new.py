from yt_dlp import YoutubeDL

songs = [

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
