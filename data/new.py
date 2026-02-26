from yt_dlp import YoutubeDL

# ---------------- SONG LIST ----------------
songs = [





]

# ---------------- YOUTUBE SEARCH SETTINGS ----------------
ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": True,
    "ignoreerrors": True
}

# ---------------- FUNCTION TO FETCH LINK ----------------
def get_link(song, artist):
    try:
        query = f"ytsearch1:{song} {artist} official video"
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

            if not info or "entries" not in info or not info["entries"]:
                return ""

            video_id = info["entries"][0].get("id")
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

            return ""

    except Exception:
        return ""


# ---------------- WRITE FINAL DATASET FILE ----------------
output_file = "classical_links_filled.py"

with open(output_file, "w", encoding="utf-8") as f:
    for song, artist, valence, arousal in songs:
        link = get_link(song, artist)

        line = f'("{song}", "{artist}", "English", {valence}, {arousal}, "Classical", ["{link}"]),\n'
        f.write(line)

print("✅ Classical dataset with YouTube links generated ->", output_file)
