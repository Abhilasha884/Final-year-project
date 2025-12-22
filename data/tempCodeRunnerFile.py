ydl_opts = {
            #     'format': 'bestaudio/best',
            #     'outtmpl': os.path.splitext(full_path)[0] + ".%(ext)s",
            #     'noplaylist': True,
            #     'geo_bypass': True,
            #     'ignoreerrors': True,
            #     'quiet': False,
            #     'force_ipv4': True,  # ✅ Force IPv4 to avoid CDN issues
            #     'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            #     'http_headers': {
            #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            #         'Accept-Language': 'en-US,en;q=0.9',
            #         'Accept': '*/*',
            #     },
            #     'prefer_ffmpeg': True,
            #     'postprocessors': [{
            #         'key': 'FFmpegExtractAudio',
            #         'preferredcodec': 'mp3',
            #         'preferredquality': '192',
            #     }],
            #     'postprocessor_args': ['-ss', '0', '-t', str(duration)],
            #     'merge_output_format': 'mp3',
            #     'logger': logger,
            # }