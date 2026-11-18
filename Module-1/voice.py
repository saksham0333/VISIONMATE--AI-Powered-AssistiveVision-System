"""Text-to-speech helper shared by the launcher and the traffic light module.

The text is converted to MP3 with gTTS (Google's online text-to-speech
service, so an internet connection is needed; Section 4.8) and played with
mpg123. The audio file is deleted after it has been played, so spoken text is
not kept on disk.
"""

import os
import shutil
import subprocess
import tempfile

from gtts import gTTS


def voice(myText, language='en'):
    output = gTTS(text=myText, lang=language, slow=False)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        output.save(path)
        if shutil.which("mpg123"):
            subprocess.run(["mpg123", "-q", path], check=False)
        else:  # fallback where mpg123 is not installed (e.g. a development PC)
            from playsound import playsound
            playsound(path)
    finally:
        os.remove(path)
