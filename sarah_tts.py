import subprocess


def speak(text):
    if not text or not text.strip():
        return

    print(f"\nSarah: {text}\n")

    subprocess.run(
        [
            "kokoro-tts",
            "CONIN$",
            "--stream",
            "--lang",
            "en-us",
            "--voice",
            "af_sarah",
        ],
        input=text,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    speak(
        "Hello! I am Sarah, the voice of AVAQ. "
        "I can now speak your answers out loud."
    )