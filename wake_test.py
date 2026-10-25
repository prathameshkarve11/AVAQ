import pyaudiowpatch as pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

print("Loading wake-word model...")

openwakeword.utils.download_models()

model = Model(
    wakeword_models=["hey_jarvis"]
)

print("================================")
print("       AVAQ WAKE TEST")
print("================================")
print('Say: "Hey Jarvis"')
print("Press Ctrl+C to stop.\n")

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

try:
    while True:

        data = stream.read(
            1280,
            exception_on_overflow=False
        )

        audio_frame = np.frombuffer(
            data,
            dtype=np.int16
        )

        prediction = model.predict(audio_frame)

        score = prediction.get("hey_jarvis", 0)

        if score > 0.5:
            print(f"🔥 WAKE WORD DETECTED! Score: {score:.2f}")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()

print("Done.")