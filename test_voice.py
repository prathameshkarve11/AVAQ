from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

print("Loading Whisper...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

samplerate = 16000
seconds = 7

print("\nSpeak for 7 seconds...")
print('Say: "Hello AVAQ, what do you see?"')

audio = sd.rec(
    int(seconds * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = np.squeeze(audio)

print("Thinking...")

segments, info = model.transcribe(
    audio,
    language="en",
    vad_filter=True
)

text = ""

for segment in segments:
    text += segment.text

print("\n" + "=" * 40)
print("YOU SAID:")
print(text.strip())
print("=" * 40)