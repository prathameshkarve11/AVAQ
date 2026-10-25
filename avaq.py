import cv2
import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

print("================================")
print("        AVAQ STARTING...")
print("================================")

# -----------------------------
# Load Whisper
# -----------------------------
print("\nLoading Whisper...")

whisper = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# Open webcam
# -----------------------------
print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera ready.")

# -----------------------------
# Listen
# -----------------------------
samplerate = 16000
seconds = 6

print("\n🎤 Speak for 6 seconds...")
print('Try saying: "What do you see?"')

audio = sd.rec(
    int(seconds * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = np.squeeze(audio)

print("🧠 Understanding your question...")

segments, info = whisper.transcribe(
    audio,
    language="en",
    vad_filter=True
)

question = ""

for segment in segments:
    question += segment.text

question = question.strip()

print("\nYOU SAID:")
print(question)

if not question:
    print("I couldn't understand you.")
    cap.release()
    exit()

# -----------------------------
# Capture webcam frame
# -----------------------------
print("\n📷 Looking at the camera...")

success, frame = cap.read()

if not success:
    print("ERROR: Could not capture webcam frame.")
    cap.release()
    exit()

cv2.imwrite("avaq_frame.jpg", frame)

cap.release()

# -----------------------------
# Ask Qwen
# -----------------------------
print("🧠 Asking Qwen3-VL...")

response = ollama.chat(
    model="qwen3-vl:2b",
    messages=[
        {
            "role": "user",
            "content": question,
            "images": ["avaq_frame.jpg"]
        }
    ]
)

answer = response["message"]["content"]

print("\n================================")
print("              AVAQ")
print("================================")
print(answer)
print("================================")