import cv2
import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO
# import pyaudiowpatch as pyaudio
# import openwakeword
# from openwakeword.model import Model
import threading
import subprocess
# import msvcrt





# ==========================================
# SARAH TTS
# ==========================================

SARAH_TTS = r"D:\AVAQ\AVAQ_TTS\Scripts\kokoro-tts.exe"


def speak_sarah(text):

    if not text or not text.strip():
        return

    print("\n🔊 Sarah is speaking...")

    try:

        subprocess.run(
            [
                SARAH_TTS,
                "CONIN$",
                "--stream",
                "--lang",
                "en-us",
                "--voice",
                "af_sarah"
            ],
            input=text.strip(),
            text=True,
            check=True
        )

    except Exception as e:

        print(f"Sarah TTS error: {e}")

# ==========================================
# AVAQ V3
# ==========================================

print("========================================")
print("          AVAQ V3 STARTING")
print("========================================")


# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading Whisper...")
whisper = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Loading YOLO26...")
yolo = YOLO("yolo26n.pt")

print("Loading wake-word detector...")

# openwakeword.utils.download_models()

# wake_model = Model(
#     wakeword_models=["hey_jarvis"],
#     inference_framework="onnx"
# )

print("All models loaded!")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)
latest_frame = None
detected_objects = []
running = True

listen_requested = threading.Event()


# ==========================================
# LIVE VISION
# ==========================================

def vision_loop():

    global latest_frame
    global detected_objects
    global running
    cv2.namedWindow("AVAQ V3", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(
    #     "AVAQ V3",
    #     cv2.WND_PROP_FULLSCREEN,
    #     cv2.WINDOW_FULLSCREEN
    # )

    while running:

        success, frame = cap.read()

        if not success:
            continue

        results = yolo(
            frame,
            verbose=False,
            imgsz=640
        )

        result = results[0]

        objects = []

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence >= 0.40:

                class_id = int(box.cls[0])

                name = yolo.names[class_id]

                objects.append(name)

        detected_objects = list(set(objects))

        latest_frame = frame.copy()

        annotated = result.plot()

        cv2.putText(
            annotated,
            "AVAQ - SAY: HEY JARVIS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "AVAQ V3",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            listen_requested.set()

        elif key == ord("q"):
            running = False
            break


# ==========================================
# START CAMERA
# ==========================================

vision_thread = threading.Thread(
    target=vision_loop,
    daemon=True
)

vision_thread.start()


# ==========================================
# WAKE WORD MICROPHONE
# ==========================================

# audio = pyaudio.PyAudio()

# stream = audio.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=16000,
#     input=True,
#     frames_per_buffer=1280
# )


# ==========================================
# FUNCTION: LISTEN FOR QUESTION
# ==========================================

def listen_for_question():

    print("\n🟢 AVAQ AWAKE!")
    print("🎤 Ask your question...")

    samplerate = 16000

    # --------------------------------------
    # Recording settings
    # --------------------------------------

    block_duration = 0.10          # 100 ms
    block_size = int(samplerate * block_duration)

    max_seconds = 8.0              # safety limit
    silence_seconds = 0.8           # stop after this much silence

    # Lower = more sensitive to quiet speech
    speech_threshold = 0.015

    max_blocks = int(max_seconds / block_duration)
    silence_blocks_needed = int(
        silence_seconds / block_duration
    )

    recorded = []

    silence_blocks = 0
    speech_started = False

    print("🎤 Listening...")

    # --------------------------------------
    # Record in small blocks
    # --------------------------------------

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=block_size
    ) as mic:

        for _ in range(max_blocks):

            audio_block, overflowed = mic.read(block_size)

            audio_block = np.squeeze(audio_block)

            recorded.append(audio_block.copy())

            # Measure volume
            volume = np.sqrt(
                np.mean(audio_block ** 2)
            )

            # ----------------------------------
            # Speech detected
            # ----------------------------------

            if volume > speech_threshold:

                speech_started = True
                silence_blocks = 0

            # ----------------------------------
            # Silence after speech
            # ----------------------------------

            elif speech_started:

                silence_blocks += 1

                if silence_blocks >= silence_blocks_needed:

                    break

    # --------------------------------------
    # Combine audio
    # --------------------------------------

    if not recorded:
        return ""

    audio_data = np.concatenate(recorded)

    # Remove a little trailing silence
    if len(audio_data) == 0:
        return ""

    print("🧠 Understanding...")

    # --------------------------------------
    # WHISPER
    # --------------------------------------

    segments, info = whisper.transcribe(
        audio_data,
        language="en",
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500
        },
        beam_size=5,
        condition_on_previous_text=False
    )

    question = ""

    for segment in segments:
        question += segment.text

    question = question.strip()

    print(f"\nYOU:\n{question}")

    return question


# ==========================================
# MAIN WAKE LOOP
# ==========================================

print("\n========================================")
print("          AVAQ IS READY")
print("========================================")
# print('👂 Say "Hey Jarvis" to wake AVAQ.')
print("press spacebar to wake avaq")
print("Press Q on the camera window to quit.")


try:

    while running:

        # ----------------------------------
        # Wait for SPACE
        # ----------------------------------

        if not listen_requested.wait(timeout=0.1):
            continue

        listen_requested.clear()

        if not running:
            break

        print("\n🔥 SPACE PRESSED!")
        print("🎤 AVAQ is listening...")

        question = listen_for_question()

        if not question:
            print("❌ I didn't hear a question.")
            print("👂 Going back to sleep...")

            continue

        print("\nYOU:")
        print(question)

            # ----------------------------------
            # Get current camera frame
            # ----------------------------------

        if latest_frame is None:

            print("No camera frame available.")

            continue

        cv2.imwrite(
                "avaq_live_frame.jpg",
                latest_frame
            )

            # ----------------------------------
            # YOLO information
            # ----------------------------------

        objects_text = ", ".join(
                detected_objects
            )

            # ----------------------------------
            # QWEN
            # ----------------------------------

        print("\n🧠 Qwen is thinking...")
        prompt = f"""
You are AVAQ, a fast local visual assistant.

YOLO detects:
{objects_text}

User:
{question}

Look at the image and answer directly.

Rules:
- Give the answer immediately.
- Maximum 1-2 short sentences.
- No reasoning.
- No explanations unless requested.
- If unsure, say you are unsure.
"""

        response = ollama.chat(
                model="qwen3-vl:2b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [
                            "avaq_live_frame.jpg"
                        ]
                    }
                ]
            )

        answer = response[
                "message"
            ][
                "content"
            ]

        print("\n========================================")
        print("                 AVAQ")
        print("========================================")
        print(answer)
        print("========================================")
            # ----------------------------------
            #  SARAH SPEAKS
            #  ----------------------------------
        speak_sarah(answer)
        print("\n🔴 AVAQ sleeping...")
        # print('👂 Say "Hey Jarvis" again.')

finally:

    running = False

    cap.release()

    cv2.destroyAllWindows()

    print("\nAVAQ stopped.")


# work done 


