import cv2
import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO
import pyaudiowpatch as pyaudio
import openwakeword
from openwakeword.model import Model
import threading


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

openwakeword.utils.download_models()

wake_model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

print("All models loaded!")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

latest_frame = None
detected_objects = []
running = True


# ==========================================
# LIVE VISION
# ==========================================

def vision_loop():

    global latest_frame
    global detected_objects
    global running

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

        if key == ord("q"):
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

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)


# ==========================================
# FUNCTION: LISTEN FOR QUESTION
# ==========================================

def listen_for_question():

    print("\n🟢 AVAQ AWAKE!")
    print("🎤 Ask your question...")

    samplerate = 16000
    seconds = 6

    audio_data = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio_data = np.squeeze(audio_data)

    print("🧠 Understanding...")

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

    return question.strip()


# ==========================================
# MAIN WAKE LOOP
# ==========================================

print("\n========================================")
print("          AVAQ IS READY")
print("========================================")
print('👂 Say "Hey Jarvis" to wake AVAQ.')
print("Press Q on the camera window to quit.")


try:

    while running:

        # ----------------------------------
        # Listen ONLY for wake word
        # ----------------------------------

        data = stream.read(
            1280,
            exception_on_overflow=False
        )

        audio_frame = np.frombuffer(
            data,
            dtype=np.int16
        )

        prediction = wake_model.predict(
            audio_frame
        )

        score = prediction.get(
            "hey_jarvis",
            0
        )

        # ----------------------------------
        # Wake detected
        # ----------------------------------

        if score >= 0.55:

            print("\n🔥 HEY JARVIS DETECTED!")

            # Reset wake detector so it doesn't
            # immediately trigger again
            wake_model.reset()

            import time
            time.sleep(1.5)



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
You are AVAQ, a local AI visual assistant.

YOLO currently detects:

{objects_text}

The user asks:

{question}

Look carefully at the camera image.

Answer the user's question naturally and briefly.

Use the image as the main source of truth.

If you are uncertain, say so.
"""

            response = ollama.chat(
                model="qwen3-vl:2b",
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

            print("\n🔴 AVAQ sleeping...")
            print('👂 Say "Hey Jarvis" again.')

finally:

    running = False

    stream.stop_stream()
    stream.close()

    audio.terminate()

    cap.release()

    cv2.destroyAllWindows()

    print("\nAVAQ stopped.")