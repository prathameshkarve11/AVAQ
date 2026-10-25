import cv2
import ollama
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO
import threading
import time


# ==========================================
# AVAQ V2
# ==========================================

print("========================================")
print("           AVAQ V2 STARTING")
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

print("Models loaded!")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera ready!")


# ==========================================
# SHARED DATA
# ==========================================

latest_frame = None
detected_objects = []

running = True


# ==========================================
# YOLO CAMERA LOOP
# ==========================================

def vision_loop():

    global latest_frame
    global detected_objects
    global running

    while running:

        success, frame = cap.read()

        if not success:
            continue

        # YOLO detection
        results = yolo(
            frame,
            verbose=False,
            imgsz=640
        )

        result = results[0]

        # Get detected object names
        objects = []

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence >= 0.40:

                class_id = int(box.cls[0])

                name = yolo.names[class_id]

                objects.append(name)

        detected_objects = list(set(objects))

        # Draw detections
        annotated = result.plot()

        latest_frame = frame.copy()

        cv2.putText(
            annotated,
            "AVAQ LIVE VISION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "AVAQ V2",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            running = False
            break


# ==========================================
# START VISION THREAD
# ==========================================

vision_thread = threading.Thread(
    target=vision_loop,
    daemon=True
)

vision_thread.start()


# ==========================================
# VOICE LOOP
# ==========================================

samplerate = 16000
seconds = 6


while running:

    print("\n----------------------------------------")
    print("🎤 Listening...")
    print("Say something like:")
    print("  'What do you see?'")
    print("  'What am I holding?'")
    print("  'How many people are there?'")
    print("----------------------------------------")

    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    print("🧠 Understanding...")

    segments, info = whisper.transcribe(
    audio,
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

    if not question:
        print("Didn't hear anything.")
        continue

    print("\nYOU:")
    print(question)

    # ======================================
    # GET CURRENT FRAME
    # ======================================

    if latest_frame is None:

        print("No camera frame available.")

        continue

    cv2.imwrite(
        "avaq_live_frame.jpg",
        latest_frame
    )


    # ======================================
    # SEND TO QWEN
    # ======================================

    print("\n🧠 Qwen is thinking...")

    objects_text = ", ".join(detected_objects)

    prompt = f"""
You are AVAQ, a local AI visual assistant.

The fast vision system currently detects these objects:

{objects_text}

The user asks:

{question}

Look carefully at the provided camera image and answer the user's question.

Use the camera image as the main source of truth.

Be concise and natural.

If you are uncertain, say that you are uncertain instead of inventing details.
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


    answer = response["message"]["content"]


    # ======================================
    # ANSWER
    # ======================================

    print("\n========================================")
    print("                 AVAQ")
    print("========================================")
    print(answer)
    print("========================================")


# ==========================================
# CLEANUP
# ==========================================

running = False

cap.release()

cv2.destroyAllWindows()

print("\nAVAQ stopped.")