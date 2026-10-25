import cv2
import ollama

print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera ready.")
print("Press SPACE to take a picture.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera.")
        break

    cv2.imshow("AVAQ - Ask Qwen", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:
        cv2.imwrite("avaq_frame.jpg", frame)
        print("Picture captured!")
        break

cap.release()
cv2.destroyAllWindows()

print("Asking Qwen...")

response = ollama.chat(
    model="qwen3-vl:2b",
    messages=[
        {
            "role": "user",
            "content": "What do you see in this image? Describe the important objects and people briefly.",
            "images": ["avaq_frame.jpg"]
        }
    ]
)

print("\n==============================")
print("AVAQ:")
print(response["message"]["content"])
print("==============================")