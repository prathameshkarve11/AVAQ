import cv2
from ultralytics import YOLO

print("Loading YOLO26...")

model = YOLO("yolo26n.pt")

print("Starting camera...")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("AVAQ - Vision", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()