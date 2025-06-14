import cv2
import numpy as np
import time
import pyaudio
import threading
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Parameters
MAX_ABSENT_FRAMES = 50
MAX_LOOKING_AWAY_FRAMES = 50
AUDIO_THRESHOLD = 500
RECORD_SECONDS = 60
FORBIDDEN_CLASSES = ['cell phone', 'book', 'laptop']
GAZE_THRESHOLD = 0.15  # Tweak this based on testing

# State
absent_frames = 0
looking_away_frames = 0
cheating = False
detected_forbidden_objects = set()

# Audio detection
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

def monitor_audio():
    global cheating
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("[INFO] Audio monitoring started...")
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        if np.abs(data).mean() > AUDIO_THRESHOLD:
            print("[ALERT] Suspicious audio detected!")
            cheating = True
    stream.stop_stream()
    stream.close()
    p.terminate()

# Start audio thread
audio_thread = threading.Thread(target=monitor_audio)
audio_thread.start()

# Start video
cap = cv2.VideoCapture(0)
start_time = time.time()

print("[INFO] Video monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = model(frame, verbose=False)[0]
    names = model.names

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = r
        label = names[int(cls_id)]
        if label in FORBIDDEN_CLASSES:
            detected_forbidden_objects.add(label)
            cheating = True
            print(f"[ALERT] Forbidden object detected: {label}")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Filter person detections
    person_boxes = [
        (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        for r in results.boxes.data.tolist()
        if names[int(r[-1])] == 'person'
    ]

    filtered_boxes = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < 2000:  # Lowered area to capture distant people
            continue
        overlaps = any(
            abs(x1 - fx1) < 80 and abs(y1 - fy1) < 80
            for fx1, fy1, fx2, fy2 in filtered_boxes
        )
        if not overlaps:
            filtered_boxes.append(box)

    person_count = len(filtered_boxes)

    if person_count > 1:
        print(f"[ALERT] Multiple people detected: {person_count}")
        cheating = True

    # Person presence
    person_detected = person_count > 0
    if not person_detected:
        absent_frames += 1
        print(f"[WARNING] No person detected. Count: {absent_frames}")
    else:
        absent_frames = 0

    # Gaze tracking using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(rgb_frame)

    if results_mp.multi_face_landmarks:
        for face_landmarks in results_mp.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]  # Left iris
            right_eye = face_landmarks.landmark[263]  # Right iris
            nose_tip = face_landmarks.landmark[1]

            # Horizontal difference between eyes and nose
            dx = abs((left_eye.x + right_eye.x) / 2 - nose_tip.x)

            if dx > GAZE_THRESHOLD:
                looking_away_frames += 1
                print(f"[WARNING] Looking away detected. Count: {looking_away_frames}")
            else:
                looking_away_frames = 0
            break  # Track only first face
    else:
        looking_away_frames += 1
        print(f"[WARNING] No face for gaze detection. Count: {looking_away_frames}")

    # Threshold triggers
    if absent_frames > MAX_ABSENT_FRAMES or looking_away_frames > MAX_LOOKING_AWAY_FRAMES:
        print("[ALERT] Cheating behavior detected!")
        cheating = True
        break

    # Display
    cv2.imshow("Proctoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - start_time > RECORD_SECONDS:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
audio_thread.join()

# Final output
print("\n=== Exam Session Summary ===")
if detected_forbidden_objects:
    print("Detected forbidden objects:", ", ".join(detected_forbidden_objects))
print("Result:", "Cheated" if cheating else "Not Cheated")
