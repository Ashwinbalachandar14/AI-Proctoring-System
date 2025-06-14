# Updated AI Proctoring System with Reduced False Positives

import os
import cv2
import mediapipe as mp
import face_recognition
from ultralytics import YOLO
import streamlit as st

from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import shutil

# List of cheating-related object labels from YOLO
CHEATING_OBJECTS = [
    "cell phone", "book", "monitor", "keyboard","paper"
    "mouse", "tv", "tablet", "headphones", "earphone", "person"
]


# FRAME EXTRACTION

def extract_frames(video_path, output_folder="frames", frame_rate=3):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
    else:
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / frame_rate))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = f"{output_folder}/frame_{count}.jpg"
            cv2.imwrite(filename, frame)
        count += 1
    cap.release()

# -----------------------------
# AUDIO EXTRACTION AND ANALYSIS
# -----------------------------
def extract_audio_from_video(video_path):
    try:
        audio = AudioSegment.from_file(video_path)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        print("Audio extraction failed:", e)
        return None

def analyze_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        word_count = len(text.split())
        return {
            "speech_detected": True,
            "multiple_voice_hint": word_count > 50,  # Adjusted heuristic
            "transcript": text
        }
    except sr.UnknownValueError:
        return {
            "speech_detected": False,
            "multiple_voice_hint": False,
            "transcript": ""
        }

# -----------------------------
# GAZE DETECTION (BASIC)
# -----------------------------
def analyze_face_gaze(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3)
    image = cv2.imread(image_path)
    if image is None:
        return "No face", 0, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    face_mesh.close()

    if not results.multi_face_landmarks:
        return "No face", 0, None

    landmarks_list = results.multi_face_landmarks
    primary_landmarks = landmarks_list[0].landmark

    if len(primary_landmarks) < 477:
        return "No face", len(landmarks_list), None  # Not enough landmarks for iris tracking

    def avg_point(indices):
        return sum([primary_landmarks[i].x for i in indices]) / len(indices)

    left_iris_x = avg_point([468, 469, 470, 471])
    right_iris_x = avg_point([473, 474, 475, 476])

    gaze_status = "Looking at screen" if 0.4 < left_iris_x < 0.6 and 0.4 < right_iris_x < 0.6 else "Looking away"

    return gaze_status, len(landmarks_list), None


def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None


@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

def detect_objects(image_path, yolo_model, confidence_threshold=0.45):
    results = yolo_model(image_path)
    boxes = results[0].boxes
    detected = []
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = results[0].names[cls_id]
        print(f"Detected: {label} with confidence {conf:.2f}")  # <--- Here
        if conf >= confidence_threshold:
            detected.append(label)
    return detected


def run_cheating_analysis(video_path):
    frame_folder = "frames"
    extract_frames(video_path, frame_folder, frame_rate=1)
    yolo_model = load_yolo_model()

    gaze_flags = []
    object_flags = []
    multi_face_flags = []
    face_mismatch_flags = []

    reference_encoding = None
    absence_counter = 0
    max_absence_threshold = 5
    prolonged_absence_flags = []

    for img_file in sorted(os.listdir(frame_folder)):
        img_path = os.path.join(frame_folder, img_file)

        gaze, face_count, _ = analyze_face_gaze(img_path)
        if gaze != "Looking at screen":
            gaze_flags.append(img_file)

        if gaze == "No face":
            absence_counter += 1
        else:
            if absence_counter >= max_absence_threshold:
                prolonged_absence_flags.append(img_file)
            absence_counter = 0

        if face_count > 1:
            multi_face_flags.append(img_file)

        # Face mismatch check
        current_encoding = get_face_encoding(img_path)
        if current_encoding is not None:
            if reference_encoding is None:
                reference_encoding = current_encoding
            elif not face_recognition.compare_faces([reference_encoding], current_encoding, tolerance=0.6)[0]:
                face_mismatch_flags.append(img_file)

        # Object detection
        objects = detect_objects(img_path, yolo_model)
        flagged = False
        for obj in objects:
            if obj in CHEATING_OBJECTS:
                if obj == "person" and objects.count("person") <= 1:
                    continue
                flagged = True
                break
        if flagged:
            object_flags.append((img_file, objects))

    # Audio
    audio_path = extract_audio_from_video(video_path)
    audio_report = {
        "speech_detected": False,
        "multiple_voice_hint": False,
        "transcript": ""
    }
    if audio_path:
        audio_report = analyze_audio(audio_path)
        shutil.rmtree(os.path.dirname(audio_path))

    return {
        "suspicious_gaze_frames": len(gaze_flags),
        "multi_face_frames": len(multi_face_flags),
        "face_mismatch_frames": len(face_mismatch_flags),
        "object_detected_frames": len(object_flags),
        "details_objects": object_flags,
        "prolonged_absence_events": len(prolonged_absence_flags),
        "audio_speech_detected": audio_report["speech_detected"],
        "audio_multiple_voice_hint": audio_report["multiple_voice_hint"],
        "audio_transcript_snippet": audio_report["transcript"][:100],
        "cheating_likely": (
            len(gaze_flags) > 10
            or bool(object_flags)
            or bool(multi_face_flags)
            or bool(face_mismatch_flags)
        ),
    }

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("AI Proctoring")

uploaded_file = st.file_uploader("Upload exam video", type=["mp4", "avi", "mov"])
if uploaded_file:
    video_ext = uploaded_file.name.split('.')[-1]
    video_path = f"temp_video.{video_ext}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    if st.button("Run Cheating Analysis"):
        with st.spinner("Analyzing video... This may take some time."):
            report = run_cheating_analysis(video_path)

        st.subheader("Cheating Report")
        st.write(f"Suspicious gaze frames: {report['suspicious_gaze_frames']}")
        st.write(f"Multiple faces detected : {report['multi_face_frames']}")
        st.write(f"Face mismatch frames: {report['face_mismatch_frames']}")
        st.write(f"Frames with suspicious objects: {report['object_detected_frames']}")
        st.write(f"Prolonged absence events: {report['prolonged_absence_events']}")
        st.write(f"Voice detected in audio: {report['audio_speech_detected']}")
        st.write(f"Multiple voices likely: {report['audio_multiple_voice_hint']}")
        if report['audio_transcript_snippet']:
            st.write("Audio transcript snippet:")
            st.info(report['audio_transcript_snippet'])

        if report["object_detected_frames"]:
            st.write("Details of detected objects per frame:")
            for frame, objs in report["details_objects"]:
                st.write(f" - {frame}: {objs}")

        if report["cheating_likely"]:
            st.error("\U0001F6A8 Cheating likely detected.")
        else:
            st.success("\u2705 No major cheating behavior detected.")
