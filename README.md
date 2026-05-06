#  AI Proctoring System

> An intelligent remote exam monitoring system that detects and flags suspicious behaviour in real-time using Computer Vision, Deep Learning, and Audio Analysis.

---

##  Problem Statement

With the rise of online education and remote examinations, ensuring academic integrity has become a major challenge. Traditional human proctoring is expensive, inconsistent, and unscalable. This system automates the entire proctoring process using AI — making online exams as secure as in-person ones.

---

##  Key Features

**Object Detection (YOLOv8)** - Detects forbidden items — phones, books, extra monitors in real time 
**Multiple Face Detection** - Flags presence of more than one person in the exam frame 
**Face Recognition** - Continuously validates the candidate's identity throughout the session 
**Gaze Tracking** - Monitors eye and head direction — flags frequent look-away behaviour 
**Audio Monitoring** - Detects multiple voices, background conversations, and suspicious speech patterns 
**Streamlit Dashboard** - Interactive UI to upload exam videos and generate detailed cheating analysis reports 

---

## Tech Stack

Language - Python 3.x 
Object Detection - YOLOv8 (Ultralytics)
Face Recognition - face_recognition
Gaze Tracking - MediaPipe / OpenCV 
Audio Analysis - SpeechRecognition/ pyaudio / librosa 
Web Interface - Streamlit
Video Processing - OpenCV

---

##  System Architecture

```
Input Video / Live Webcam Feed
         │
         ▼
┌─────────────────────────────────────┐
│         Frame Extraction            │
└──────────────┬──────────────────────┘
               │
       ┌───────┼──────────┐
       ▼       ▼          ▼
  YOLOv8   Face Rec   Gaze Track
  (Objects) (Identity) (Eye/Head)
       │       │          │
       └───────┼──────────┘
               ▼
     Audio Monitoring Layer
               │
               ▼
     Suspicious Event Logger
               │
               ▼
   Streamlit Report Dashboard
```

---

##  Project Structure

```
AI-Proctoring-System/
├── app.py                  # Streamlit main app
├── detection/
│   ├── object_detect.py    # YOLOv8 forbidden item detection
│   ├── face_detect.py      # Multiple face detection
│   ├── face_recognize.py   # Identity verification
│   └── gaze_track.py       # Eye and head tracking
├── audio/
│   └── audio_monitor.py    # Voice activity & speech analysis
├── utils/
│   └── report_generator.py # PDF/visual report generation
├── models/                 # YOLOv8 weights
├── sample_videos/          # Test input videos
├── requirements.txt
└── README.md
```

---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Ashwinbalachandar14/AI-Proctoring-System.git
cd AI-Proctoring-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```


## Real World Use Cases

- **Universities & Colleges** — Secure online semester exams
- **Recruitment Platforms** — Proctored coding assessments
- **Certification Bodies** — Online certification exams
- **EdTech Platforms** — Verified skill assessments

---

##  Future Improvements

- [ ] Real-time live stream proctoring (not just pre-recorded video)
- [ ] Automated PDF report generation with timestamps
- [ ] Integration with LMS platforms (Moodle, Canvas)
- [ ] Cloud deployment (AWS / GCP)
- [ ] Mobile device detection improvement

