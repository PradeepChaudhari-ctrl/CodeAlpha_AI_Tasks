# 🤖 CodeAlpha AI Internship — All Tasks

> **Intern:** CodeAlpha AI Intern  
> **Program:** Artificial Intelligence Internship — CodeAlpha  
> **Website:** [www.codealpha.tech](https://www.codealpha.tech)

---

## 📋 Tasks Completed

| # | Task | Tech Stack | Status |
|---|------|-----------|--------|
| 1 | Language Translation Tool | Python · Tkinter · deep-translator | ✅ Done |
| 2 | Chatbot for FAQs | Python · Tkinter · TF-IDF (stdlib only) | ✅ Done |
| 3 | Music Generation with AI | TensorFlow · Keras LSTM · music21 | ✅ Done |
| 4 | Object Detection & Tracking | YOLOv8 · OpenCV · Custom SORT tracker | ✅ Done |

---

## 📂 Project Structure

```
CodeAlpha_AI_Tasks/
├── Task1_LanguageTranslation/
│   ├── app.py               ← Main application
│   └── requirements.txt
├── Task2_ChatbotFAQ/
│   ├── chatbot.py           ← Main application
│   └── requirements.txt
├── Task3_MusicGeneration/
│   ├── music_generator.py   ← Main application
│   └── requirements.txt
└── Task4_ObjectDetection/
    ├── object_detection.py  ← Main application
    └── requirements.txt
```

---

## 🚀 Task 1 — Language Translation Tool

### Features
- 🌍 Supports **50+ languages** including Hindi, French, German, Japanese, Arabic, etc.
- ⇄ **Swap languages** button for quick reversal
- 📋 **Copy to clipboard** for easy use
- 🎨 Beautiful dark UI (Catppuccin theme)
- Character counter (5000 limit)
- Threaded translation — UI never freezes

### How to Run
```bash
cd Task1_LanguageTranslation
pip install -r requirements.txt
python app.py
```

### Screenshot Preview
```
┌─────────────────────────────────────────────────────┐
│  🌍  Language Translation Tool — CodeAlpha          │
│  ─────────────────────────────────────────────────  │
│  [Source: Auto Detect ▼]  ⇄  [Target: French ▼]    │
│  ┌──────────────────┐   ┌──────────────────────┐   │
│  │ Hello, how are   │   │ Bonjour, comment     │   │
│  │ you today?       │   │ allez-vous           │   │
│  │                  │   │ aujourd'hui?         │   │
│  └──────────────────┘   └──────────────────────┘   │
│     [🌐 Translate]  [📋 Copy]  [🗑️ Clear]          │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Task 2 — Chatbot for FAQs

### Features
- 🧠 **TF-IDF + Cosine Similarity** engine (zero external ML deps)
- 24 built-in FAQ pairs covering CodeAlpha, internship, and AI topics
- Smooth **Tkinter chat UI** with colour-coded messages
- Confidence threshold — gracefully handles unknown questions
- Temperature-insensitive: works with casual phrasing

### How to Run
```bash
cd Task2_ChatbotFAQ
python chatbot.py   # No pip install needed!
```

### Sample Conversation
```
🧑 You: What perks does the internship offer?
🤖 Bot: Internship perks include an offer letter, completion certificate (QR verified),
        unique ID certificate, letter of recommendation (based on performance),
        job placement support, and resume building assistance.

🧑 You: How do I submit tasks?
🤖 Bot: Submit your completed tasks via the submission form shared in your WhatsApp
        group. Upload source code to GitHub in a repository named CodeAlpha_ProjectName
        and post a video on LinkedIn.
```

---

## 🚀 Task 3 — Music Generation with AI

### Features
- 🎵 **LSTM neural network** (Embedding → LSTM → LSTM → Softmax)
- Trains on built-in classical/jazz note sequences
- **Temperature-controlled** generation (0.8 = creative, 0.3 = conservative)
- Exports generated sequences as **MIDI files** via music21
- **Demo mode** — generates music even without TensorFlow installed
- Early stopping + model checkpointing

### How to Run
```bash
cd Task3_MusicGeneration
pip install -r requirements.txt
python music_generator.py
```
Output: `generated_music/generated_music.mid`  
Open with **VLC, GarageBand, Windows Media Player**, or any DAW.

### Model Architecture
```
Input (seq_len=20)
    ↓
Embedding (vocab_size → 64)
    ↓
LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM (128 units)
    ↓
Dropout (0.3)
    ↓
Dense (vocab_size, softmax)
```

---

## 🚀 Task 4 — Object Detection & Tracking

### Features
- 🎯 **YOLOv8** (ultralytics) — state-of-the-art detection
- 🔢 **Custom IoU-based tracker** — assigns consistent IDs across frames
- Detects **80 COCO classes** (person, car, phone, bottle, etc.)
- Real-time **FPS counter** and object count overlay
- Supports **webcam** or **video file** input
- Colour-coded bounding boxes per track ID

### How to Run
```bash
cd Task4_ObjectDetection
pip install -r requirements.txt

# Webcam (default)
python object_detection.py

# Video file
python object_detection.py --source path/to/video.mp4

# Adjust confidence & model size
python object_detection.py --conf 0.5 --model yolov8s.pt
```

### CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | `0` for webcam, or path to video file |
| `--conf` | `0.4` | Detection confidence threshold |
| `--model` | `yolov8n.pt` | Model: n/s/m/l (nano=fastest, large=best) |

---

## 🛠️ Global Setup

```bash
# Python 3.9+ required
python --version

# Install all at once (optional)
pip install deep-translator tensorflow music21 numpy ultralytics opencv-python
```

---

## 📤 Submission Details

- ✅ Source code uploaded to GitHub: `CodeAlpha_AI_Tasks`
- ✅ Video explanation posted on LinkedIn (tagging @CodeAlpha)
- ✅ Submitted via WhatsApp group submission form

---

## 📞 Contact

| | |
|--|--|
| 🌐 Website | [www.codealpha.tech](https://www.codealpha.tech) |
| 📧 Email | services@codealpha.tech |
| 💬 WhatsApp | +91 9336576683 |
