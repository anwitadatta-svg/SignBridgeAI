# SignBridgeAI 🤟

An AI-powered computer vision system designed to bridge the communication gap for the hearing impaired by translating real-time hand gestures into text and speech. Developed for the **Neural Storm 2026 Hackathon**.

## 🚀 Features
- **Real-Time Hand Detection:** High-accuracy gesture tracking via camera feed.
- **Instant Translation:** Seamless processing of sign movements.
- **Isolated Workspace:** Uses a dedicated Python virtual environment for reliable performance.

## 🛠️ Setup & Installation

1. **Activate the Virtual Environment:**
   ```bash
   .\venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 How to Run

To run the primary hand detection model, execute:
```bash
python hand_detection.py
```

To verify your web camera configuration:
```bash
python camera_test.py
```