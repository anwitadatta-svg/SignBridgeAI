# SignBridgeAI 🤟

Real-time sign language alphabet recognition from a webcam. It tracks your hand, predicts the letter, builds sentences and speaks them aloud.

## Features
- **Real-time hand tracking** with MediaPipe
- **Custom-trained letter classifier** (scikit-learn Random Forest)
- **Sentence builder**: hold a sign for 1 second to add the letter
- **Text-to-speech** output with pyttsx3

## Setup
Requires Python 3.12.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Use
1. **Record samples** (press a letter key while showing the sign, ESC to quit):
   ```
   python collect_data.py
   ```
2. **Train the model:**
   ```
   python train.py
   ```
3. **Sign to text with speech:**
   ```
   python sign_to_text.py
   ```
   Controls: `SPACE` space, `BACKSPACE` delete, `ENTER` speak, `C` clear, `ESC` quit.

Other scripts: `hand_detection.py` (hand tracking demo), `camera_test.py` (webcam check), `inference_classifier.py` (live letter prediction).

## Tech Stack
Python, OpenCV, MediaPipe, scikit-learn, pyttsx3

## Limitations
J and Z need motion and are not recognised as moving signs.