# SignBridgeAI

SignBridgeAI recognizes hand signs (starting with the alphabet) through a webcam and converts them into text. It uses MediaPipe for hand tracking, OpenCV for the camera, and a machine learning classifier trained on hand landmarks.

## Project Files

| File | Purpose |
|------|---------|
| `camera_test.py` | Checks that the webcam works |
| `hand_detection.py` | Detects the hand and draws landmarks |
| `collect_data.py` | Records hand landmark data for each sign |
| `data.csv` | The collected training data |
| `train.py` | Trains the classifier and saves `model.pkl` |
| `model.pkl` | The trained model |
| `inference_classifier.py` | Live sign recognition from the webcam |
| `sign_to_text.py` | Converts recognized signs into text |
| `requirements.txt` | Python libraries needed |

## Setup (Step by Step)

1. Install Python 3.10 or newer.
2. Download or clone this repository:
   ```
   git clone https://github.com/anwitadatta-svg/SignBridgeAI.git
   cd SignBridgeAI
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate it (Windows):
   ```
   venv\Scripts\activate
   ```
5. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## How to Use (Step by Step)

1. **Test your camera**
   ```
   python camera_test.py
   ```
2. **Check hand detection**
   ```
   python hand_detection.py
   ```
3. **Collect data** for each sign and follow the on-screen instructions:
   ```
   python collect_data.py
   ```
4. **Train the model**
   ```
   python train.py
   ```
5. **Run live recognition**
   ```
   python inference_classifier.py
   ```
6. **Convert signs to text**
   ```
   python sign_to_text.py
   ```

Press `q` in the camera window to quit.

## Tips

- Use good lighting and keep your hand clearly visible.
- Collect many samples per sign for better accuracy.
- Re-run `train.py` every time you add new data.

## Tech Stack

Python, OpenCV, MediaPipe, scikit-learn

## Author

Anwita Datta