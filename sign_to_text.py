import threading
import time
import warnings

import cv2
import joblib
import mediapipe as mp
import pyttsx3

warnings.filterwarnings("ignore")

HOLD_SECONDS = 1.0
MIN_CONFIDENCE = 0.60

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def normalize(landmarks):
    wrist = landmarks[0]
    coords = [(p.x - wrist.x, p.y - wrist.y, p.z - wrist.z) for p in landmarks]
    scale = max(max(abs(v) for v in c) for c in coords) or 1.0
    return [v / scale for c in coords for v in c]


def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)

    if text.strip():
        threading.Thread(target=run, daemon=True).start()


cap = cv2.VideoCapture(0)
sentence = ""
current = None
hold_start = 0.0
locked = False

print("Hold a sign for 1 second to add the letter.")
print("SPACE = space | BACKSPACE = delete | ENTER = speak | C = clear | ESC = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    progress = 0.0
    shown = "No hand"
    letter = None

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        features = normalize(lm.landmark)
        pred = model.predict([features])[0]
        conf = model.predict_proba([features]).max()
        shown = f"{pred} ({conf:.0%})"
        if conf >= MIN_CONFIDENCE:
            letter = pred

    if letter is None:
        current = None
        locked = False
    elif letter != current:
        current = letter
        hold_start = time.time()
        locked = False
    elif not locked:
        progress = min((time.time() - hold_start) / HOLD_SECONDS, 1.0)
        if progress >= 1.0:
            sentence += letter
            locked = True

    h, w = frame.shape[:2]
    cv2.putText(frame, shown, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0), 3)
    cv2.rectangle(frame, (10, 60), (210, 78), (255, 255, 255), 2)
    cv2.rectangle(frame, (10, 60), (10 + int(200 * progress), 78),
                  (0, 200, 0), -1)
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, sentence[-40:] + "_", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("SignBridge AI - Sign to Text", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        sentence += " "
    elif key == 8:
        sentence = sentence[:-1]
    elif key == 13:
        speak(sentence)
    elif key in (ord("c"), ord("C")):
        sentence = ""

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Final text:", sentence)