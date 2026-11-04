import warnings

import cv2
import joblib
import mediapipe as mp

warnings.filterwarnings("ignore")

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


cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "No hand"
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        features = normalize(lm.landmark)
        letter = model.predict([features])[0]
        conf = model.predict_proba([features]).max()
        text = f"{letter} ({conf:.0%})"

    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 0), 3)
    cv2.imshow("SignBridge AI - Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()