import csv
import os
import warnings

import cv2
import mediapipe as mp

warnings.filterwarnings("ignore")

DATA_FILE = "data.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        header = ["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
        csv.writer(f).writerow(header)


def normalize(landmarks):
    wrist = landmarks[0]
    coords = [(p.x - wrist.x, p.y - wrist.y, p.z - wrist.z) for p in landmarks]
    scale = max(max(abs(v) for v in c) for c in coords) or 1.0
    return [v / scale for c in coords for v in c]


def count_samples():
    counts = {}
    with open(DATA_FILE) as f:
        next(f)
        for line in f:
            label = line.split(",", 1)[0]
            counts[label] = counts.get(label, 0) + 1
    return counts


cap = cv2.VideoCapture(0)
counts = count_samples()
last_saved = ""

print("Show a sign, then press its letter key (A-Z) to save a sample.")
print("Hold the key to save many samples. Press ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_found = bool(result.multi_hand_landmarks)
    if hand_found:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    if 97 <= key <= 122 and hand_found:
        label = chr(key).upper()
        row = [label] + normalize(lm.landmark)
        with open(DATA_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
        counts[label] = counts.get(label, 0) + 1
        last_saved = label

    status = "Hand found" if hand_found else "No hand"
    color = (0, 200, 0) if hand_found else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if last_saved:
        text = f"Saved {last_saved}: {counts[last_saved]} samples"
        cv2.putText(frame, text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)

    cv2.imshow("SignBridge AI - Collect Data", frame)

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Samples per letter:", dict(sorted(counts.items())))