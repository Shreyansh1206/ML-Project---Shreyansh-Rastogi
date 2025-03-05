import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

data_file = "Gesture_Data.csv"

if not os.path.exists(data_file):
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gesture"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

gesture_name = input("Enter Gesture Name: ")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                     min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])

                with open(data_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gesture_name] + data)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collecting Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
