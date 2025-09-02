import cv2
import mediapipe as mp
import time
import os
import csv
from datetime import datetime
import numpy as np
from collections import deque

from emotion_detection.emotion_model import detect_emotions
from emotion_detection.utils import draw_emotions, draw_intensity_bars, apply_night_vision
from intensity_detection.intensity_logic import calculate_emotion_intensity

# Optional: sound alert (Windows)
import winsound

# -------------------- Settings --------------------
night_vision = False
log_data = True
max_fps = 30  # for efficiency score

# Emotion thresholds for alert
ALERT_THRESHOLDS = {
    'angry': 70,
    'fear': 60,
    'sad': 80,
    'stress': 75  # if calculated from multiple emotions
}

# -------------------- Webcam Setup --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# -------------------- Pose Setup --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# -------------------- Logging Setup --------------------
if log_data:
    if not os.path.exists("data"):
        os.makedirs("data")
    log_file = "data/emotion_log.csv"
    if not os.path.isfile(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Dominant Emotion", "Emotion Scores",
                "Intensity Scores", "FPS", "Efficiency", "Alerts"
            ])

# -------------------- Efficiency Function --------------------
def efficiency_score(fps, max_fps=max_fps):
    score = min((fps / max_fps) * 100, 100)
    return round(score, 2)

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    frame = cv2.flip(frame, 1)
    start_time = time.time()

    # -------------------- Night Vision --------------------
    if night_vision:
        frame = apply_night_vision(frame)

    # -------------------- Pose Estimation --------------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    pose_landmarks = result.pose_landmarks
    if pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # -------------------- Emotion Detection --------------------
    dominant_emotion, emotion_scores = detect_emotions(frame)

    # -------------------- Advanced Intensity --------------------
    intensity_scores = calculate_emotion_intensity(emotion_scores, pose_landmarks)

    # -------------------- Drawing --------------------
    frame = draw_emotions(frame, dominant_emotion, intensity_scores)
    if intensity_scores:
        frame = draw_intensity_bars(frame, intensity_scores)

    # -------------------- Alerts --------------------
    alerts = []
    if intensity_scores:
        for emo, score in intensity_scores.items():
            if emo in ALERT_THRESHOLDS and score >= ALERT_THRESHOLDS[emo]:
                alerts.append(f"{emo.upper()} ALERT!")
        # Display alerts
        y0 = 100
        for alert in alerts:
            cv2.putText(frame, alert, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            y0 += 30
            # Optional beep
            winsound.Beep(1000, 150)

    # -------------------- Efficiency --------------------
    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time if processing_time > 0 else 0
    eff_score = efficiency_score(fps)

    cv2.putText(frame, f"FPS: {int(fps)}", (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Efficiency: {eff_score}%", (500, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------------------- Logging to CSV --------------------
    if log_data and dominant_emotion and intensity_scores:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                dominant_emotion,
                emotion_scores,
                intensity_scores,
                round(fps,2),
                eff_score,
                ", ".join(alerts)
            ])

    # -------------------- Display --------------------
    cv2.imshow("Emotion + Intensity Detection", frame)

    # -------------------- Key Handling --------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('Q'):
        night_vision = not night_vision
        print("Night Vision:", "ON" if night_vision else "OFF")

cap.release()
cv2.destroyAllWindows()
