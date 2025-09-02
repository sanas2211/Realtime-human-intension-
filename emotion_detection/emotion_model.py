import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

# Night vision toggle
night_vision = False

def apply_night_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert to green-tinted image
    night_vision_frame = cv2.merge([np.zeros_like(enhanced), enhanced, np.zeros_like(enhanced)])
    return night_vision_frame

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    # Flip and convert
    frame = cv2.flip(frame, 1)

    # Apply night vision if toggled
    if night_vision:
        frame = apply_night_vision(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose estimation
    result = pose.process(rgb)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Emotion recognition
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_percentages = analysis[0]['emotion']

        # Display emotion
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show all percentages
        y0 = 70
        for emotion, score in emotion_percentages.items():
            cv2.putText(frame, f"{emotion}: {round(score, 2)}%", (30, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 20

    except Exception as e:
        cv2.putText(frame, "Emotion: Not detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time Intention Detection", frame)

    # Handle key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        night_vision = not night_vision
        print("Night Vision:", "ON" if night_vision else "OFF")

cap.release()
cv2.destroyAllWindows()
