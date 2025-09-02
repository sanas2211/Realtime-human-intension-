import numpy as np
import mediapipe as mp
def calculate_emotion_intensity(emotion_scores, pose_landmarks=None):
    
    intensity_scores = {}
    if not emotion_scores:
        return intensity_scores

    # Directly map facial emotions to intensity (0-100)
    for emo, score in emotion_scores.items():
        intensity_scores[emo] = min(max(score, 0), 100)

    # -------------------- Posture Factor --------------------
    posture_factor = 0
    if pose_landmarks:
        # Example: calculate slouching based on shoulder landmarks
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        shoulder_avg_y = (left_shoulder.y + right_shoulder.y)/2
        hip_avg_y = (left_hip.y + right_hip.y)/2

        torso_slope = hip_avg_y - shoulder_avg_y
        # Slouching = smaller slope → higher posture factor
        posture_factor = np.clip((0.3 - torso_slope)*300, 0, 100)  # scaled 0-100

    # -------------------- Stress Score --------------------
    angry = intensity_scores.get('angry',0)
    fear = intensity_scores.get('fear',0)
    sad = intensity_scores.get('sad',0)

    stress_score = 0.4*angry + 0.3*fear + 0.2*sad + 0.1*posture_factor
    stress_score = min(max(stress_score,0),100)

    intensity_scores['stress'] = stress_score

    return intensity_scores

