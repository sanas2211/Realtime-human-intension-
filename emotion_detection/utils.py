import cv2
import numpy as np

# -------------------- Night Vision --------------------
def apply_night_vision(frame):
    """
    Converts frame to night vision style (green tint + CLAHE contrast)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    night_vision_frame = cv2.merge([np.zeros_like(enhanced), enhanced, np.zeros_like(enhanced)])
    return night_vision_frame

# -------------------- Draw Dominant Emotion --------------------
def draw_emotions(frame, dominant_emotion, intensity_scores=None):
    """
    Draws dominant emotion text and optionally intensity scores on frame.
    """
    if dominant_emotion:
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if intensity_scores:
        y0 = 70
        for emo, score in intensity_scores.items():
            cv2.putText(frame, f"{emo}: {round(score,2)}%", (30, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 20

    return frame

# -------------------- Draw Intensity Bars --------------------
def draw_intensity_bars(frame, intensity_scores):
    """
    Draw horizontal bars for each intensity score.
    """
    x_start, y_start = 30, 300
    bar_width = 200
    bar_height = 20
    gap = 10

    for emo, score in intensity_scores.items():
        filled_width = int((score/100) * bar_width)
        # Background bar
        cv2.rectangle(frame, (x_start, y_start), (x_start+bar_width, y_start+bar_height), (50,50,50), -1)
        # Filled bar
        cv2.rectangle(frame, (x_start, y_start), (x_start+filled_width, y_start+bar_height), (0,255,0), -1)
        # Text
        cv2.putText(frame, f"{emo} {round(score,2)}%", (x_start + bar_width + 10, y_start + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y_start += bar_height + gap

    return frame
