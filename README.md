Project Overview

This project detects human emotions and stress levels in real-time using a webcam. It combines facial emotion recognition and pose/posture analysis to provide an intensity score for each emotion and an overall stress level.

Key features:

Real-time facial emotion detection using DeepFace.

Pose estimation using MediaPipe to analyze posture and slouching.

Calculation of stress score based on emotions and posture.

Optional night vision mode for low-light environments.

Interactive dashboard (Dash + Plotly) for visualization of emotion intensity over time.

Modular structure for easy extension and integration.

Folder Structure
Emotion-Intensity-Detection/
│── data/                  # Optional dataset or CSV logs
│── models/                # Pre-trained or saved models
│── emotion_detection/
│   │── emotion_model.py   # Real-time emotion detection code
│   │── utils.py           # Helper functions
│── intensity_detection/
│   │── intensity_logic.py # Emotion + posture intensity calculation
│── dashboard.py           # Dash-based visualization dashboard
│── main.py                # Entry point (runs webcam + emotion + intensity)
│── requirements.txt       # All required Python packages
│── README.md              # Project explanation

Installation

Clone the repository:


git clone <your-repo-link>
cd Emotion-Intensity-Detection


Install dependencies:


pip install -r requirements.txt


Run the project:

python main.py


Press n to toggle night vision mode, q to quit.

Usage

The system will capture your webcam feed.

Detects dominant emotion and displays percentage intensity for each emotion.

Calculates posture-based intensity and stress score.

Optional dashboard visualizes emotion and stress trends over time.

Technologies Used

Python, OpenCV, MediaPipe, DeepFace

NumPy, Pandas, Matplotlib, Plotly

Flask, Dash for dashboard visualization


Future Enhancements

Multi-person emotion and posture detection.

Integration with AI-based behavior prediction.

Export CSV or PDF reports of emotion and stress trends.