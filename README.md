Description
A Python-based application that translates human hand gestures into system commands or digital input. By utilizing MediaPipe’s hand landmark model, the system identifies 21 3D hand joints to recognize complex patterns with high precision and low latency.

Key Features
Landmark Detection: Real-time tracking of 21 hand points using MediaPipe.

Gesture Classification: Recognition of custom gestures (e.g., Thumb Up, Peace, OK).

System Control: Map gestures to keyboard shortcuts, mouse movement, or volume control.

Low Latency: Optimized for standard webcams without the need for a dedicated GPU.

🛠️ Tech Stack
Language: Python

Libraries: OpenCV, MediaPipe, NumPy, PyAutoGUI (for system control).

🚀 Getting Started
Install Requirements: pip install opencv-python mediapipe pyautogui

Run Application: python gesture_control.py
