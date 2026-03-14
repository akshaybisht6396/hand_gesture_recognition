import cv2
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# --- 1. Load and Prepare Data ---

# Load the dataset (Pandas)
DATA_PATH = "./gesture_data"
df = pd.read_csv(os.path.join(DATA_PATH, 'gesture_data.csv'), header=None)

# Separate features (X) and labels (y)(Pandas)
X = df.iloc[:, 1:]  # All columns except the first one (landmarks)
y = df.iloc[:, 0]  # The first column (gesture name)

# Split data into training and testing sets(scikit learn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train the Machine Learning Model ---

# Initialize the K-Nearest Neighbors (KNN) classifier (scikit learn)
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# --- 3. Evaluate the Model (Optional but good practice) ---

# Make predictions on the test data(scikit learn)
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 4. Real-Time Gesture Recognition ---

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands  #  Initializes the MediaPipe Hands module again for the real-time recognition part.
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #Initializing Cv2

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:#detected any hands in the current video frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize landmarks for prediction
            landmarks = hand_landmarks.landmark
            wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

            relative_landmarks = []#This line extracts the list of 21 landmark objects for the detected hand
            for landmark in landmarks:
                relative_landmarks.append(landmark.x - wrist_x)
                relative_landmarks.append(landmark.y - wrist_y)
                relative_landmarks.append(landmark.z - wrist_z)

            # Make a prediction (scikit learn)
            prediction = model.predict([np.array(relative_landmarks)])
            gesture_name = prediction[0]

            # Display the prediction on the screen
            cv2.putText(image, gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)#final processed image

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()