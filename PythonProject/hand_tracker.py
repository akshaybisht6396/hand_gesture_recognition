import cv2
import mediapipe as mp
# --- ADDED LIBRARIES ---
import os
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- ADDED: DATA COLLECTION SETUP ---
# Create a directory to save the data
DATA_PATH = "./gesture_data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# The gestures you want to collect
gestures = ['fist', 'open_palm', 'thumbs_up','done']
current_gesture_index = 0
print(f"Starting data collection for: {gestures[current_gesture_index]}")

# Open the CSV file to save the data
# We add the 'w' flag to write to the file
csv_file = open(os.path.join(DATA_PATH, 'gesture_data.csv'), 'w', newline='')
csv_writer = csv.writer(csv_file)
# --- END OF ADDED SETUP ---

# Start video capture
cap = cv2.VideoCapture(0)

# Main loop to process video frames
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a natural view and convert color space
    image = cv2.flip(image, 1)

    # --- ADDED: Display which gesture you are collecting ---
    cv2.putText(image, f'Collecting for: {gestures[current_gesture_index]}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # --- converts the image from BGR TO RGB---
    image_for_processing = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_processing.flags.writeable = False  # Performance optimization

    # Process the image to find hand landmarks
    results = hands.process(image_for_processing)

    # image.flags.writeable = True # No longer needed here
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # No need to convert back and forth

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- ADDED: DATA EXTRACTION AND SAVING LOGIC ---
            # 1. Normalize coordinates relative to the wrist
            landmarks = hand_landmarks.landmark
            wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

            relative_landmarks = []
            for landmark in landmarks:
                relative_landmarks.append(landmark.x - wrist_x)
                relative_landmarks.append(landmark.y - wrist_y)
                relative_landmarks.append(landmark.z - wrist_z)

            # 2. Check for key presses to save data or switch gestures
            key = cv2.waitKey(5) & 0xFF

            # Press 's' to save the current frame's landmark data
            if key == ord('s'):
                # Add the label (the gesture name) and the landmarks to a list
                row = [gestures[current_gesture_index]] + relative_landmarks
                csv_writer.writerow(row)
                print(f"Saved data point for {gestures[current_gesture_index]}")

            # Press 'n' to switch to the next gesture
            if key == ord('n'):
                current_gesture_index = (current_gesture_index + 1) % len(gestures)
                print(f"Switched to collecting data for: {gestures[current_gesture_index]}")

            # Press 'q' to quit
            if key == ord('q'):
                cap.release()  # Break out of the loop by releasing the camera
                break

    # This is to handle the 'q' press when no hand is detected
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Display the output
    cv2.imshow('Data Collection', image)

# Release resources
print("Finished data collection.")
csv_file.close()
cap.release()
cv2.destroyAllWindows()