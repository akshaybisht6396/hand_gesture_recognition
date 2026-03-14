import cv2
import mediapipe as mp

# Initialize MediaPipe and Webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Landmark indices for the tips of the fingers
finger_tips = [8, 12, 16, 20]
# Landmark indices for the second joint (PIP) of the fingers
pip_joints = [6, 10, 14, 18]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get all the landmark coordinates
            landmarks = hand_landmarks.landmark

            # --- Thumb Logic ---
            # Check if the thumb tip is to the left (for right hand) of the joint below it
            # This logic assumes a right hand in a flipped image
            if landmarks[4].x < landmarks[3].x:
                finger_count += 1

            # --- Four Fingers Logic ---
            for i in range(4):  # Loop through index, middle, ring, pinky
                # Check if the finger tip is above the PIP joint
                if landmarks[finger_tips[i]].y < landmarks[pip_joints[i]].y:
                    finger_count += 1

    # Display the finger count on the screen
    cv2.putText(image, str(finger_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow('Finger Counter', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()