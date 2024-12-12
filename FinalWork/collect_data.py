import os
import cv2
import mediapipe as mp

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

DATA_DIR = './test'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 100

# Use the correct camera index (try different values like 0, 1, 2, etc.)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened. Check the camera index.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read a frame from the camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect dataset_size images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read a frame from the camera.")
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Draw hand landmarks for visualization
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save the image
            cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
            counter += 1
            print(f"Image {counter} saved for class {j}")
        else:
            print("No hand detected, skipping frame.")

        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            print("Exiting data collection for this class.")
            break

cap.release()
cv2.destroyAllWindows()
