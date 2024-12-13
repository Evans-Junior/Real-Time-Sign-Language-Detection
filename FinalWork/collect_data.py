# Import necessary libraries
import os  # For file and directory operations
import cv2  # For capturing images and video frames
import mediapipe as mp  # For hand tracking and detection

# MediaPipe Hands setup
mp_hands = mp.solutions.hands  # Access MediaPipe's hands module
hands = mp_hands.Hands(
    static_image_mode=False,  # Use real-time detection (not static images)
    max_num_hands=1,  # Detect only one hand at a time
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
)

# Directory to save collected data
DATA_DIR = './test'
if not os.path.exists(DATA_DIR):  # Check if the directory exists
    os.makedirs(DATA_DIR)  # Create the directory if it doesn't exist

# Number of gesture classes to collect and number of samples per class
number_of_classes = 10  # Total number of classes
dataset_size = 100  # Number of images per class

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use the default camera (0). Adjust index for other cameras.
if not cap.isOpened():  # Check if the camera is accessible
    print("Error: Camera could not be opened. Check the camera index.")
    exit()  # Exit the program if the camera cannot be opened

# Loop over each class to collect data
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')  # Inform the user which class is being collected

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret or frame is None:  # Check if the frame is valid
            print("Error: Failed to read a frame from the camera.")
            break

        # Display a message on the frame
        cv2.putText(
            frame, 'Ready? Press "Q" ! :)', (100, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA
        )
        cv2.imshow('frame', frame)  # Show the frame in a window
        if cv2.waitKey(25) == ord('q'):  # Wait for the user to press 'Q' to start
            break

    # Collect `dataset_size` images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret or frame is None:  # Check if the frame is valid
            print("Error: Failed to read a frame from the camera.")
            break

        # Convert the frame to RGB (MediaPipe works with RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)  # Detect hands in the frame

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Draw detected hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Save the current frame as an image in the class directory
            cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
            counter += 1  # Increment the counter
            print(f"Image {counter} saved for class {j}")  # Inform the user about progress
        else:
            print("No hand detected, skipping frame.")  # Skip the frame if no hand is detected

        # Display the frame with landmarks
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):  # Allow the user to exit the loop early by pressing 'Q'
            print("Exiting data collection for this class.")
            break

# Release the webcam and close all OpenCV windows
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV display windows
