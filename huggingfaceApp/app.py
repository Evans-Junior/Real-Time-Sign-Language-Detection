import pickle  # For loading the trained model
import cv2  # For camera access and image processing
import mediapipe as mp  # For hand tracking and detection
import numpy as np  # For numerical operations

# Load the trained model from a pickle file
model_path = './model.p'  # Path to the trained model
model = pickle.load(open(model_path, 'rb'))  # Load the model from the file

# Check if the model is loaded correctly
if model is None:
    raise ValueError("Model could not be loaded. Ensure 'model.p' contains a valid model.")

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands  # Access MediaPipe's hands module
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined styles for landmarks and connections
hands = mp_hands.Hands(
    static_image_mode=False,  # Enable dynamic mode for real-time detection
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)

# Define a dictionary to map class indices to gesture labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K'
}

# Initialize the camera
cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
if not cap.isOpened():  # Check if the camera is accessible
    print("Error: Camera could not be opened. Check the camera index.")
    exit()  # Exit the program if the camera cannot be opened

# Start the real-time loop for hand gesture detection and prediction
while True:
    data_aux = []  # Auxiliary list to store normalized hand landmark data
    x_ = []  # List to store x-coordinates of hand landmarks
    y_ = []  # List to store y-coordinates of hand landmarks

    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret or frame is None:  # Check if the frame is valid
        print("Error: Failed to read a frame from the camera.")
        break

    H, W, _ = frame.shape  # Get the frame dimensions (height, width, channels)

    # Process the frame with MediaPipe Hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    results = hands.process(frame_rgb)  # Detect hands in the frame

    if results.multi_hand_landmarks:  # Check if any hands are detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,  # Frame to draw on
                hand_landmarks,  # Hand landmarks data
                mp_hands.HAND_CONNECTIONS,  # Connections between landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Style for landmarks
                mp_drawing_styles.get_default_hand_connections_style()  # Style for connections
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)  # Append x-coordinate to the list
                y_.append(landmark.y)  # Append y-coordinate to the list

            # Normalize the landmarks to avoid coordinate overflow
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append((landmark.x - min_x) / (max_x - min_x + 1e-6))
                data_aux.append((landmark.y - min_y) / (max_y - min_y + 1e-6))

            # Predict the gesture using the trained model
            try:
                prediction = model.predict([np.asarray(data_aux)])  # Make a prediction with normalized data
                predicted_character = labels_dict[int(prediction[0])]  # Map the prediction to the corresponding label

                # Calculate bounding box coordinates for the detected hand
                x1 = int(min_x * W)  # Convert normalized min x to pixel
                y1 = int(min_y * H)  # Convert normalized min y to pixel
                x2 = int(max_x * W)  # Convert normalized max x to pixel
                y2 = int(max_y * H)  # Convert normalized max y to pixel

                # Draw the bounding box and predicted gesture on the frame
                cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 255, 0), 2)  # Draw the bounding box
                cv2.putText(
                    frame, predicted_character, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )  # Display the predicted gesture

            except Exception as e:
                print(f"Prediction error: {e}")

    # Display the processed frame
    cv2.imshow('Real-Time Hand Gesture Recognition', frame)

    if cv2.waitKey(1) == ord('q'):  # Exit the loop if 'q' is pressed
        break

# Release the camera and close OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV display windows
