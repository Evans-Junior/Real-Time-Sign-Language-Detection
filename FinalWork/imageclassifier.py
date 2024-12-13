# Import necessary libraries
import pickle  # For loading the trained model
import cv2  # For camera access and image processing
import mediapipe as mp  # For hand tracking and detection
import numpy as np  # For numerical operations

# Load the trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  # Load the model dictionary from 'model.p'
model = model_dict['model']  # Extract the trained model from the dictionary

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands  # Access MediaPipe's hands module
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined styles for landmarks and connections
hands = mp_hands.Hands(
    static_image_mode=True,  # Use static image mode for consistent hand detection
    min_detection_confidence=0.3  # Minimum confidence required for hand detection
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

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):  # Loop through each landmark
                x = hand_landmarks.landmark[i].x  # Get the normalized x-coordinate
                y = hand_landmarks.landmark[i].y  # Get the normalized y-coordinate
                x_.append(x)  # Append x-coordinate to the list
                y_.append(y)  # Append y-coordinate to the list

            # Normalize the landmarks to avoid coordinate overflow
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x-coordinates
                data_aux.append(y - min(y_))  # Normalize y-coordinates

        # Calculate bounding box coordinates for the detected hand
        x1 = int(min(x_) * W) - 10  # Convert normalized min x to pixel and add margin
        y1 = int(min(y_) * H) - 10  # Convert normalized min y to pixel and add margin
        x2 = int(max(x_) * W) - 10  # Convert normalized max x to pixel and add margin
        y2 = int(max(y_) * H) - 10  # Convert normalized max y to pixel and add margin

        # Predict the gesture using the trained model
        prediction = model.predict([np.asarray(data_aux)])  # Make a prediction with normalized data
        predicted_character = labels_dict[int(prediction[0])]  # Map the prediction to the corresponding label

        # Draw the bounding box and predicted gesture on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw the bounding box
        cv2.putText(
            frame, predicted_character, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
        )  # Display the predicted gesture

    # Display the processed frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):  # Exit the loop if 'q' is pressed
        break

# Release the camera and close OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV display windows
