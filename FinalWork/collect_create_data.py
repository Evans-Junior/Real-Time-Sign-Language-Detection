import os
import cv2
import pickle
import mediapipe as mp

# Set up directories
DATA_DIR = './data'  # Directory to store captured images
PICKLE_DIR = './pickle_data'  # Directory to store processed data in pickle files

# Create the directories if they don't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

number_of_classes = 3  # Total number of gesture classes to collect
dataset_size = 100  # Number of images to collect per class

# Initialize Mediapipe Hands for hand detection and landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks on images
hands = mp_hands.Hands(
    static_image_mode=True,  # Whether to run on static images
    min_detection_confidence=0.3  # Minimum confidence threshold for detections
)

def collect_data(class_id):
    """
    Captures images for a specific gesture class using the webcam.

    Args:
        class_id (int): The class ID of the gesture being captured.

    Returns:
        bool: True if data collection was successful, False otherwise.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera (0). Adjust index for other cameras.
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return False

    # Create a directory for the class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}')

    # Display a ready screen before starting data capture
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Check camera connection.")
            break

        # Display a ready message
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Wait for the user to press 'Q' to start
            break

    # Capture images for the dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Skipping.")
            continue

        # Display progress
        cv2.putText(frame, f'Captured {counter + 1}/{dataset_size}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        # Save the captured image to the class directory
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Allow early exit by pressing 'Q'
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close any OpenCV windows
    return True

def create_dataset(class_id):
    """
    Processes the captured images to extract hand landmarks and save them as a dataset.

    Args:
        class_id (int): The class ID for which the dataset is being created.
    """
    data = []  # List to store extracted landmark features
    labels = []  # List to store class labels
    class_dir = os.path.join(DATA_DIR, str(class_id))  # Path to the class's images

    if not os.path.exists(class_dir):
        print(f"Class directory {class_dir} does not exist.")
        return

    # Iterate through all images in the class directory
    for img_file in os.listdir(class_dir):
        data_aux = []  # Temporary list to store normalized landmark data
        x_, y_ = [], []  # Lists to store raw landmark coordinates

        # Load the image
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert the image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)  # Detect hand landmarks in the image

        if results.multi_hand_landmarks:
            # Process each detected hand in the image
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)  # Store x-coordinates
                    y_.append(landmark.y)  # Store y-coordinates

                # Normalize the landmarks based on the bounding box
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)  # Append processed data
            labels.append(class_id)  # Append corresponding class label

    # Save the processed data and labels to a pickle file
    pickle_file_path = os.path.join(PICKLE_DIR, f'class_{class_id}.pickle')
    with open(pickle_file_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Pickle file for class {class_id} saved at: {pickle_file_path}")

if __name__ == "__main__":
    """
    Main function to collect data for all classes and create datasets.
    """
    for class_index in range(number_of_classes):
        print(f"Starting collection for class {class_index}")
        if collect_data(class_index):  # Collect images for the class
            print(f"Creating dataset for class {class_index}")
            create_dataset(class_index)  # Create and save the dataset

    print("Data collection and dataset creation complete.")
