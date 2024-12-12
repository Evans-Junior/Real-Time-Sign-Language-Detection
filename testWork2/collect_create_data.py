import os
import cv2
import pickle
import mediapipe as mp

# Set up directories
DATA_DIR = './data'
PICKLE_DIR = './pickle_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

number_of_classes = 3
dataset_size = 100

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def collect_data(class_id):
    cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 for different cameras
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return False

    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Check camera connection.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Skipping.")
            continue

        cv2.putText(frame, f'Captured {counter + 1}/{dataset_size}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

def create_dataset(class_id):
    data = []
    labels = []
    class_dir = os.path.join(DATA_DIR, str(class_id))

    if not os.path.exists(class_dir):
        print(f"Class directory {class_dir} does not exist.")
        return

    for img_file in os.listdir(class_dir):
        data_aux = []
        x_, y_ = [], []

        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(class_id)

    # Save the pickle file for the class
    pickle_file_path = os.path.join(PICKLE_DIR, f'class_{class_id}.pickle')
    with open(pickle_file_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Pickle file for class {class_id} saved at: {pickle_file_path}")

if __name__ == "__main__":
    for class_index in range(number_of_classes):
        print(f"Starting collection for class {class_index}")
        if collect_data(class_index):
            print(f"Creating dataset for class {class_index}")
            create_dataset(class_index)

    print("Data collection and dataset creation complete.")
