import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.2,  # Lower threshold for detection
    max_num_hands=1
)

DATA_DIR = './test'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(folder_path):  # Skip non-directories
        continue

    for img_path in os.listdir(folder_path):
        img_file_path = os.path.join(folder_path, img_path)
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image
            continue

        # Read and process the image
        img = cv2.imread(img_file_path)
        if img is None:
            print(f"Could not read image: {img_file_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Optional resize to standardize input
        img_rgb = cv2.resize(img_rgb, (300, 300))

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_, y_ = [], []

                # Extract and normalize landmarks
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                if not x_ or not y_:
                    print(f"No valid landmarks in image: {img_file_path}")
                    continue

                x_min = min(x_)
                y_min = min(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - x_min)  # Normalize x
                    data_aux.append(lm.y - y_min)  # Normalize y

                data.append(data_aux)  # Add features
                labels.append(dir_)  # Add class label

print(f"Extracted {len(data)} samples with {len(labels)} labels.")

# Save extracted keypoints and labels for training
with open('keypoints_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Keypoints extraction complete!")
