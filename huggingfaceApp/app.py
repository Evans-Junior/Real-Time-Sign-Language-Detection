import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import pickle
from huggingface_hub import hf_hub_download
from PIL import Image

# Download the model from Hugging Face Hub
model_path = hf_hub_download(repo_id="KEJEvans/sign-language-model", filename="model.pkl")

# Load the model directly (if not wrapped in a dictionary)
model = pickle.load(open(model_path, 'rb'))

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K'}

# Function to make predictions from the uploaded image
def predict_image(image):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x
                data_aux.append(y - min(y_))  # Normalize y

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        return predicted_character
    else:
        return "No hand detected"

# Streamlit app layout
st.title("Sign Language Gesture Recognition")
st.write("Upload an image of a hand gesture to recognize it.")

# Image uploader
image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if image is not None:
    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict and display result
    predicted_character = predict_image(Image.open(image))
    st.subheader(f"Predicted Gesture: {predicted_character}")

# Run the app using streamlit run <your_script.py>
