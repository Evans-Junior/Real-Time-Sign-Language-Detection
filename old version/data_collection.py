import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import absl.logging

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "test/M"
counter = 0

# Ensure output folder exists
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from webcam.")
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure coordinates stay within image bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            print("Invalid cropped image; skipping frame.")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    print(f"Key pressed: {key}")  # Debugging keypresses

    if key == ord("s"):
        if 'imgWhite' in locals() and imgWhite is not None:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Image {counter} saved successfully!")
        else:
            print("imgWhite is not ready, cannot save.")
