# detect_mask.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/mask_detector.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1,100,100,3))

        prediction = model.predict(face_input)
        label = np.argmax(prediction)
        text = "Mask" if label == 0 else "No Mask"
        color = (0, 255, 0) if label == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
