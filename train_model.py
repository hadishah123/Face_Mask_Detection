# train_model.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
img_size = 100
categories = ["with_mask", "without_mask"]

# Load dataset
data = []
labels = []

for category in categories:
    path = os.path.join("dataset", category)
    label = categories.index(category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (img_size, img_size))
        data.append(image)
        labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)

# Save model
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/mask_detector.h5")
print("Model trained and saved successfully!")
