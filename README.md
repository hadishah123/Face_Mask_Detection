# 😷 Face Mask Detection

**A Python project to detect face masks using deep learning with TensorFlow and OpenCV.**

---

## 📁 Project Structure
```bash
Face_Mask_Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── model/
├── train_model.py
├── detect_mask.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

**1️⃣ Clone the Repository**

```bash
git clone https://github.com/sofialtamsh/Face_Mask_Detection.git
```

**2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
```

**3️⃣ Activate the Virtual Environment**

Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```

**4️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```


📊 Dataset Format

Ensure your dataset is structured as follows:
```bash
dataset/
├── with_mask/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── without_mask/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Guidelines:
- Images should be clear and front-facing when possible
- Supported formats: .jpg, .png
- Balanced classes improve model accuracy


## 🧠 Model Architecture

- Convolutional Neural Network (CNN)
- Image preprocessing using OpenCV
- Binary classification:
- With Mask
- Without Mask
- Trained using TensorFlow/Keras
- Model saved in the model/ directory after training


## 🚀 Usage
**🔹 1. Train the Model**
```bash
python train_model.py
```

- Loads images from the dataset
- Trains the CNN model
- Saves the trained model in model


**🔹 2. Run Face Mask Detection**
```bash
python detect_mask.py
```
Uses webcam or video feed, Detects faces in real-time
* Displays:
* Green box → Mask detected
* Red box → No mask detected

🖼️ Example Output
```bash
[INFO] Starting video stream...
[INFO] Mask detected with 98.45% confidence
```

On-screen:
- Bounding box around face
- Label: Mask / No Mask
- Confidence percentage

## 📦 Dependencies
```bash
TensorFlow
OpenCV-Python
NumPy
Matplotlib
Scikit-learn
```

## 📝 Notes
- Always activate the virtual environment before running scripts
- Ensure proper lighting for better face detection
- You can improve accuracy by adding more training images


## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to change.

## 🪪 License
MIT License © 2025 [Sofi Altamsh](https://github.com/sofialtamsh)