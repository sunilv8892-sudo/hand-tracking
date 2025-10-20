REAL-TIME SIGN LANGUAGE DIGIT RECOGNITION

This project allows real-time recognition of sign language digits (0–9) using MediaPipe, OpenCV, and a Convolutional Neural Network (CNN) trained with TensorFlow/Keras.

PROJECT STRUCTURE

train_signs.py → Trains the CNN model on the sign language dataset
realtime_sign.py → Uses webcam and the trained model to predict digits in real time
hand_tracker.py → Demonstrates real-time hand detection and tracking
sign_digit_model.h5 → Saved model generated after training

REQUIREMENTS

Before running the scripts, install the following dependencies:

pip install opencv-python mediapipe tensorflow numpy

MODEL TRAINING (train_signs.py)

This script trains a CNN model to recognize digits (0–9) from sign language images.

Steps:

Download the “Sign Language Digits Dataset” from Kaggle or GitHub.
Example: https://github.com/ardamavi/Sign-Language-Digits-Dataset

Update the dataset path inside the script:
DATASET_PATH = r"C:\path\to\Sign-Language-Digits-Dataset\Dataset"

Run the script:
python train_signs.py

After training, a model file named “sign_digit_model.h5” will be created.

REAL-TIME SIGN DETECTION (realtime_sign.py)

This script uses your webcam to detect and predict digits in real time using the trained model.

Run the script:
python realtime_sign.py

Controls:

Press ‘Q’ to quit.

The window will display the predicted digit along with a confidence percentage.

HAND TRACKING DEMO (hand_tracker.py)

This script uses MediaPipe to track hand landmarks in real time. It helps test whether the camera detects your hand properly.

Run the script:
python hand_tracker.py

Controls:

Press ‘Q’ to exit the live camera feed.

NOTES

If your camera doesn’t open, try changing the camera index in cv2.VideoCapture(0) to 1 or 2.

You can modify the image size, batch size, or number of epochs in train_signs.py for better accuracy.

Make sure your lighting conditions are good when running the real-time test.

OUTPUT SUMMARY

Script → Output
train_signs.py → Trained model file (sign_digit_model.h5)
realtime_sign.py → Live prediction with confidence percentage
hand_tracker.py → Hand landmark visualization

AUTHOR

Developed by: Sunil V
Purpose: Educational and research project on AI-based gesture recognition.

LICENSE

This project is open-source under the MIT License.
