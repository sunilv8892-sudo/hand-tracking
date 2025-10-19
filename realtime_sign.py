import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained model
model = tf.keras.models.load_model("sign_digit_model.h5")
IMG_SIZE = (100, 100)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            x_coords = [int(lm.x * w) for lm in hand_lms.landmark]
            y_coords = [int(lm.y * h) for lm in hand_lms.landmark]
            x_min, x_max = max(min(x_coords)-20,0), min(max(x_coords)+20, w)
            y_min, y_max = max(min(y_coords)-20,0), min(max(y_coords)+20, h)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # Preprocess for model
                hand_resized = cv2.resize(hand_img, IMG_SIZE)
                hand_resized = hand_resized / 255.0
                hand_resized = np.expand_dims(hand_resized, axis=0)

                # Predict
                pred = model.predict(hand_resized, verbose=0)
                digit = np.argmax(pred)
                confidence = np.max(pred)

                # Draw result
                cv2.putText(frame, f"Digit: {digit} ({confidence*100:.1f}%)",
                            (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

    # Show
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
