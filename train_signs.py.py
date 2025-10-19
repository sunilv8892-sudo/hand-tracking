import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Correct Path to dataset
DATASET_PATH = r"C:\Users\sunil.v\Desktop\projects\sign recognition\Sign-Language-Digits-Dataset-master\Dataset"

# Parameters
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 10   # <-- fixed

# Load data with generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),  # helps reduce overfitting
    layers.Dense(10, activation="softmax")  # 10 classes (digits 0-9)
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save
model.save("sign_digit_model.h5")
print("✅ Model trained & saved as sign_digit_model.h5")
