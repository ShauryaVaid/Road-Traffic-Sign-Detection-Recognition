import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# -----------------------------
# 1. Settings
# -----------------------------

NUM_CLASSES = 43
IMAGE_SIZE = 32

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "Train")

# Model and results paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_sign_model.keras")
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")

# Create plot folder if it does not exist
os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------
# 2. Load dataset
# -----------------------------

def load_dataset(data_path):

    images = []
    labels = []

    # Go through classes 0 to 42
    for class_id in range(NUM_CLASSES):

        class_path = os.path.join(data_path, str(class_id))

        print("Loading class:", class_id)

        # Go through all images in the class folder
        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            # Read image
            image = cv2.imread(image_path)

            # Skip image if it cannot be read
            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

            # Store image and label
            images.append(image)
            labels.append(class_id)

    X = np.array(images)
    y = np.array(labels)

    return X, y


# Load actual traffic-sign dataset
X, y = load_dataset(DATA_PATH)

print("\nDataset loaded successfully!")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)


# -----------------------------
# 3. Normalize images
# -----------------------------

X = X.astype("float32") / 255.0

print("\nImages normalized.")
print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())


# -----------------------------
# 4. Split dataset
# -----------------------------

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temporary data into
# 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# -----------------------------
# 5. Create CNN
# -----------------------------

model = models.Sequential([

    # Input image: 32 × 32 × 3
    layers.Input(shape=(32, 32, 3)),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into one-dimensional vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    layers.Dropout(0.5),

    # Output layer: 43 traffic-sign classes
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# -----------------------------
# 6. Display CNN structure
# -----------------------------

model.summary()


# -----------------------------
# 7. Compile CNN
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 8. Train CNN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)


# -----------------------------
# 9. Evaluate CNN
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest loss:", test_loss)
print("Test accuracy:", test_accuracy)


# -----------------------------
# 10. Save trained model
# -----------------------------

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)


# -----------------------------
# 11. Plot accuracy
# -----------------------------

plt.figure()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training",
    "Validation"
])

plt.savefig(
    os.path.join(PLOT_DIR, "accuracy.png")
)

plt.show()


# -----------------------------
# 12. Plot loss
# -----------------------------

plt.figure()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Training",
    "Validation"
])

plt.savefig(
    os.path.join(PLOT_DIR, "loss.png")
)

plt.show()