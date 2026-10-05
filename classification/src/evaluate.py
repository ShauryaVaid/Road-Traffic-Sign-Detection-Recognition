import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# -----------------------------
# 1. Settings
# -----------------------------

IMAGE_SIZE = 32
NUM_CLASSES = 43

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(BASE_DIR, "data", "Test.csv")

TEST_FOLDER = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "traffic_sign_model.keras"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------
# 2. Load trained model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# -----------------------------
# 3. Load official test CSV
# -----------------------------

test_data = pd.read_csv(TEST_CSV)

print("\nTest data loaded successfully!")
print("Number of test images:", len(test_data))


# -----------------------------
# 4. Load and preprocess images
# -----------------------------

images = []
labels = []

for index, row in test_data.iterrows():

    # Get image path
    image_path = os.path.join(
        TEST_FOLDER,
        row["Path"]
    )

    # Read image
    image = cv2.imread(image_path)

    # Skip image if it cannot be read
    if image is None:
        print("Could not read:", image_path)
        continue

    # -------------------------
    # Crop using ROI coordinates
    # -------------------------

    x1 = int(row["Roi.X1"])
    y1 = int(row["Roi.Y1"])
    x2 = int(row["Roi.X2"])
    y2 = int(row["Roi.Y2"])

    image = image[y1:y2, x1:x2]

    # Skip invalid crop
    if image.size == 0:
        print("Invalid ROI:", image_path)
        continue

    # -------------------------
    # Resize image
    # -------------------------

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Store image
    images.append(image)

    # Store actual class
    labels.append(int(row["ClassId"]))


# Convert to NumPy arrays
X_test = np.array(images)
y_test = np.array(labels)


print("\nTest images after ROI cropping:")
print("Test images shape:", X_test.shape)
print("Test labels shape:", y_test.shape)


# -----------------------------
# 5. Normalize images
# -----------------------------

X_test = X_test.astype("float32") / 255.0

print("\nImages normalized.")
print("Minimum pixel value:", X_test.min())
print("Maximum pixel value:", X_test.max())


# -----------------------------
# 6. Evaluate model
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nOfficial Test Loss:", test_loss)

print(
    "Official Test Accuracy:",
    test_accuracy
)

print(
    "Official Test Accuracy (%):",
    test_accuracy * 100
)


# -----------------------------
# 7. Make predictions
# -----------------------------

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# -----------------------------
# 8. Classification report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )
)


# -----------------------------
# 9. Confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar()

plt.xticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES),
    rotation=90
)

plt.yticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES)
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "confusion_matrix.png"
    )
)

plt.show()