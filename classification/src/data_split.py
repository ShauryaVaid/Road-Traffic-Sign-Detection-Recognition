import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


NUM_CLASSES = 43
IMAGE_SIZE = 32


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

            # Skip images that cannot be read
            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

            # Store image and label
            images.append(image)
            labels.append(class_id)

    # Convert to NumPy arrays
    X = np.array(images)
    y = np.array(labels)

    return X, y


# Path to Train folder
data_path = "data/Train"


# Load actual dataset
X, y = load_dataset(data_path)


print("\nDataset loaded successfully!")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)


# Split into training and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Split temporary data into validation and test data
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