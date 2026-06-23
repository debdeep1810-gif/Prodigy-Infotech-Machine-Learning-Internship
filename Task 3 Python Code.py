import os
import cv2
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Main dataset folder
DATADIR = r"D:\DEBDEEP\train"

IMG_SIZE = 64

if not os.path.exists(DATADIR):
    print("Folder NOT found!")
    exit()

print("Folder found successfully!")
print("Current path:", DATADIR)

data = []

# Categories
categories = ["cats", "dogs"]

for category in categories:

    folder_path = os.path.join(DATADIR, category)

    if category == "cats":
        label = 0
    else:
        label = 1

    print(f"\nLoading {category} images...")

    for img in os.listdir(folder_path)[:1000]:

        try:
            path = os.path.join(folder_path, img)

            img_array = cv2.imread(path)

            if img_array is None:
                continue

            img_array = cv2.cvtColor(
                img_array,
                cv2.COLOR_BGR2GRAY
            )

            img_array = cv2.resize(
                img_array,
                (IMG_SIZE, IMG_SIZE)
            )

            data.append([img_array, label])

        except Exception as e:
            print("Error:", img, e)

print("\nTotal Images Loaded =", len(data))

if len(data) == 0:
    print("No images loaded!")
    exit()

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features.flatten())
    y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
