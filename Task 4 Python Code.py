import os

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

data = []
labels = []

base_path = r"D:\DEBDEEP\Dataset"

print("Base Exists:", os.path.exists(base_path))

folders = os.listdir(base_path)

print("\nFolders found:")
for f in folders:
    print(repr(f))

dataset_path = os.path.join(base_path, folders[0])

print("\nDetected dataset path:")
print(dataset_path)

print("\nExists:", os.path.exists(dataset_path))

print("\nGesture folders:")
for g in os.listdir(dataset_path):
    print(g)

IMG_SIZE = 64

for gesture in os.listdir(dataset_path):

    gesture_path = os.path.join(dataset_path, gesture)

    if not os.path.isdir(gesture_path):
        continue

    print("Loading:", gesture)

    for image in os.listdir(gesture_path):

        img_path = os.path.join(gesture_path, image)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img)
        labels.append(gesture)

X = np.array(data)
y = np.array(labels)

print("Images:", X.shape)
print("Labels:", y.shape)

X = X.astype("float32") / 255.0

X = X.reshape(-1, 64, 64, 1)

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("Classes:")
print(encoder.classes_)

num_classes = len(encoder.classes_)

y = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

data = data[:10000]
labels = labels[:10000]

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy =", accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()
