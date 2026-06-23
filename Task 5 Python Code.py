import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# ==========================
# DATA GENERATORS
# ==========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================
# TRAIN
# ==========================

train_generator = train_datagen.flow_from_directory(
    r"D:\DEBDEEP\Dataset\train",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# ==========================
# VALIDATION
# ==========================

validation_generator = val_datagen.flow_from_directory(
    r"D:\DEBDEEP\Dataset\validation",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# ==========================
# TEST
# ==========================

test_generator = test_datagen.flow_from_directory(
    r"D:\DEBDEEP\Dataset\test",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ==========================
# CNN MODEL
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(
    Dense(
        256,
        activation='relu'
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        train_generator.num_classes,
        activation='softmax'
    )
)

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# ==========================
# TEST ACCURACY
# ==========================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\nTest Accuracy:", test_accuracy)

# ==========================
# SAVE MODEL
# ==========================

model.save(
    r"D:\DEBDEEP\food_classifier.keras"
)

print("\nModel saved successfully.")

# ==========================
# CLASS NAMES
# ==========================

class_names = list(
    train_generator.class_indices.keys()
)

print("\nClasses:")
print(class_names)

# AUTOMATIC TEST IMAGE
# ==========================

test_image = None

for root, dirs, files in os.walk(r"D:\DEBDEEP\Dataset\test"):
    for file in files:
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            test_image = os.path.join(root,file)
            break
    if test_image:
        break

if test_image:

    print("\nTesting image:")
    print(test_image)

    img = image.load_img(
        test_image,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    food_name = class_names[predicted_class]

    print("\nPredicted Food:", food_name)

else:
    print("No image found in test folder.")

# ==========================
# CALORIES
# ==========================

calories = {
    'chicken_curry': 240,
    'chocolate_cake': 371,
    'fish_and_chips': 290,
    'hamburger': 295,
    'ice_cream': 207,
    'pad_thai': 300,
    'pizza': 285,
    'ramen': 436,
    'sushi': 200,
    'tacos': 226
}

if food_name in calories:
    print(
        "Estimated Calories:",
        calories[food_name],
        "kcal"
    )

# ==========================
# ACCURACY GRAPH
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title(
    "Food Classification Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()
