import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model_path = 'model/cat_dog_classifier.h5'
if not os.path.exists(model_path):
    print("❌ Model not found. Train it first using main.py.")
    exit()

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Step 1: Pick a random image from dataset
dataset_path = 'dataset'
categories = ['cats', 'dogs']

# Randomly choose a category and an image
category = random.choice(categories)
folder_path = os.path.join(dataset_path, category)
random_image = random.choice(os.listdir(folder_path))
img_path = os.path.join(folder_path, random_image)

print(f"🎯 Selected image: {img_path}")

# Step 2: Preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Step 3: Predict
prediction = model.predict(img_array)[0][0]
label = "🐱 Cat" if prediction < 0.5 else "🐶 Dog"
print(f"Prediction: {label}")

# Step 4: Show image with prediction
plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {label}")
plt.axis('off')
plt.show()