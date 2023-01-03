# ashen
turn images into anime
import cv2
import numpy as np
from keras.models import load_model

# Load the AnimeGAN model
model = load_model('animegan.h5')

# Load the input image
image = cv2.imread('input.jpg')

# Preprocess the image for the model
image = cv2.resize(image, (256, 256))
image = np.expand_dims(image, axis=0)

# Generate the anime version of the image
anime = model.predict(image)

# Save the output image
cv2.imwrite('output.jpg', anime)
![ce492be5-38c9-412c-94ca-e27b2e03dc0b](https://user-images.githubusercontent.com/121843228/210298524-a9bfc8f7-e986-461a-99c8-c2a4b8400b77.jpg)
