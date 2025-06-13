import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


IMG_SIZE = (64, 64)


model = tf.keras.models.load_model("asl_model_basic.h5")


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']


img_path = input("👉 Enter image path to predict: ")
img = image.load_img(img_path, target_size=IMG_SIZE)
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)


pred = model.predict(x)
predicted_class = np.argmax(pred)
confidence = np.max(pred) * 100

print(f" Prediction: {labels[predicted_class]} ({confidence:.2f}% confidence)")
