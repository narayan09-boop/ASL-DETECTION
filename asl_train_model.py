import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

train_dir = r"C:\Users\nk090\OneDrive\Desktop\Unified mentor\ASL DETECTION\data\raw"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

print(f" Loading training data from: {train_dir}")

train_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

if train_gen.num_classes < 1:
    raise ValueError(" No class folders found in training data. Check your 'raw' folder structure.")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"\n Classes found in training data: {train_gen.class_indices}")
print(f" Total classes: {train_gen.num_classes}")

print("\n Model Summary:")
model.summary()

print("\n Starting training...\n")
history = model.fit(train_gen, epochs=EPOCHS)

print("\n Training complete. Saving model as 'asl_model_basic.h5'")
model.save("asl_model_basic.h5")

print(" Model saved successfully!")
print(" Class Indices:", train_gen.class_indices)
print(" Training and model saving process completed successfully!")