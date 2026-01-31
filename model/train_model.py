import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from tensorflow.keras.applications import MobileNetV2 #type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #type:ignore
from tensorflow.keras.models import Model #type:ignore
import os, json

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "plantvillage")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 🔹 Data Augmentation (KEY FOR REAL-WORLD IMAGES)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# 🔹 Base Model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 🔹 Fine-tuning (IMPORTANT)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), #type:ignore
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# 🔹 Save model
model.save(os.path.join(BASE_DIR, "plant_disease_model.h5"))

# 🔹 Save class indices
with open(os.path.join(BASE_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f)

print(" Training complete")
