import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMAGE_SIZE = (300, 300)
train_path = "Datasets/train"
test_path = "Datasets/test"
BATCH_SIZE = 16
EPOCHS = 25

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_set = train_gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_set = test_gen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.35)(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    "pneumonia_optimized_Dibyendu.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    factor=0.2,
    min_lr=1e-7,
    verbose=1
)

r = model.fit(
    train_set,
    validation_data=test_set,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, lr_reduce]
)

model.save("pneumonia_final_Dibyendu.h5")
