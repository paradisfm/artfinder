import keras
import tensorflow as tf
from keras import layers

batch_size = 16
img_h = 180
img_w = 180

train_ds = keras.utils.image_dataset_from_directory(
        directory="data",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_w, img_h),
        batch_size=batch_size,
    )

val_ds = keras.utils.image_dataset_from_directory(
        directory="data",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_w, img_h),
        batch_size=batch_size,
    )

class_names = train_ds.class_names
#print(class_names)

#for image_batch, labels_batch in train_ds:
#  print(image_batch.shape)
#  print(labels_batch.shape)
#  break
# ----> (16, 180, 180, 3)
# ----> (16,)

normalization_layer = keras.layers.Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = tf.keras.Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='Nadam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=25,
)