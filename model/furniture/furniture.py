import keras
import cm
import tensorflow as tf
from keras import layers
from keras.applications import VGG16
import matplotlib.pyplot as plt

tf.experimental.numpy.experimental_enable_numpy_behavior()
tf.compat.v1.enable_eager_execution()

batch_size = 32
img_h = 180
img_w = 180
img_size = (img_h, img_w)
epochs = 50
classes = ["bedframe", "chair", "coffee_tables", "desks", "dining_tables", "dressers", "lamps", "nightstand", "ottoman", "plants", "shelves", "sofas"]

train_ds = keras.utils.image_dataset_from_directory(
    directory="model/data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = keras.utils.image_dataset_from_directory(
    directory="model/data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomContrast(0.25),
])

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_data(image, label): 
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, img_size) 
    return image, label

train_ds = train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001))
])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True,
)

for images, labels in train_ds.take(batch_size):
    sample_images = images.numpy()
    samble_labels = labels.numpy()

w = tf.summary.create_file_writer('logs')

tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=cm.log_confusion_matrix(0, model, sample_images, samble_labels, classes, w)) 

#with w.as_default():
#    tf.summary.image("training data examples", sample_images[:25], max_outputs=batch_size, step=0)
#
model.compile(
    optimizer='Adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[tb_callback, cm_callback],
    verbose=1
)

loss, accuracy = model.evaluate(val_ds)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
