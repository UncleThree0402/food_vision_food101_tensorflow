import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
from tools import data_processing, callbacks, plot_graph

(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             data_dir="../Dataset")

# Print features
print(ds_info.features)

# Extract Class name
class_names = ds_info.features["label"].names

# Plot Random Image
plt.figure(figsize=(15, 15))

for i, data in enumerate(train_data.take(9)):
    image, label = data
    print(f"i : {i},"
          f"Shape of image : {image.shape},"
          f"Dtype of image : {image.dtype},"
          f"Label of image : {label},"
          f"Class name of image : {class_names[label.numpy()]}")

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_names[label.numpy()])
    plt.axis(False)
plt.show()

# Preprocess data
for image, label in train_data.take(1):
    preprocessed_data = data_processing.raw_data_processing(image, label)
    print(preprocessed_data)

# Batching & finish preprocess
train_data, test_data = data_processing.batch_prefetch(train_data, test_data)

processed_train_data = train_data.unbatch()

plt.figure(figsize=(15, 15))

for i, data in enumerate(processed_train_data.take(9)):
    image, label = data
    print(f"i : {i},"
          f"Shape of image : {image.shape},"
          f"Dtype of image : {image.dtype},"
          f"Label of image : {label},"
          f"Class name of image : {class_names[label.numpy()]}")
    plt.subplot(3, 3, i + 1)
    plt.imshow(image / 255.)
    plt.title(class_names[label.numpy()])
    plt.axis(False)
plt.show()

tf.random.set_seed(42)

# Base model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)  # Freeze top layer
base_model.trainable = False

# Base model layers
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)

# Input
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2)
], name="augmentation_layer")

# Baseline model
x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAvgPool2D(name="global_avg_pool_2d")(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_0 = model_0.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=int(0.5 * len(test_data)),
                        callbacks=[callbacks.create_tensorboard_callback("food101", "baseline_model"),
                                   callbacks.create_model_checkpoint("baseline_model")])

model_0.load_weights("baseline_model/checkpoint.ckpt")
result_0 = model_0.evaluate(test_data)

model_0.save("./models/model_0.h5", save_format='h5')

plot_graph.plot_loss_curves(history_0, "Baseline Model ")
plot_graph.plot_accuracy_curves(history_0, "Baseline Model ")

pred_probs = model_0.predict(test_data, verbose=1)

y_labels, pred_classes = data_processing.create_y_labels_y_pred(pred_probs=pred_probs, test_data=test_data)

plot_graph.plot_confusion_matrix(y_true=y_labels, y_preds=pred_classes, name="model_0", classes=class_names,
                                 figsize=(100, 100), text_size=20)

plot_graph.plot_classification_report(y_true=y_labels, y_pred=pred_classes, name="model_0", class_names=class_names)

# After base_model trained
# model_0 = tf.keras.models.load_model("./models/model_0.h5")
# model_0.load_weights("baseline_model/checkpoint.ckpt")
# model_0.evaluate(test_data)

# model_1
model_0.load_weights("baseline_model/checkpoint.ckpt")

base_model.trainable = True

model_0.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=["accuracy"])

history_1 = model_0.fit(train_data,
                        epochs=100,
                        initial_epoch=history_0.epoch[-1],
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=int(0.15 * len(test_data)),
                        callbacks=[callbacks.create_tensorboard_callback("food101", "model_1"),
                                   callbacks.create_model_checkpoint("model_1"),
                                   callbacks.create_early_stopping("val_accuracy", 3)])

model_0.load_weights("model_1/checkpoint.ckpt")
result_1 = model_0.evaluate(test_data)

model_0.save("./models/model_1.h5", save_format='h5')

plot_graph.plot_loss_curves(history_1, "Model 1 ")
plot_graph.plot_accuracy_curves(history_1, "Model 1 ")

pred_probs = model_0.predict(test_data, verbose=1)

y_labels, pred_classes = data_processing.create_y_labels_y_pred(pred_probs=pred_probs, test_data=test_data)

plot_graph.plot_confusion_matrix(y_true=y_labels, y_preds=pred_classes, name="model_1", classes=class_names,
                                 figsize=(100, 100), text_size=20)

plot_graph.plot_classification_report(y_true=y_labels, y_pred=pred_classes, name="model_1", class_names=class_names)

# model_2
model_0.load_weights("model_1/checkpoint.ckpt")

base_model.trainable = True

model_0.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                metrics=["accuracy"])

history_2 = model_0.fit(train_data,
                        epochs=100,
                        initial_epoch=history_1.epoch[-1],
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=int(0.15 * len(test_data)),
                        callbacks=[callbacks.create_tensorboard_callback("food101", "model_2"),
                                   callbacks.create_model_checkpoint("model_2"),
                                   callbacks.create_early_stopping("val_accuracy", 3)])

model_0.load_weights("model_2/checkpoint.ckpt")
result_2 = model_0.evaluate(test_data)

model_0.save("./models/model_2.h5", save_format='h5')

plot_graph.plot_loss_curves(history_1, "Model 2 ")
plot_graph.plot_accuracy_curves(history_1, "Model 2 ")

pred_probs = model_0.predict(test_data, verbose=1)

y_labels, pred_classes = data_processing.create_y_labels_y_pred(pred_probs=pred_probs, test_data=test_data)

plot_graph.plot_confusion_matrix(y_true=y_labels, y_preds=pred_classes, name="model_2", classes=class_names,
                                 figsize=(100, 100), text_size=20)

plot_graph.plot_classification_report(y_true=y_labels, y_pred=pred_classes, name="model_2", class_names=class_names)

print(f"Result 0 : {result_0}")
print(f"Result 1 : {result_1}")
print(f"Result 2 : {result_2}")
