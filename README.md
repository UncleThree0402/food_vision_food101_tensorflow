# Food 101 Image 

## Dataset
Food101 From tensorflow_datasets

### Import Dataset

[`dateset_load`](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/dataset_load.py)
```python
import tensorflow_datasets as tfds

# Check is food101 exists
dataset_list = tfds.list_builders()
print("food101" in dataset_list)

# Download or Load Dataset
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             data_dir="../Dataset")
```
> Data are shuffled while loading in

### Preprocess

#### Check Data Shape & Info
```python
# Print features
print(ds_info.features)

# Extract Class name
class_names = ds_info.features["label"].names
```

> We can know that datatype and shape is not we want.

#### Plot random image
```python
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
```

![random_image_before_preprocess]()

#### Preprocess Function
[`tools/data_processing`]()
##### Function
We need to resize and cast datatype to float datatype.
```python
def raw_data_processing(image, label, image_shape=224):
    # Resize
    image = tf.image.resize(image, [image_shape, image_shape])
    # Cast data type
    image = tf.cast(image, tf.float32)
    return image, label
```

##### Test
```python
# Preprocess data
for image, label in train_data.take(1):
    preprocessed_data = data_processing.raw_data_processing(image, label)
    print(preprocessed_data)
```

```shell

```

#### Batch & Finish
[`tools/batch_prefetch`]()

##### Function
```python
def batch_prefetch(train_data, test_data):
    train_data = train_data.map(map_func=raw_data_processing, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=2000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(map_func=raw_data_processing, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, test_data
```

```python
train_data, test_data = data_processing.batch_prefetch(train_data, test_data)
```

##### Test
```python
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
```

![random_image_after_preprocess]()

## Model

### Description
In this model we will use transfer learning with EfficientNetB0.

### Base Model Layer

```python
# Base model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)  # Freeze top layer
base_model.trainable = False

# Base model layers
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
```

### Input Layer
```python
# Input
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
```
> EfficientNetB0 are built for size 224

### Augmentation Layer
```python
# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2)
], name="augmentation_layer")
```

### baseline_model
> Functional Layers are used
```python
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
```

### Fine Tuning
I unfreeze all feature extractor layer, decrease learning fom 0.001 to 0.00005.

```python
# Load best model state
model_0.load_weights("baseline_model/checkpoint.ckpt")

# Unfreeze
base_model.trainable = True

model_0.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                metrics=["accuracy"])

history_1 = model_0.fit(train_data,
                        epochs=100,
                        initial_epoch=history_0.epoch[-1],
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=int(0.15 * len(test_data)),
                        callbacks=[callbacks.create_tensorboard_callback("food101", "model_1"),
                                   callbacks.create_model_checkpoint("model_1"),
                                   callbacks.create_early_stopping("val_accuracy", 5)])
```

### Result
| Model      | Accuracy |
|------------|----------|
| Base Model | 66.76%   |
| Fine Tuned | 75.19    |