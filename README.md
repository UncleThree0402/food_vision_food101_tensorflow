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

```shell
FeaturesDict({
    'image': Image(shape=(None, None, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=101),
})
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

```shell
i : 0,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 77,Class name of image : pork_chop
i : 1,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 11,Class name of image : caesar_salad
i : 2,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 96,Class name of image : tacos
i : 3,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 100,Class name of image : waffles
i : 4,Shape of image : (342, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 18,Class name of image : chicken_curry
i : 5,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 59,Class name of image : lasagna
i : 6,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 80,Class name of image : pulled_pork_sandwich
i : 7,Shape of image : (512, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 76,Class name of image : pizza
i : 8,Shape of image : (384, 512, 3),Dtype of image : <dtype: 'uint8'>,Label of image : 22,Class name of image : chocolate_mousse
```

![random_image_before_preprocess](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/ribp.png)

#### Preprocess Function

[`tools/data_processing`](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/tools/data_processing.py)

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

#### Batch & Finish

[`tools/data_processing`](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/tools/data_processing.py)

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

```shell
i : 0,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 82,Class name of image : ravioli
i : 1,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 73,Class name of image : panna_cotta
i : 2,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 77,Class name of image : pork_chop
i : 3,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 88,Class name of image : seaweed_salad
i : 4,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 79,Class name of image : prime_rib
i : 5,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 9,Class name of image : breakfast_burrito
i : 6,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 25,Class name of image : club_sandwich
i : 7,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 23,Class name of image : churros
i : 8,Shape of image : (224, 224, 3),Dtype of image : <dtype: 'float32'>,Label of image : 48,Class name of image : greek_salad
```

![random_image_after_preprocess](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/riap.png)

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

![loss_b](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Baseline%20Model%20_loss.png)
![accuracy_b](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Baseline%20Model%20_accuracy.png)

### Fine-Tuning Step One

I unfreeze all feature extractor layer, decrease learning from 0.001 to 0.0001.

```python
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
```

![loss_1](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Model%201%20_loss.png)
![accuracy_1](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Model%201%20_accuracy.png)

### Fine-Tuning Step Two

I unfreeze all feature extractor layer, decrease learning from 0.0001 to 0.00001.

```python
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
```

![loss_2](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Model%202%20_loss.png)
![accuracy_2](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/Model%202%20_accuracy.png)

### Result

| Model               | Accuracy |
|---------------------|----------|
| Base Model          | 67.44%   |
| Fine Tuned Step One | 75.28%   |
| Fine Tuned Step Two | 77.51%   |

#### Base Model

![f1_b](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_0_f1-score.png)
![precision_b](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_0_precision.png)
![recall_b](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_0_recall.png)

#### Fine Tune 1

![f1_1](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_1_f1-score.png)
![precision_1](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_1_precision.png)
![recall_1](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_1_recall.png)

#### Fine Tune 2

![f1_2](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_2_f1-score.png)
![precision_2](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_2_precision.png)
![recall_2](https://github.com/UncleThree0402/food_vision_food101_tensorflow/blob/master/Image/model_2_recall.png)