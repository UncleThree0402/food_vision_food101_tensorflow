import tensorflow as tf


# Preprocess image
def raw_data_processing(image, label, image_shape=224):
    # Resize
    image = tf.image.resize(image, [image_shape, image_shape])
    # Cast data type
    image = tf.cast(image, tf.float32)
    return image, label


def batch_prefetch(train_data, test_data):
    train_data = train_data.map(map_func=raw_data_processing, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=2000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(map_func=raw_data_processing, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, test_data


def create_y_labels_y_pred(pred_probs, test_data):
    pred_classes = tf.argmax(pred_probs, axis=1)

    y_labels = []

    for image, label in test_data.unbatch():
        y_labels.append(label.numpy())

    return y_labels, pred_classes
