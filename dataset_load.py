import tensorflow_datasets as tfds

dataset_list = tfds.list_builders()
print("food101" in dataset_list)

(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             data_dir="../Dataset")