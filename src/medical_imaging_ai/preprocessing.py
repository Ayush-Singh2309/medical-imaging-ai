
import tensorflow as tf

from keras.applications.densenet import preprocess_input

from medical_imaging_ai.constants import CLASSES, IMAGE_SIZE


def load_image(image_path, label):
    """
    Load and preprocess a single chest X-ray image.

    Returns:
        model_image: Preprocessed RGB image.
        label: Multi-hot encoded disease labels.
    """

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.image.grayscale_to_rgb(image)

    image = tf.cast(image, tf.float32)

    model_image = preprocess_input(image)

    return model_image, label


def create_dataset(
    dataframe,
    image_dir,
    batch_size=32,
    training=False,
    shuffle_buffer=2048
):
    """
    Create an efficient TensorFlow dataset.

    Args:
        dataframe: DataFrame containing image_index and disease labels.
        image_dir: Directory containing NIH PNG images.
        batch_size: Number of images per batch.
        training: Whether this is a training dataset.
        shuffle_buffer: Number of samples used for shuffling.

    Returns:
        A tf.data.Dataset.
    """

    image_paths = [
        str(image_dir / filename)
        for filename in dataframe["image_index"]
    ]

    labels = dataframe[CLASSES].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer, len(image_paths)),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
