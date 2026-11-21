import tensorflow as tf
import keras
import numpy as np

import matplotlib.pyplot as plt

from medical_imaging_ai.constants import CLASSES, IMAGE_SIZE
from keras.applications.densenet import preprocess_input


def load_image_for_gradcam(image_path):
    """
    Load one image using the same preprocessing
    applied during model training.
    """

    image = tf.io.read_file(str(image_path))

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

    image = preprocess_input(image)

    return tf.expand_dims(image, axis=0)


def generate_gradcam(model, image_tensor, class_index):

    if not 0 <= class_index < len(CLASSES):
        raise ValueError("Invalid class index.")

    # Retrieve the nested DenseNet121 model
    base_model = model.get_layer("densenet121")

    # Create a model that returns DenseNet feature maps
    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("relu").output
    )

    with tf.GradientTape() as tape:

        # Extract convolutional feature maps
        conv_outputs = feature_extractor(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        # Pass feature maps through the classification head
        x = conv_outputs

        for layer in model.layers:
            if layer.name == "densenet121":
                continue

            if isinstance(layer, keras.layers.InputLayer):
                continue

            x = layer(x, training=False) if isinstance(
                layer,
                (keras.layers.Dropout, keras.layers.BatchNormalization)
            ) else layer(x)

        predictions = x

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + keras.backend.epsilon()
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )

    return tf.squeeze(heatmap).numpy()



def visualize_gradcam(
    original_image,
    heatmap,
    disease,
    probability,
    alpha=0.4
):
    """
    Display original X-ray and Grad-CAM overlay.
    """

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        original_image,
        cmap="gray"
    )

    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(
        original_image,
        cmap="gray"
    )

    axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=alpha
    )

    axes[1].set_title(
        f"{disease} | Probability: {probability:.3f}"
    )

    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
