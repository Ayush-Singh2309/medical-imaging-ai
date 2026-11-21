import keras
from keras import layers

from medical_imaging_ai.constants import CLASSES, IMAGE_SIZE


def build_model():
    """
    Build a multi-label chest X-ray classifier
    using pretrained DenseNet121.
    """

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="image"
    )

    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        len(CLASSES),
        activation="sigmoid",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="chest_xray_classifier"
    )

    return model