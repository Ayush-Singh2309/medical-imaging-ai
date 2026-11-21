
import keras

from medical_imaging_ai.paths import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_callbacks():

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

    ]
