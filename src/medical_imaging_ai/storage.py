
import base64
from pathlib import Path

from medical_imaging_ai.paths import PROJECT_ROOT


STUDIES_DIR = (
    PROJECT_ROOT / "data" / "processed" / "studies"
)


def save_study_images(
    study_id: int,
    original_image_bytes: bytes,
    gradcam_base64: str,
    original_filename: str
):
    """
    Save the original uploaded image and Grad-CAM overlay
    under the corresponding study ID.
    """

    study_dir = STUDIES_DIR / str(study_id)

    study_dir.mkdir(parents=True, exist_ok=True)

    # Preserve a supported original image extension.
    extension = Path(original_filename).suffix.lower()

    if extension not in {".png", ".jpg", ".jpeg"}:
        extension = ".png"

    original_path = study_dir / f"original{extension}"
    gradcam_path = study_dir / "gradcam.png"

    # Decode the Grad-CAM Base64 string.
    gradcam_bytes = base64.b64decode(
        gradcam_base64,
        validate=True
    )

    # Save both images.
    original_path.write_bytes(original_image_bytes)
    gradcam_path.write_bytes(gradcam_bytes)

    return original_path, gradcam_path
