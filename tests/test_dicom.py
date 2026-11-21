from pathlib import Path

import numpy as np
import pydicom

from medical_imaging_ai.api.main import preprocess_dicom_image
from medical_imaging_ai.constants import IMAGE_SIZE


DICOM_PATH = Path("data/processed/sample_chest_xray.dcm")


def test_dicom_file_can_be_read():
    assert DICOM_PATH.exists()

    dicom = pydicom.dcmread(DICOM_PATH)

    assert dicom.Modality == "OT"
    assert dicom.PatientID == "DEMO001"


def test_dicom_pixel_data():
    dicom = pydicom.dcmread(DICOM_PATH)

    image = dicom.pixel_array

    assert isinstance(image, np.ndarray)
    assert image.ndim == 2
    assert image.dtype == np.uint8


def test_dicom_preprocessing():
    dicom = pydicom.dcmread(DICOM_PATH)

    image = preprocess_dicom_image(dicom)

    expected_shape = (1, *IMAGE_SIZE, 3)

    assert tuple(image.shape) == expected_shape