from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

# Change this to an existing NIH JPG image
IMAGE_PATH = Path("data/raw/NIH_ChestXray14/images/00000001_000.png")
OUTPUT_PATH = Path("data/processed/sample_chest_xray.dcm")

# Read the JPG as an 8-bit grayscale image
image = Image.open(IMAGE_PATH).convert("L")
pixel_array = np.asarray(image, dtype=np.uint8)

# DICOM file metadata
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()

# Create DICOM dataset
ds = FileDataset(
    str(OUTPUT_PATH),
    {},
    file_meta=file_meta,
    preamble=b"\0" * 128,
)

ds.SOPClassUID = SecondaryCaptureImageStorage
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.Modality = "OT"
ds.PatientName = "SYNTHETIC"
ds.PatientID = "DEMO001"
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()

ds.Rows, ds.Columns = pixel_array.shape
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = 8
ds.BitsStored = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
ds.PixelData = pixel_array.tobytes()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ds.save_as(OUTPUT_PATH, enforce_file_format=True)

print(f"Created synthetic DICOM: {OUTPUT_PATH}")