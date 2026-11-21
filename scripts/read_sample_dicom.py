from pathlib import Path

import pydicom
import matplotlib.pyplot as plt

DICOM_PATH = Path("data/processed/retrieved_sample.dcm")

# Read DICOM
ds = pydicom.dcmread(DICOM_PATH)

# Inspect metadata
print("Patient ID:", ds.PatientID)
print("Modality:", ds.Modality)
print("Study UID:", ds.StudyInstanceUID)
print("Rows:", ds.Rows)
print("Columns:", ds.Columns)
print("Photometric Interpretation:", ds.PhotometricInterpretation)

# Extract image pixels
image = ds.pixel_array

print("Pixel shape:", image.shape)
print("Pixel dtype:", image.dtype)
print("Pixel minimum:", image.min())
print("Pixel maximum:", image.max())

# Display image
plt.imshow(image, cmap="gray")
plt.title("DICOM Chest X-ray")
plt.axis("off")
plt.show()