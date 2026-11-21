from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "NIH_ChestXray14"
IMAGE_DIR = RAW_DATA_DIR / "images"
METADATA_PATH = RAW_DATA_DIR / "Data_Entry_2017.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"