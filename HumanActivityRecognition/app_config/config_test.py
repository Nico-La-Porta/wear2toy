DEBUG = True

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" #in formato .csv
RAW_DATA_DIR_TRAIN = RAW_DATA_DIR/"train" #in formato .csv
RAW_DATA_DIR_TEST=RAW_DATA_DIR/"test"  #in formato .csv
FEATURES_DATA_DIR = DATA_DIR / "features" #le due matrici di input con feature extracted di ML
PROCESSED_DATA_DIR = DATA_DIR / "processed" #in formato .npz
PROCESSED_DATA_DIR_TRAIN= PROCESSED_DATA_DIR/"train" #in formato .npz
PROCESSED_DATA_DIR_TEST= PROCESSED_DATA_DIR/"test" #in formato .npz


MODELS_DIR = PROJ_ROOT / "models" #cartella per i modelli salvati

REPORTS_DIR = PROJ_ROOT / "reports" # Report generati
FIGURES_DIR = REPORTS_DIR / "figures" # Grafici inclusi nei report

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
