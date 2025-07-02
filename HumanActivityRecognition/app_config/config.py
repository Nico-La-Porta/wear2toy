import os
import dotenv
from pathlib import Path

# Set a default environment if none is provided
APP_ENV = os.getenv('APP_ENV')

if APP_ENV == 'hpc_supsi':
    from .config_hpc_supsi import *
elif APP_ENV == 'testing':
    from .config_test import *
elif APP_ENV == 'dev':
    from .config_dev import *
else:
    dotenv.load_dotenv()

    # Configuration - Load from environment variables
    NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
    NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

    # Paths
    PROJ_ROOT = Path(__file__).resolve().parents[2]

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