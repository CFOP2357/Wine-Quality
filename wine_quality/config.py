from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ===================== CONFIGURACIÓN DEL EXPERIMENTO =====================
# Datos crudos (UCI Wine Quality; separador ';')
RED_PATH = RAW_DATA_DIR / "winequality-red.csv"
WHITE_PATH = RAW_DATA_DIR / "winequality-white.csv"

TARGET = "quality"

# Partición 70 % entrenamiento / 15 % validación / 15 % prueba
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Semilla por defecto (sobreescribible con --seed en el script de entrenamiento)
DEFAULT_SEED = 42

# MLflow (tracking local; lo consumen modeling/train.py y modeling/tune.py)
EXPERIMENT_NAME = "TC5061_Actividad3_Experimentacion"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
