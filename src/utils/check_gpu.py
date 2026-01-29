import torch
from utils.log_config import logger
#verifico se la GPU è disponibile
def check_gpu_availability():

    if torch.cuda.is_available():
        logger.info('GPU available, training on GPU.')
        return True
    else:
        logger.info('GPU not available, training on CPU.')
        return False

