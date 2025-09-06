import pathlib
import logging
import logging.config
import json
import os
import sys

def setup_logging(exp_name: str):
    """
    Configura il logging per salvare i log in un file specifico per l'esperimento.

    Args:
        exp_name (str): Il nome univoco dell'esperimento, usato per il nome del file di log.
    """
    LOGGING_CONFIG = os.path.join(pathlib.Path(__file__).parent.resolve(), 'base_config.json')

    with open(LOGGING_CONFIG, 'r') as fd:
        log_config = json.load(fd)
    
    # Crea la cartella dei log se non esiste
    log_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # --- MODIFICA CHIAVE: Usa exp_name per il nome del file ---
    log_file_path = os.path.join(log_dir, f"{exp_name}.log")
    error_log_file_path = os.path.join(log_dir, f"{exp_name}_error.log")

    # Aggiorna il dizionario di configurazione con i nuovi percorsi dei file
    if 'file' in log_config['handlers']:
        log_config['handlers']['file']['filename'] = log_file_path
    
    if 'error_file' in log_config['handlers']:
        log_config['handlers']['error_file']['filename'] = error_log_file_path

    # Applica la configurazione
    logging.config.dictConfig(log_config)
    
    # Ottieni e restituisci il logger configurato
    logger = logging.getLogger('myapp')
    logger.info(f"Il logging è stato configurato. I log verranno salvati in: {log_file_path}")
    return logger

# Manteniamo un logger di base per essere importato da altri moduli.
# Sarà riconfigurato correttamente non appena setup_logging viene chiamato.
logger = logging.getLogger('myapp')