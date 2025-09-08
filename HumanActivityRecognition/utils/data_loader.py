# === data_loader.py ===
import os
import pandas as pd
import glob
from utils.log_config import logger
from utils import mapping_activity

def load_and_preprocess_data_unified(toy_name, data_path, target_actions=None):
    """
    Funzione UNIFICATA e ROBUSTA per caricare e pre-processare i dati per un giocattolo.
    Questa funzione deve essere usata da TUTTI gli script (zupt_analysis, run_har_experiment)
    per garantire la coerenza al 100%.

    Args:
        toy_name (str): Nome del giocattolo (es. 'car').
        data_path (str): Percorso alla cartella dei dati.
        target_actions (list, optional): Se fornita, il DataFrame conterrà SOLO
                                         le azioni in questa lista. Defaults to None.

    Returns:
        tuple: (DataFrame, dict) -> (df_toy, toy_mapping)
    """
    logger.info(f"--- ESECUZIONE FUNZIONE DI CARICAMENTO UNIFICATA per '{toy_name}' ---")
    
    toy_configs = {
        'ball': {'prefix': 'BA', 'mapping': mapping_activity.BALL_ACTION_MAPPING, 'classes_to_remove': []},
        'car': {'prefix': 'C', 'mapping': mapping_activity.CAR_ACTION_MAPPING, 'classes_to_remove': [2, 3, 5, 9, 11, 12, 14, 16, 18, 19, 27, 28, 29, 31, 32, 37, 38, 39, 40]},
        'doll': {'prefix': 'DO', 'mapping': mapping_activity.DOLL_ACTION_MAPPING, 'classes_to_remove': [3, 4, 7, 9, 12, 19, 27, 31]},
        'spoon': {'prefix': 'SP', 'mapping': mapping_activity.SPOON_ACTION_MAPPING, 'classes_to_remove': [4]},
        'elephant': {'prefixes': ["BE", "GE", "RE", "YE", "WE"], 'mapping': mapping_activity.ELEPHANT_ACTION_MAPPING, 'classes_to_remove': [7, 20, 5, 2, 19, 29, 31]},
    }
    
    config = toy_configs[toy_name]
    
    files = []
    if 'prefixes' in config:
        for p in config['prefixes']:
            files.extend(glob.glob(os.path.join(data_path, f"*_{p}*.csv")))
    else:
        files = glob.glob(os.path.join(data_path, f"*_{config['prefix']}*.csv"))
    
    # GARANZIA DI ORDINE: Ordina sempre la lista dei file!
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"Nessun file CSV trovato per il giocattolo '{toy_name}' nel percorso '{data_path}'")
        
    logger.info(f"Trovati {len(files)} file. Primo file: {os.path.basename(files[0])}")

    df_list = []
    for file in files:
        df_single_file = pd.read_csv(file)
        # Aggiungo metadati tracciabili
        df_single_file['original_file'] = os.path.basename(file)
        df_single_file['original_row_id'] = df_single_file.index
        df_list.append(df_single_file)
        
    df_toy = pd.concat(df_list, ignore_index=True)
    
    # 1. Filtro Azioni Nulle
    rows_before = len(df_toy)
    df_toy = df_toy[df_toy['action_id'] != 0]
    logger.info(f"Filtro action_id != 0: rimosse {rows_before - len(df_toy)} righe.")

    # 2. Rimuovi Classi Specifiche del Giocattolo
    if config['classes_to_remove']:
        rows_before = len(df_toy)
        df_toy = df_toy[~df_toy['action_id'].isin(config['classes_to_remove'])]
        logger.info(f"Filtro classes_to_remove: rimosse {rows_before - len(df_toy)} righe.")
    
    # 3. (OPZIONALE) Filtro su Azioni Target Specifiche
    if target_actions is not None:
        rows_before = len(df_toy)
        logger.info(f"Applico filtro su target_actions: {target_actions}")
        df_toy = df_toy[df_toy['action_id'].isin(target_actions)]
        logger.info(f"Filtro target_actions: rimosse {rows_before - len(df_toy)} righe.")

    logger.info(f"Caricamento completato. Shape finale del DataFrame: {df_toy.shape}")
    logger.info(f"Azioni uniche presenti: {sorted(df_toy['action_id'].unique().tolist())}")
        
    return df_toy, config['mapping']

def add_consecutive_segment_id(df):
    """
    Aggiunge una colonna 'consecutive_segment_id'.
    Questa funzione è identica in entrambi gli script, la spostiamo qui per coerenza.
    """
    logger.info("Identificazione dei segmenti di attività consecutivi...")
    
    df = df.sort_values(by=['original_file', 'original_row_id']).reset_index(drop=True)

    is_new_segment = (
        (df['original_file'] != df['original_file'].shift(1)) |
        (df['action_id'] != df['action_id'].shift(1)) |
        (df['original_row_id'] != df['original_row_id'].shift(1) + 1)
    )
    
    df['consecutive_segment_id'] = is_new_segment.cumsum()
    
    logger.info(f"Trovati {df['consecutive_segment_id'].nunique()} segmenti unici.")
    return df