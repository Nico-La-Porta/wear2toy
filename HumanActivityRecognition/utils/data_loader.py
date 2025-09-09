# === data_loader.py ===
import os
import pandas as pd
import glob
from utils.log_config import logger
from utils import mapping_activity

def load_and_preprocess_data_unified(toy_name, data_path, target_actions=None):
    logger.info(f"--- LOADING (UNIFIED) for '{toy_name}' ---")

    toy_configs = {
        'ball':     {'prefix': 'BA', 'mapping': mapping_activity.BALL_ACTION_MAPPING, 'classes_to_remove': []},
        'car':      {'prefix': 'C',  'mapping': mapping_activity.CAR_ACTION_MAPPING,  'classes_to_remove': [2,3,5,9,11,12,14,16,18,19,27,28,29,31,32,37,38,39,40]},
        'doll':     {'prefix': 'DO', 'mapping': mapping_activity.DOLL_ACTION_MAPPING, 'classes_to_remove': [3,4,7,9,12,19,27,31]},
        'spoon':    {'prefix': 'SP', 'mapping': mapping_activity.SPOON_ACTION_MAPPING, 'classes_to_remove': [4]},
        'elephant': {'prefixes': ["BE","GE","RE","YE","WE"], 'mapping': mapping_activity.ELEPHANT_ACTION_MAPPING, 'classes_to_remove': [7,20,5,2,19,29,31]},
    }
    cfg = toy_configs[toy_name]

    files = []
    if 'prefixes' in cfg:
        for p in cfg['prefixes']:
            files.extend(glob.glob(os.path.join(data_path, f"*_{p}*.csv")))
    else:
        files = glob.glob(os.path.join(data_path, f"*_{cfg['prefix']}*.csv"))

    files = sorted(files)  # determinismo
    if not files:
        raise FileNotFoundError(f"Nessun CSV per '{toy_name}' in '{data_path}'")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # colonne chiave e tipi robusti
        df['original_file']  = os.path.basename(f)
        df['original_row_id'] = df.index

        # tipi numerici coerenti per action_id / kid_id
        df['action_id'] = pd.to_numeric(df['action_id'], errors='coerce').fillna(0).astype(int)
        df['kid_id']    = pd.to_numeric(df['kid_id'],    errors='coerce').fillna(-1).astype(int)

        dfs.append(df)

    df_toy = pd.concat(dfs, ignore_index=True)

    # 1) rimuovi action==0
    before = len(df_toy)
    df_toy = df_toy[df_toy['action_id'] != 0]
    logger.info(f"Filtro action_id!=0: rimosse {before - len(df_toy)} righe.")

    # 2) rimuovi classi specifiche
    if cfg['classes_to_remove']:
        before = len(df_toy)
        df_toy = df_toy[~df_toy['action_id'].isin(cfg['classes_to_remove'])]
        logger.info(f"Filtro classi_to_remove: rimosse {before - len(df_toy)} righe.")

    # 3) opzionale: target actions
    if target_actions is not None:
        before = len(df_toy)
        logger.info(f"Applico target_actions: {target_actions}")
        df_toy = df_toy[df_toy['action_id'].isin(target_actions)]
        logger.info(f"Filtro target_actions: rimosse {before - len(df_toy)} righe.")

    logger.info(f"Shape finale: {df_toy.shape}, azioni: {sorted(df_toy['action_id'].unique().tolist())}")
    return df_toy, cfg['mapping']

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