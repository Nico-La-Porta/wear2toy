import numpy as np
import os
from typing import Tuple,List, Dict, Any
import sys
from app_config import PROJ_ROOT
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))

from utils.log_config import logger
import pandas as pd

def concatenate_recordings(dataset_labeled):
    # Verifico che il dataset non sia vuoto
    if not dataset_labeled:
        return None
    
    all_data = []
    
    # Itero su ogni traccia nel dataset
    for trace in dataset_labeled:
        # Aggiungo i dati di questa traccia alla lista all_data
        all_data.append(trace['TraceData'])
    
    # Concateno tutti i dati lungo l'asse 0 (righe)
    concatenated_data = np.vstack(all_data)
    
    return concatenated_data


def compute_global_median_iqr(concatenated_data: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    features = concatenated_data[:, :-1]  # escludo l'ultima colonna (activity)
    logger.debug(f"Shape of features: {features.shape}")
    medians = np.median(features, axis=0)
    logger.debug(f"Shape of medians: {medians.shape}")
    q75 = np.percentile(features, 75, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1e-6  # per evitare divisioni per zero
    logger.debug(f"Shape of IQR: {iqr.shape}")
    return medians, iqr



def compute_mean_std(trs_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la media e la deviazione standard per ogni colonna (feature) del TRS.
    L'ultima colonna (activity) viene esclusa.
    """
    features = trs_data[:, :-1]
    logger.debug(f"Shape of TRS features: {features.shape}")
    
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1e-6  # evita divisione per zero
    
    logger.debug(f"Means shape: {means.shape}, Stds shape: {stds.shape}")
    return means, stds


def compute_dataframe_mean_std(df: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la media e la deviazione standard per ogni colonna dei sensori del DataFrame.
    Le colonne non-sensore (come activity, kid_id, etc.) vengono escluse.
    
    Args:
        df: DataFrame pandas con i dati dei sensori
        
    Returns:
        Tuple contenente (means, stds) per le colonne dei sensori
    """
    
    # Identifico le colonne delle feature (sensori)
    sensor_columns = [col for col in df.columns if col in [
        'Accel_LN_X', 'Accel_LN_Y', 'Accel_LN_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'
    ]]
    
    logger.debug(f"Colonne sensori trovate: {sensor_columns}")
    
    if not sensor_columns:
        logger.warning("Nessuna colonna sensore trovata nel DataFrame")
        return np.array([]), np.array([])
    
    # Estraggo solo le colonne dei sensori
    sensor_data = df[sensor_columns].values
    logger.debug(f"Shape of DataFrame sensor features: {sensor_data.shape}")
    
    # Calcolo media e deviazione standard
    means = np.mean(sensor_data, axis=0)
    stds = np.std(sensor_data, axis=0)
    stds[stds == 0] = 1e-6  # evita divisione per zero
    
    logger.debug(f"Means shape: {means.shape}, Stds shape: {stds.shape}")
    logger.info(f"Calcolate statistiche per {len(sensor_columns)} colonne sensori")
    
    return means, stds


def compute_dataframe_median_iqr(df: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la mediana e l'IQR per ogni colonna dei sensori del DataFrame.
    Le colonne non-sensore (come activity, kid_id, etc.) vengono escluse.
    
    Args:
        df: DataFrame pandas con i dati dei sensori
        
    Returns:
        Tuple contenente (medians, iqr) per le colonne dei sensori
    """
    import pandas as pd
    
    # Identifico le colonne delle feature (sensori)
    sensor_columns = [col for col in df.columns if col in [
        'Accel_LN_X', 'Accel_LN_Y', 'Accel_LN_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'
    ]]
    
    logger.debug(f"Colonne sensori trovate: {sensor_columns}")
    
    if not sensor_columns:
        logger.warning("Nessuna colonna sensore trovata nel DataFrame")
        return np.array([]), np.array([])
    
    # Estraggo solo le colonne dei sensori
    sensor_data = df[sensor_columns].values
    logger.debug(f"Shape of DataFrame sensor features: {sensor_data.shape}")
    
    # Calcolo mediana e IQR
    medians = np.median(sensor_data, axis=0)
    q75 = np.percentile(sensor_data, 75, axis=0)
    q25 = np.percentile(sensor_data, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1e-6  # per evitare divisioni per zero
    
    logger.debug(f"Medians shape: {medians.shape}, IQR shape: {iqr.shape}")
    logger.info(f"Calcolate statistiche robuste per {len(sensor_columns)} colonne sensori")
    
    return medians, iqr






def normalize_each_recording_med_iqr(dataset_labeled: List[Dict[str, Any]], medians: np.ndarray, iqr: np.ndarray) -> List[Dict[str, Any]]:
    normalized_dataset = []

    for trace in dataset_labeled:
        data = trace['TraceData']
        features = data[:, :-1]  # dati senza activity
        activity_col = data[:, -1].reshape(-1, 1)  # solo la colonna activity per poi ricombinarla 

        # Normalizzazione robusta
        normalized_features = (features - medians) / iqr # sottraggo la mediana globale dalla colonna corrispondente
        logger.debug(f"Shape of normalized features: {normalized_features.shape}")

        # Ricombino con activity
        normalized_data = np.hstack((normalized_features, activity_col)) ## ricombino i dati normalizzati con la colonna activity
        logger.debug(f"Shape of normalized data: {normalized_data.shape}")
        new_trace = trace.copy() # creo una copia della traccia originale
        new_trace['TraceData'] = normalized_data # sostituisco i dati originali con quelli normalizzati
        normalized_dataset.append(new_trace)

    return normalized_dataset


def normalize_each_recording_mean_std(dataset_labeled: List[Dict[str, Any]], 
                              means: np.ndarray, 
                              stds: np.ndarray) -> List[Dict[str, Any]]:
    """
    Normalizza ogni traccia (TraceData) usando le statistiche (media e std) calcolate sul TRS.
    """
    normalized_dataset = []
    for trace in dataset_labeled:
        data = trace['TraceData']
        features = data[:, :-1]
        activity_col = data[:, -1].reshape(-1, 1)

        normalized_features = (features - means) / stds
        logger.debug(f"Normalized features shape: {normalized_features.shape}")

        normalized_data = np.hstack((normalized_features, activity_col))
        new_trace = trace.copy()
        new_trace['TraceData'] = normalized_data
        normalized_dataset.append(new_trace)

    return normalized_dataset



def normalize_dataframe_mean_std(df: 'pd.DataFrame', means: np.ndarray, stds: np.ndarray) -> 'pd.DataFrame':
    """
    Normalizza un DataFrame usando le statistiche (media e std) calcolate sul TRS.
    
    Args:
        df: DataFrame pandas con i dati dei sensori
        means: Array delle medie per la normalizzazione
        stds: Array delle deviazioni standard per la normalizzazione
    
    Returns:
        DataFrame normalizzato
    """
    import pandas as pd
    
    # Identifico le colonne delle feature (sensori)
    sensor_columns = [col for col in df.columns if col in [
        'Accel_LN_X', 'Accel_LN_Y', 'Accel_LN_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'
    ]]
    
    # Verifico che abbiamo il numero corretto di feature
    if len(sensor_columns) != len(means):
        logger.warning(f"Numero di colonne sensori ({len(sensor_columns)}) diverso da numero di medie ({len(means)})")
        # PrendO solo le prime len(means) colonne
        sensor_columns = sensor_columns[:len(means)]
    
    # CopiO il DataFrame per non modificare l'originale
    df_normalized = df.copy()
    
    # NormalizzO solo le colonne dei sensori
    for i, col in enumerate(sensor_columns):
        if i < len(means):
            df_normalized[col] = (df[col] - means[i]) / stds[i]
            logger.debug(f"Normalizzata colonna {col} usando mean={means[i]:.6f}, std={stds[i]:.6f}")
    
    logger.info(f"DataFrame normalizzato - Shape: {df_normalized.shape}")
    return df_normalized


def normalize_dataframe_median_iqr(df: 'pd.DataFrame', medians: np.ndarray, iqr: np.ndarray) -> 'pd.DataFrame':
    """
    Normalizza un DataFrame usando le statistiche (mediana e IQR) calcolate sul TRS.
    
    Args:
        df: DataFrame pandas con i dati dei sensori
        medians: Array delle mediane per la normalizzazione
        iqr: Array degli IQR per la normalizzazione
    
    Returns:
        DataFrame normalizzato
    """
    import pandas as pd
    
    # Identifico le colonne delle feature (sensori)
    sensor_columns = [col for col in df.columns if col in [
        'Accel_LN_X', 'Accel_LN_Y', 'Accel_LN_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'
    ]]
    
    # Verifico che abbiamo il numero corretto di feature
    if len(sensor_columns) != len(medians):
        logger.warning(f"Numero di colonne sensori ({len(sensor_columns)}) diverso da numero di mediane ({len(medians)})")
        # Prendo solo le prime len(medians) colonne
        sensor_columns = sensor_columns[:len(medians)]
    
    # Copio il DataFrame per non modificare l'originale
    df_normalized = df.copy()
    
    # Normalizzo solo le colonne dei sensori
    for i, col in enumerate(sensor_columns):
        if i < len(medians):
            df_normalized[col] = (df[col] - medians[i]) / iqr[i]
            logger.debug(f"Normalizzata colonna {col} usando median={medians[i]:.6f}, iqr={iqr[i]:.6f}")
    
    logger.info(f"DataFrame normalizzato (median-IQR) - Shape: {df_normalized.shape}")
    return df_normalized

# Esempio di testing
if __name__ == "__main__":
    # Dati di esempio
    dataset_labeled = [
        {'TraceData': np.array([[0.1, 0.2, 0.3, 1],
                               [0.2, 0.3, 0.4, 2],
                               [0.3, 0.4, 0.5, 1]])},  # prima traccia
        {'TraceData': np.array([[0.5, 0.6, 0.7, 2],
                               [0.6, 0.7, 0.8, 1],
                               [0.7, 0.8, 0.9, 2]])}   # seconda traccia 
    ]
    
    # Concateno i dati
    concatenated_data = concatenate_recordings(dataset_labeled)
    logger.info(f"Concatenated Data Shape: {concatenated_data.shape}")
    
    # Calcolo la mediana e l'IQR globali
    medians, iqr = compute_global_median_iqr(concatenated_data)
    logger.info(f"Medians: {medians}")
    logger.info(f"IQR: {iqr}")
    
    # Normalizzo i dati
    normalized_dataset = normalize_each_recording_med_iqr(dataset_labeled, medians, iqr)
    logger.info(f"Normalized Data: {normalized_dataset[0]['TraceData']}")

    