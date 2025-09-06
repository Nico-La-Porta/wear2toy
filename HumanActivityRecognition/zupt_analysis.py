"""
Script di preprocessing per filtraggio ZUPT e downsampling
Questo script deve essere eseguito PRIMA di run_har_experiment.py

Funzionalità:
1. Analisi ZUPT per giocattolo e azione
2. Filtraggio finestre con pause
3. Downsampling selettivo (max 50 finestre per bambino/azione)
4. Salvataggio indici finestre da mantenere
5. Distribuzione prima/dopo per verifica
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import pickle
from collections import defaultdict, Counter

# Aggiungi il path del progetto
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'HumanActivityRecognition')))
sys.path.append(os.path.abspath('HumanActivityRecognition'))

# Import custom modules
import sliding_window_on_data
import normalization
from utils.zupt import zupt_detect
from utils.log_config import setup_logging, logger
from utils import mapping_activity

def analyze_zupt_by_toy_and_action(data_path, toy_name, target_actions=None, plot_results=True, save_plots=True, figures_dir=None):
    """
    Analizza ZUPT per ogni giocattolo e azione
    Solo per le azioni target specificate.
    """
    
    logger.info(f"=== ANALISI ZUPT COMPLETA PER {toy_name.upper()} ===")
    
    # Configura i pattern dei file per ogni giocattolo
    toy_configs = {
        'ball': {'suffixes': ['BA']},
        'car': {'suffixes': ['C']}, 
        'doll': {'suffixes': ['DO']},
        'spoon': {'suffixes': ['SP']},
        'elephant': {'suffixes': ["BE", "GE", "RE", "YE", "WE"]},
    }
    
    if toy_name not in toy_configs:
        raise ValueError(f"Giocattolo {toy_name} non supportato")
    
    config = toy_configs[toy_name]
    
    # Trova i file del giocattolo
    toy_files = []
    for suffix in config['suffixes']:
        toy_files.extend([
            f for f in os.listdir(data_path)
            if f.endswith(".csv") and f"_{suffix}" in f and not f.startswith("df_")
        ])
    
    toy_files = list(set(toy_files))  # Rimuovi duplicati
    logger.info(f"File trovati per {toy_name}: {len(toy_files)}")
    
    # Crea directory per i plot se necessario
    if save_plots and figures_dir:
        zupt_dir = os.path.join(figures_dir, f"zupt_analysis_{toy_name}")
        os.makedirs(zupt_dir, exist_ok=True)
    else:
        zupt_dir = None
    
    complete_analysis = {
        'toy_name': toy_name,
        'files_analyzed': {},
        'summary': {}
    }
    
    # Processa ogni file
    for file_name in toy_files:
        logger.info(f"\n--- Processando file: {file_name} ---")
        file_path = os.path.join(data_path, file_name)
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"File caricato: {len(df)} righe")
        except Exception as e:
            logger.error(f"Errore nel caricamento di {file_name}: {e}")
            continue
        
        # Verifica colonne necessarie
        required_cols = ['Accel_WR_X', 'Accel_WR_Y', 'Accel_WR_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'action']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Colonne mancanti in {file_name}, saltando...")
            continue
        
        file_analysis = {
            'file_name': file_name,
            'total_rows': len(df),
            'actions_found': [],
            'actions_analysis': {}
        }
        
        # Trova tutte le azioni uniche nel file

        unique_action_ids = df['action_id'].dropna().unique()
        unique_action_ids = [int(aid) for aid in unique_action_ids if not pd.isna(aid)]
        
        # Filtra solo le azioni target se specificate
        if target_actions is not None:
            unique_action_ids = [action for action in unique_action_ids if action in target_actions]

        logger.info(f"Azioni da processare in {file_name}: {list(unique_action_ids)}")

        # Processa ogni azione
        for action_id in unique_action_ids:
            logger.info(f"\n  Azione: {action_id}")
            
            # Filtra i dati per questa azione 
            df_action = df[df['action_id'] == action_id]

            if df_action.empty:
                logger.info(f"    Nessun dato per azione '{action_id}'")
                continue
            action_name = df_action['action'].iloc[0] if 'action' in df_action.columns else f"action_{action_id}"

            # Trova i blocchi consecutivi 
            indices = df_action.index.to_list()
            start = indices[0]
            prev = indices[0]
            
            action_blocks = []
            logger.info(f"    Azione '{action_name}' trovata in {file_name} nei blocchi:")
            
            for idx in indices[1:]:
                if idx != prev + 1:  # Fine di un blocco
                    block_info = {
                        'start_row_excel': start + 2,  # +2 per numerazione Excel
                        'end_row_excel': prev + 2,
                        'start_row_df': start,
                        'end_row_df': prev,
                        'length': prev - start + 1
                    }
                    action_blocks.append(block_info)
                    logger.info(f"      righe {start + 2} → {prev + 2} (lunghezza: {prev - start + 1})")
                    start = idx
                prev = idx
            
            # Ultimo blocco
            block_info = {
                'start_row_excel': start + 2,
                'end_row_excel': prev + 2, 
                'start_row_df': start,
                'end_row_df': prev,
                'length': prev - start + 1
            }
            action_blocks.append(block_info)
            logger.info(f"      righe {start + 2} → {prev + 2} (lunghezza: {prev - start + 1})")
            
            # Estrai dati sensori per questa azione
            accel_action = df_action[['Accel_WR_X', 'Accel_WR_Y', 'Accel_WR_Z']].to_numpy()
            gyro_action = df_action[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].to_numpy()
            
            # Applica ZUPT detection 
            zupt = zupt_detect(
                accel_action, gyro_action,
                acc_thresh=1, gyro_thresh=0.25,
                window_size=10, min_segment_len_seconds=0.1
            )
            
            # Trova segmenti di pausa 
            pause_segments = []
            zupt_indices = np.where(zupt)[0]  # indici locali in accel_action/gyro_action
            
            if len(zupt_indices) > 0:
                start_idx_action = df_action.index[0]  # Primo indice della azione nel DataFrame originale
                
                start = zupt_indices[0]
                prev = zupt_indices[0]
                
                logger.info(f"    Segmenti di pausa rilevati per '{action_name}' (righe nel file Excel):")
                
                for idx in zupt_indices[1:]:
                    if idx != prev + 1:  # Fine di un segmento di pausa
                        pause_info = {
                            'start_row_excel': start_idx_action + start + 2,  # +2 per numerazione Excel
                            'end_row_excel': start_idx_action + prev + 2,
                            'start_row_df': start_idx_action + start,
                            'end_row_df': start_idx_action + prev,
                            'length': prev - start + 1
                        }
                        pause_segments.append(pause_info)
                        logger.info(f"      da riga {pause_info['start_row_excel']} a riga {pause_info['end_row_excel']}")
                        start = idx
                    prev = idx
                
                # Ultimo segmento
                pause_info = {
                    'start_row_excel': start_idx_action + start + 2,
                    'end_row_excel': start_idx_action + prev + 2,
                    'start_row_df': start_idx_action + start,
                    'end_row_df': start_idx_action + prev,
                    'length': prev - start + 1
                }
                pause_segments.append(pause_info)
                logger.info(f"      da riga {pause_info['start_row_excel']} a riga {pause_info['end_row_excel']}")
            else:
                logger.info(f"    Nessuna pausa rilevata per '{action_name}'")
            
            # Plotta se richiesto 
            if plot_results:
                plot_action_with_zupt(
                    accel_action, gyro_action, zupt,
                    action_name, file_name, 
                    start_idx=df_action.index[0],
                    save_plots=save_plots,
                    save_dir=zupt_dir
                )
            
            # Salva analisi per questa azione
            action_analysis = {
                'action_name': action_name,
                'total_samples': len(df_action),
                'action_blocks': action_blocks,
                'pause_segments': pause_segments,
                'total_pause_samples': len(zupt_indices),
                'pause_percentage': len(zupt_indices) / len(df_action) * 100 if len(df_action) > 0 else 0
            }
            
            file_analysis['actions_analysis'][action_id] = action_analysis
            file_analysis['actions_found'].append(action_id)
            
            logger.info(f"    Totale campioni pausa: {len(zupt_indices)}/{len(df_action)} ({action_analysis['pause_percentage']:.1f}%)")
        
        complete_analysis['files_analyzed'][file_name] = file_analysis
    
    # Calcola statistiche di riepilogo
    total_files = len(complete_analysis['files_analyzed'])
    total_actions = sum(len(fa['actions_found']) for fa in complete_analysis['files_analyzed'].values())
    
    complete_analysis['summary'] = {
        'total_files_analyzed': total_files,
        'total_actions_found': total_actions,
        'files_with_pauses': len([f for f in complete_analysis['files_analyzed'].values() 
                                 if any(len(a['pause_segments']) > 0 for a in f['actions_analysis'].values())])
    }
    
    logger.info(f"\n=== RIEPILOGO {toy_name.upper()} ===")
    logger.info(f"File analizzati: {total_files}")
    logger.info(f"Azioni totali trovate: {total_actions}")
    logger.info(f"File con pause: {complete_analysis['summary']['files_with_pauses']}")
    
    return complete_analysis


def plot_action_with_zupt(accel, gyro, zupt, action_name, file_name, start_idx=0, 
                         save_plots=True, save_dir=None):
    """
    Plotta un'azione con ZUPT 
    """
    
    def plot_signals(accel, gyro, fs=100, zupt=None, pause_idx=None, start_times_pp_sam=None, start_idx=0):
        t_axis = np.arange(len(accel))/fs
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot accelerometro
        axs[0].plot(t_axis, accel[:, 0], label='Acc_X', color='r', alpha=0.5, linewidth=0.3)
        axs[0].plot(t_axis, accel[:, 1], label='Acc_Y', color='g', alpha=0.5, linewidth=0.3)
        axs[0].plot(t_axis, accel[:, 2], label='Acc_Z', color='b', alpha=0.5, linewidth=0.3)
        axs[0].set_title('Accelerometer Data')
        axs[0].set_ylabel('Acceleration (g)')
        axs[0].set_xlim(t_axis[0], t_axis[-1])
        axs[0].set_ylim(-16, 16)
        axs[0].legend()

        # Plot giroscopio
        axs[1].plot(t_axis, gyro[:, 0], label='Gyr_X', color='r', alpha=0.5, linewidth=0.3)
        axs[1].plot(t_axis, gyro[:, 1], label='Gyr_Y', color='g', alpha=0.5, linewidth=0.3)
        axs[1].plot(t_axis, gyro[:, 2], label='Gyr_Z', color='b', alpha=0.5, linewidth=0.3)
        axs[1].axhline(y=513.4977679749721, color='k', linestyle='--', linewidth=5, label='Gyro Bias Y')
        axs[1].set_title('Gyroscope Data')
        axs[1].set_xlabel('Seconds')
        axs[1].set_xlim(t_axis[0], t_axis[-1])
        axs[1].set_ylim(np.percentile(gyro, 1), np.percentile(gyro, 99))
        axs[1].set_ylabel('Angular Velocity (°/s)')
        axs[1].legend()

        if zupt is not None:
            # STAMPA GLI INDICI DELLE PAUSE 
            indices = np.where(zupt)[0]
            if len(indices) > 0:
                start = indices[0]
                prev = indices[0]
                print("Segmenti di pausa rilevati (righe nel file originale):")
                for idx in indices[1:]:
                    if idx != prev + 1:
                        print(f"  da riga {start_idx + start + 2} a riga {start_idx + prev + 2}")
                        start = idx
                    prev = idx
                print(f"  da riga {start_idx + start + 2} a riga {start_idx + prev + 2}")
            
            # Evidenzia le pause
            axs[0].fill_between(t_axis, np.min(accel) - 1, np.max(accel) + 1,
                               where=zupt.astype(bool), alpha=0.4, label='ZUPT', color='red')
            axs[1].fill_between(t_axis, np.min(gyro) - 1, np.max(gyro) + 1,
                               where=zupt.astype(bool), alpha=0.4, label='ZUPT', color='red')

        plt.tight_layout()
        return fig, axs
    
    # Plotta l'azione
    fig, axs = plot_signals(accel, gyro, fs=100, zupt=zupt, start_idx=start_idx)
    
    # Aggiungi titolo
    fig.suptitle(f'ZUPT Analysis - {file_name}\nAction: {action_name}', 
                fontsize=12, fontweight='bold', y=0.98)
    
    # Salva se richiesto
    if save_plots and save_dir:
        import re
        safe_action = action_name.replace(' ', '_').replace('_', '_')
        safe_file = file_name.replace('.csv', '')
        filename = f"zupt_{safe_file}_{safe_action}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot salvato: {save_path}")
    
    plt.close(fig)  # Chiudi per evitare accumulo in memoria
    return fig, axs


def load_and_preprocess_data_for_zupt(toy_name, data_path, target_actions):
    """
    """
    logger.info(f"Caricamento per {toy_name} ")
    
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
        
    df_list = []
    for file_idx,file in enumerate(files):
        df_single_file = pd.read_csv(file)
        df_single_file['original_file'] = os.path.basename(file)
        df_single_file['source_file_id'] = file_idx
        df_single_file['original_row_id'] = df_single_file.index
        df_list.append(df_single_file)
        
    df_toy = pd.concat(df_list, ignore_index=True)
    df_toy = df_toy[df_toy['action_id'] != 0]

    if config['classes_to_remove']:
        df_toy = df_toy[~df_toy['action_id'].isin(config['classes_to_remove'])]
    
    df_toy = df_toy[df_toy['action_id'].isin(target_actions)]
        
    return df_toy, config['mapping']


def add_consecutive_segment_id(df):
    """
    Aggiunge una colonna 'consecutive_segment_id' basata su file originale,
    action_id e consecutività del 'original_row_id'.
    """
    logger.info("Identificazione dei segmenti di attività consecutivi...")
    
    # Assicura che il DataFrame sia ordinato correttamente per la logica di 'shift'
    df = df.sort_values(by=['original_file', 'original_row_id']).reset_index(drop=True)

    # Un nuovo segmento inizia se:
    # 1. Cambia il file originale
    # 2. Cambia l'action_id
    # 3. C'è un "salto" nel numero di riga originale (es. da 20 a 40)
    is_new_segment = (
        (df['original_file'] != df['original_file'].shift(1)) |
        (df['action_id'] != df['action_id'].shift(1)) |
        (df['original_row_id'] != df['original_row_id'].shift(1) + 1)
    )
    
    # Usa cumsum() per assegnare un ID univoco a ogni blocco consecutivo
    df['consecutive_segment_id'] = is_new_segment.cumsum()
    
    logger.info(f"Trovati {df['consecutive_segment_id'].nunique()} segmenti unici.")
    return df


def map_pause_rows_to_windows(pause_analysis, all_window_indices, pause_threshold=0.5):
    """
    Mappa le righe di pausa alle finestre generate dalla sliding window.
    """
    
    logger.info("=== MAPPING PAUSE -> FINESTRE ===")

    # Debug: stampa i primi metadati
    logger.info("=== DEBUG METADATI ===")
    for i in range(min(3, len(all_window_indices))):
        logger.info(f"Metadati finestra {i}: {all_window_indices[i]}")
    
    # Verifica se row_indices esistono
    has_row_indices = all('row_indices' in metadata for metadata in all_window_indices)
    logger.info(f"Tutti i metadati hanno 'row_indices': {has_row_indices}")
    
    if not has_row_indices:
        logger.error("ERRORE: I metadati non contengono 'row_indices'!")
        # Restituisci risultato vuoto
        return {
            'windows_to_remove': [],
            'windows_analysis': [],
            'summary': {'total_windows': len(all_window_indices), 'windows_removed': 0, 'removal_percentage': 0}
        }
    
    # Raccogli tutte le righe di pausa da tutti i file e azioni
    all_pause_rows_by_file = {}
    
    for file_name, file_analysis in pause_analysis['files_analyzed'].items():
        file_pause_rows = set()
        
        for action_name, action_analysis in file_analysis['actions_analysis'].items():
            for pause_segment in action_analysis['pause_segments']:
                # Aggiungi tutte le righe del segmento di pausa (usando indici DataFrame)
                for row in range(pause_segment['start_row_df'], pause_segment['end_row_df'] + 1):
                    file_pause_rows.add(row)
        
        if file_pause_rows:
            all_pause_rows_by_file[file_name] = file_pause_rows
            logger.info(f"File {file_name}: {len(file_pause_rows)} righe di pausa")
    
    mapping_results = {
    'windows_to_remove': [],
    'windows_analysis': [],
    'summary': {}
    }
    # Analizza ogni finestra
    for window_idx, metadata in enumerate(all_window_indices):
        window_file = metadata.get('original_file', 'unknown')
        window_row_indices = metadata.get('row_indices', [])
        
        if window_file not in all_pause_rows_by_file:
            # Nessuna pausa in questo file
            mapping_results['windows_analysis'].append({
                'window_index': window_idx,
                'file_name': window_file,
                'total_rows': len([r for r in window_row_indices if r != -1]),
                'pause_rows': 0,
                'pause_percentage': 0.0,
                'remove': False
            })
            continue
        
        file_pause_rows = all_pause_rows_by_file[window_file]
        
        # Conta quante righe della finestra sono pause
        pause_rows_in_window = 0
        valid_rows_in_window = 0
        
        for row_id in window_row_indices:
            if row_id != -1:  # -1 indica padding
                valid_rows_in_window += 1
                if row_id in file_pause_rows:
                    pause_rows_in_window += 1
        
        pause_percentage = pause_rows_in_window / valid_rows_in_window if valid_rows_in_window > 0 else 0
        should_remove = pause_percentage >= pause_threshold
        
        window_analysis = {
            'window_index': window_idx,
            'file_name': window_file,
            'total_rows': valid_rows_in_window,
            'pause_rows': pause_rows_in_window,
            'pause_percentage': pause_percentage,
            'remove': should_remove,
            'action_id': metadata.get('action_id', 'unknown'),
            'kid_id': metadata.get('kid_id', 'unknown')
        }
        
        mapping_results['windows_analysis'].append(window_analysis)
        
        if should_remove:
            mapping_results['windows_to_remove'].append(window_idx)
            logger.info(f"Finestra {window_idx} rimossa: {pause_percentage*100:.1f}% pause "
                       f"({pause_rows_in_window}/{valid_rows_in_window} righe) - File: {window_file}")
    
    # Statistiche finali
    total_windows = len(all_window_indices)
    windows_removed = len(mapping_results['windows_to_remove'])
    
    mapping_results['summary'] = {
        'total_windows': total_windows,
        'windows_removed': windows_removed,
        'removal_percentage': windows_removed / total_windows * 100 if total_windows > 0 else 0
    }
    
    logger.info(f"\n=== RISULTATI MAPPING ===")
    logger.info(f"Finestre totali: {total_windows}")
    logger.info(f"Finestre da rimuovere: {windows_removed}")
    logger.info(f"Percentuale rimossa: {mapping_results['summary']['removal_percentage']:.1f}%")
    
    return mapping_results


def count_padding_in_window(window):
    """
    Conta il numero di time steps (righe) che sono interamente padding (tutti zeri).
    """
    if window.ndim == 1:
        return np.sum(window == 0)
    else:
        padding_rows = np.all(window == 0, axis=1)
        return np.sum(padding_rows)


def apply_downsampling(X, Y, window_metadata, max_windows_per_kid_action=50, seeds=[42, 43, 44, 45, 46]):
    """
    Applica downsampling limitando a max_windows_per_kid_action finestre per ogni 
    combinazione bambino/azione, eliminando prima quelle con più padding.
    LOGICA IBRIDA:
    1. Elimina TUTTE le finestre con padding (se rimangono comunque > max_windows)
    2. Se serve ancora, selezione casuale tra quelle senza padding
    """
    results = {}  # {seed: (X_filtered, Y_filtered, metadata_filtered, indices_to_keep)}

    
    logger.info(f"=== APPLICAZIONE DOWNSAMPLING ===")
    logger.info(f"Numero massimo di finestre per bambino/azione: {max_windows_per_kid_action}")

    for seed in seeds:
    # Imposta il seed per riproducibilità
        random.seed(seed)
        np.random.seed(seed)

        
        # Mappatura di ogni finestra con i suoi metadati
        window_info = []
        for i, metadata in enumerate(window_metadata):
            window_info.append({
                'index': i,
                'kid_id': metadata.get('kid_id', 'unknown'),
                'action_id': metadata.get('action_id', 'unknown'),
                'padding_count': count_padding_in_window(X[i])
            })
        
        # Raggruppa le finestre per bambino e azione
        kid_action_groups = defaultdict(list)
        for info in window_info:
            key = (info['kid_id'], info['action_id'])
            kid_action_groups[key].append(info)

        # Logica di filtraggio
        indices_to_keep = []
        
        for (kid_id, action_id), windows in kid_action_groups.items():

            if len(windows) <= max_windows_per_kid_action:
                indices_to_keep.extend([w['index'] for w in windows])
                logger.info(f"Gruppo kid={kid_id}, action={action_id}: Mantenute tutte le {len(windows)} finestre")
                continue

            #Analizza padding: # Split padding vs no padding
            windows_with_padding = [w for w in windows if w['padding_count'] > 0]
            windows_without_padding = [w for w in windows if w['padding_count'] == 0]

            logger.info(f"Finestre CON padding: {len(windows_with_padding)}")
            logger.info(f"Finestre SENZA padding: {len(windows_without_padding)}")

            # Caso 1: posso eliminare tutte quelle con padding
            if len(windows_without_padding) >= max_windows_per_kid_action:
                # Caso A: butto via tutte quelle col padding
                candidate_windows = windows_without_padding
                logger.info(f"FASE 1: Eliminate TUTTE le {len(windows_with_padding)} finestre con padding")
                logger.info(f"Rimangono {len(candidate_windows)} finestre senza padding")
                if len(candidate_windows) > max_windows_per_kid_action:
                    selected_windows = random.sample(candidate_windows, max_windows_per_kid_action)
                else:
                    selected_windows = candidate_windows

            else:
                 # Caso B: devo tenerne anche alcune col padding
                 # ordino le finestre per padding_count (dal più alto = peggiore)

                sorted_windows = sorted(windows, key=lambda w: w['padding_count'], reverse=True)
                # tengo le migliori (meno padding) fino a max_windows
                selected_windows = sorted_windows[-max_windows_per_kid_action:]

                # se ci sono più finestre "accettabili" del necessario, scelgo random tra le ultime
                if len(selected_windows) > max_windows_per_kid_action:
                    selected_windows = random.sample(selected_windows, max_windows_per_kid_action)

            indices_to_keep.extend([w['index'] for w in selected_windows])

        # Filtra i dati
        indices_to_keep.sort()
        X_filtered = X[indices_to_keep]
        Y_filtered = Y[indices_to_keep]
        metadata_filtered = [window_metadata[i] for i in indices_to_keep]

        results[seed] = (X_filtered, Y_filtered, metadata_filtered, indices_to_keep)
        
        logger.info(f"Downsampling completato:")
        logger.info(f"  Finestre originali: {len(X)}")
        logger.info(f"  Finestre mantenute: {len(X_filtered)}")
        logger.info(f"  Finestre rimosse: {len(X) - len(X_filtered)}")


        final_distribution = defaultdict(lambda: defaultdict(int))
        for meta in metadata_filtered:
            final_distribution[meta.get('action_id', 'unknown')][meta.get('kid_id', 'unknown')] += 1
        
        logger.info(f"\n DISTRIBUZIONE FINALE PER AZIONE:")
        for action_id, kids_dict in final_distribution.items():
            total_for_action = sum(kids_dict.values())
            logger.info(f"  Action {action_id}: {total_for_action} finestre totali")
            for kid_id, count in sorted(kids_dict.items()):
                logger.info(f"  Kid {kid_id}: {count} finestre")

    return results


def create_distribution_plot(Y, toy_name, title_suffix, save_path=None):
    """
    Crea un bar plot per la distribuzione delle classi.
    """
    
    distribution = Counter(Y)
    all_classes = sorted(distribution.keys())
    total_samples = len(Y)
    
    counts = [distribution[cls] for cls in all_classes]
    percentages = [(count / total_samples) * 100 for count in counts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
    bar_colors = [colors[i % len(colors)] for i in range(len(all_classes))]
    
    bars = ax.bar(range(len(all_classes)), percentages, color=bar_colors, 
                  alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Action ID', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{toy_name.upper()} - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels([f'{cls}' for cls in all_classes], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage, count) in enumerate(zip(bars, percentages, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
               f'{percentage:.1f}%\n({count})', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Plot salvato: {save_path}.png")
    
    plt.close(fig)
    
    # Stampa statistiche
    logger.info(f"\n=== {toy_name.upper()} - {title_suffix} ===")
    logger.info(f"Campioni totali: {total_samples}")
    logger.info(f"Numero classi: {len(all_classes)}")
    for i, cls in enumerate(all_classes):
        logger.info(f"  Classe {cls}: {counts[i]} campioni ({percentages[i]:.1f}%)")
    
    return fig


def get_target_actions_for_toy(toy_name):
    """
    Restituisce le azioni target per ogni giocattolo (quelle che NON vengono rimosse).
    """
    
    target_actions = {
        'car': [4, 6, 7, 8, 10, 13, 21, 23, 25, 41],  # Tutte meno quelle rimosse
        'doll': [6, 10, 11, 16, 18, 30, 32, 34, 41],  # Tutte meno quelle rimosse  
        'elephant': [3, 4, 6, 8, 9, 10, 11, 12, 13, 16, 18, 21, 27, 32, 36, 41],  # Tutte meno quelle rimosse
        'spoon': [4, 10, 11, 41],  # Tutte meno quelle rimosse
        'ball': [11, 19, 21, 41]  # Assumendo che ball abbia queste azioni
    }
    
    return target_actions.get(toy_name, [])

def inspect_zupt_results(pkl_path):
    """
    Ispeziona il contenuto del file PKL dei risultati ZUPT.
    """
    
    import pickle
    
    print(f"=== ISPEZIONE FILE: {pkl_path} ===\n")
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        print("STRUTTURA DEL FILE:")
        for key in results.keys():
            print(f"  - {key}: {type(results[key])}")
        
        print(f"\n INFORMAZIONI GENERALI:")
        print(f"  - Giocattolo: {results['toy_name']}")
        print(f"  - Azioni target: {results['target_actions']}")
        print(f"  - Finestre originali: {results['original_windows']}")
        print(f"  - Finestre dopo ZUPT: {results['zupt_filtered_windows']}")
        print(f"  - Finestre finali: {results['final_windows']}")
        print(f"  - Percentuale mantenuta: {results['final_windows']/results['original_windows']*100:.1f}%")
        
        print(f"\n PARAMETRI USATI:")
        for param, value in results['parameters'].items():
            print(f"  - {param}: {value}")
        
        print(f"\n ANALISI PAUSE:")
        pause_analysis = results['pause_analysis']
        print(f"  - File analizzati: {len(pause_analysis['files_analyzed'])}")
        
        total_pause_segments = 0
        for file_name, file_data in pause_analysis['files_analyzed'].items():
            file_segments = sum(len(action['pause_segments']) for action in file_data['actions_analysis'].values())
            total_pause_segments += file_segments
            if file_segments > 0:
                print(f"{file_name}: {file_segments} segmenti di pausa")
        
        print(f"  - Totale segmenti di pausa trovati: {total_pause_segments}")
        
        print(f"\nMAPPING RESULTS:")
        mapping = results['mapping_results']
        print(f"  - Finestre analizzate: {mapping['summary']['total_windows']}")
        print(f"  - Finestre rimosse per pause: {mapping['summary']['windows_removed']}")
        print(f"  - Percentuale rimossa per pause: {mapping['summary']['removal_percentage']:.1f}%")
        
        print(f"\nINDICI FINESTRE DA MANTENERE:")
        print(f"  - Numero indici: {len(results['windows_to_keep_indices'])}")
        print(f"  - Primi 10 indici: {results['windows_to_keep_indices'][:10]}")
        
        return results
        
    except Exception as e:
        print(f"ERRORE nel caricamento: {e}")
        return None


def main():
    """
    Funzione principale del preprocessing ZUPT.
    """
    
    parser = argparse.ArgumentParser(description="Preprocessing ZUPT per eliminare finestre con pause")
    parser.add_argument('--toy', type=str, required=True, 
                       choices=['ball', 'car', 'doll', 'spoon', 'elephant'],
                       help='Giocattolo da processare')
    parser.add_argument('--pause-threshold', type=float, default=0.5,
                       help='Soglia percentuale per eliminare finestre (default: 0.5 = 50%)')
    parser.add_argument('--max-windows', type=int, default=50,
                       help='Numero massimo di finestre per bambino/azione (default: 50)')
    parser.add_argument('--plot-zupt', action='store_true', default=True,
                       help='Genera plot delle analisi ZUPT')
    parser.add_argument('--output-dir', type=str, default='zupt_preprocessing_output',
                       help='Directory per salvare i risultati')
    
    args = parser.parse_args()
    
    # Setup logging
    exp_name = f"zupt_preprocessing_{args.toy}"
    setup_logging(exp_name)
    
    #CREA STRUTTURA DIRECTORY ORGANIZZATA PER GIOCATTOLO
    base_output_dir = args.output_dir
    toy_output_dir = os.path.join(base_output_dir, args.toy)  # zupt_preprocessing_output/car/
    
    os.makedirs(toy_output_dir, exist_ok=True)
    
    # Directory specifiche per questo giocattolo
    figures_dir = os.path.join(toy_output_dir, 'figures')
    reports_dir = os.path.join(toy_output_dir, 'reports')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger.info(f"=== INIZIO PREPROCESSING ZUPT PER {args.toy.upper()} ===")
    
    # Path dati
    data_path = "C:\\codes\\HumanActivityRecognition\\data\\downstream_data"
    
    # Ottieni le azioni target per questo giocattolo
    target_actions = get_target_actions_for_toy(args.toy)
    logger.info(f"Azioni target per {args.toy}: {target_actions}")
    
    # 1. CARICA E PREPARA I DATI (solo classi target)
    logger.info("\n=== STEP 1: CARICAMENTO DATI ===")
    df_toy, toy_mapping = load_and_preprocess_data_for_zupt(args.toy, data_path, target_actions)
    df_toy = add_consecutive_segment_id(df_toy)
    
    # 2. ANALISI ZUPT PER AZIONE
    logger.info("\n=== STEP 2: ANALISI ZUPT PER AZIONE ===")
    

    logger.info(f"Action IDs target: {target_actions}")

    
    pause_analysis = analyze_zupt_by_toy_and_action(
        data_path=data_path,
        toy_name=args.toy,
        target_actions=target_actions,
        plot_results=args.plot_zupt,
        save_plots=True,
        figures_dir=figures_dir
    )
    
    # 3. NORMALIZZAZIONE 
    logger.info("\n=== STEP 3: NORMALIZZAZIONE ===")
    df_normalized = df_toy  # nessuna normalizzazione

    # STEP 4: SLIDING WINDOW
    logger.info("\n=== STEP 4: SLIDING WINDOW ===")
    X_list, Y_list, all_consecutivity, all_windows_metadata_raw = [], [], [], []
    global_window_id = 0

    logger.info("=== DEBUG PRIMA DELLA SLIDING WINDOW ===")
    logger.info(f"DataFrame normalizzato shape: {df_normalized.shape}")
    logger.info(f"Action IDs unici: {df_normalized['action_id'].unique()}")
    logger.info(f"Colonne: {df_normalized.columns.tolist()}")

    temp_action_dir = os.path.join(data_path, "temp_zupt_actions")
    os.makedirs(temp_action_dir, exist_ok=True)

    for action_id in df_normalized['action_id'].unique():
        df_action = df_normalized[df_normalized['action_id'] == action_id]
        temp_path = os.path.join(temp_action_dir, f'df_zupt_{args.toy}_action_{action_id}.csv')
        df_action.to_csv(temp_path, index=False)

        logger.info(f"=== DEBUG ACTION {action_id} ===")
        logger.info(f"  Righe salvate: {len(df_action)}")
        logger.info(f"  File temporaneo: {temp_path}")
        logger.info(f"  Kid IDs: {df_action['kid_id'].unique()}")
    

        X_w, Y_w, kid_dict, is_consecutive, window_metadata = sliding_window_on_data.new_process_csv(
            temp_path, 9, 100, 50
        )
        
        X_list.append(X_w)
        Y_list.append(Y_w)
        all_consecutivity.extend(is_consecutive) 
        all_windows_metadata_raw.extend(window_metadata)
        
        logger.info(f"Azione {action_id}: {len(X_w)} finestre generate")

    shutil.rmtree(temp_action_dir)

    # Concatena tutti i dati
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0).flatten()
    all_consecutivity = np.array(all_consecutivity)  


    all_window_indices = []
    for i, meta in enumerate(all_windows_metadata_raw):
        all_window_indices.append({
            'global_window_id': i,
            'action_id': meta['action_id'],
            'kid_id': meta.get('kid_id', 'unknown'),
            'original_file': meta.get('original_file', 'unknown'),
            'row_indices': meta['row_indices']  
        })
    
    logger.info("=== VERIFICA METADATI DOPO SLIDING WINDOW ===")
    logger.info(f"Numero finestre: {len(X)}")
    logger.info(f"Numero metadati raw: {len(all_windows_metadata_raw)}")
    logger.info(f"Numero metadati processed: {len(all_window_indices)}")

    if len(all_window_indices) > 0:
        sample_meta = all_window_indices[0]
        logger.info(f"Chiavi primo metadato: {list(sample_meta.keys())}")
        
        if 'row_indices' in sample_meta:
            logger.info(f"row_indices PRESENTE: {sample_meta['row_indices'][:5] if len(sample_meta['row_indices']) > 5 else sample_meta['row_indices']}...")
        else:
            logger.error("row_indices MANCANTE!")
            logger.error("PROBLEMA: sliding_window_on_data.new_process_csv() non restituisce row_indices")

    
    logger.info(f"Totale finestre generate: {len(X)}")
    
    # Plot distribuzione iniziale
    create_distribution_plot(
        Y, args.toy, "Distribuzione Iniziale", 
        save_path=os.path.join(figures_dir, f"{args.toy}_distribuzione_iniziale")
    )
    
    # 5. MAPPING PAUSE -> FINESTRE
    logger.info("\n=== STEP 5: MAPPING PAUSE ALLE FINESTRE ===")
    mapping_results = map_pause_rows_to_windows(
    pause_analysis, all_window_indices, args.pause_threshold  
    )
    
    # 6. FILTRAGGIO FINESTRE CON PAUSE
    logger.info("\n=== STEP 6: FILTRAGGIO FINESTRE ===")
    if mapping_results['windows_to_remove']:
        indices_to_keep = [i for i in range(len(X)) if i not in mapping_results['windows_to_remove']]
        
        X_filtered = X[indices_to_keep]
        Y_filtered = Y[indices_to_keep]
        metadata_filtered = [all_window_indices[i] for i in indices_to_keep]  
        
        logger.info(f"Finestre rimosse per pause: {len(mapping_results['windows_to_remove'])}")
    else:
        X_filtered = X
        Y_filtered = Y
        metadata_filtered = all_window_indices
        indices_to_keep = list(range(len(X)))
        logger.info("Nessuna finestra rimossa per pause")
    
    # Plot distribuzione dopo filtraggio ZUPT
    create_distribution_plot(
        Y_filtered, args.toy, "Distribuzione Dopo Filtraggio ZUPT", 
        save_path=os.path.join(figures_dir, f"{args.toy}_distribuzione_dopo_zupt")
    )
    
    # 7. DOWNSAMPLING
    logger.info("\n=== STEP 7: DOWNSAMPLING ===")
    downsampling_results = apply_downsampling(X_filtered, Y_filtered, metadata_filtered, args.max_windows)
    logger.info(f"Downsampling eseguito con {len(downsampling_results)} seed diversi")
    # Plot distribuzione finale
    for seed, (X_final, Y_final, metadata_final, final_indices) in downsampling_results.items():
        create_distribution_plot(
        Y_final,
        args.toy,
        f"Distribuzione Finale (seed={seed})",
        save_path=os.path.join(figures_dir, f"{args.toy}_distribuzione_finale_seed{seed}")
        )
    
    # 8. SALVATAGGIO RISULTATI
    logger.info("\n=== STEP 8: SALVATAGGIO RISULTATI ===")

    # CALCOLO CORRETTO DEGLI INDICI FINALI RISPETTO AL DATASET ORIGINALE
    logger.info("=== CALCOLO INDICI FINALI ===")
    
    # Step 1: Indici mantenuti dopo ZUPT (rispetto al dataset originale)
    zupt_removed_indices = set(mapping_results['windows_to_remove'])
    indices_after_zupt = [i for i in range(len(X)) if i not in zupt_removed_indices]

    logger.info(f"Finestre originali: {len(X)}")
    logger.info(f"Finestre rimosse per ZUPT: {len(zupt_removed_indices)}")
    logger.info(f"Finestre dopo ZUPT: {len(indices_after_zupt)}")

    # Itera su tutti i seed
    for seed, (X_final, Y_final, metadata_final, final_indices) in downsampling_results.items():
        logger.info(f"\n=== SALVATAGGIO SPLIT SEED={seed} ===")
        seed_dir = os.path.join(toy_output_dir, f"seed_{seed}")  # zupt_preprocessing_output/car/seed_42/
        seed_figures_dir = os.path.join(seed_dir, "figures")     # zupt_preprocessing_output/car/seed_42/figures/

        os.makedirs(seed_dir, exist_ok=True)
        os.makedirs(seed_figures_dir, exist_ok=True)
        
        logger.info(f"Cartella seed: {seed_dir}")

        # Step 2: Indici finali dopo downsampling (rispetto al dataset originale)
        # final_indices contiene gli indici rispetto a X_filtered
        # li dobbiamo mappare agli indici originali
        final_indices_global = [indices_after_zupt[i] for i in final_indices]

        logger.info(f"Finestre mantenute dal downsampling: {len(final_indices)}")
        logger.info(f"Finestre finali (rispetto originale): {len(final_indices_global)}")

        # VERIFICA CORRETTEZZA
        logger.info("=== VERIFICA CORRETTEZZA INDICI ===")
        logger.info(f"Indici finali (primi 10): {final_indices_global[:10]}")
        logger.info(f"Max indice finale: {max(final_indices_global) if final_indices_global else 'N/A'}")
        logger.info(f"Min indice finale: {min(final_indices_global) if final_indices_global else 'N/A'}")

        # Verifica che tutti gli indici siano validi
        if final_indices_global and max(final_indices_global) >= len(X):
            logger.error(f"ERRORE: Indice fuori range {max(final_indices_global)} >= {len(X)}")
            raise ValueError("Errore nel calcolo degli indici finali!")

        # VERIFICA CHE I DATI CORRISPONDANO
        if len(final_indices_global) != len(X_final):
            logger.error(f"ERRORE: Mismatch indici {len(final_indices_global)} vs dati {len(X_final)}")
            raise ValueError("Errore nella corrispondenza indici-dati!")

        logger.info("Indici calcolati correttamente!")

        # TEST: Verifica che estraendo dalle finestre originali ottieni le stesse finestre finali
        if len(final_indices_global) > 0:
            test_X = X[final_indices_global]
            test_Y = Y[final_indices_global]
            
            # Dovrebbero essere identici a X_final e Y_final
            arrays_match = np.allclose(test_X, X_final, rtol=1e-5, atol=1e-8)
            labels_match = np.array_equal(test_Y, Y_final)
            
            logger.info(f"Test corrispondenza X: {'PASS' if arrays_match else 'FAIL'}")
            logger.info(f"Test corrispondenza Y: {'PASS' if labels_match else 'FAIL'}")
            
            if not (arrays_match and labels_match):
                logger.error("ATTENZIONE: I dati estratti non corrispondono ai dati finali!")
                logger.error("Questo indica un errore nel calcolo degli indici.")

        # Salva un file dedicato per il caricamento rapido 
        filtered_indices_for_har = {
            'toy_name': args.toy,
            'target_actions': target_actions,
            'original_total_windows': len(X),  # Numero finestre prima di qualsiasi filtraggio
            'final_total_windows': len(X_final),  # Numero finestre dopo tutti i filtraggi
            'windows_indices_to_keep': final_indices_global,  # Indici corretti rispetto al dataset originale
            'processing_info': {
                'zupt_threshold': args.pause_threshold,
                'max_windows_per_kid_action': args.max_windows,
                'seed': seed,
                'zupt_removed': len(mapping_results['windows_to_remove']),
                'downsampling_removed': len(X_filtered) - len(X_final),
                'total_removed': len(X) - len(X_final)
            },
            'metadata_sample': {
                'first_kept_window': all_window_indices[final_indices_global[0]] if final_indices_global else None,
                'total_actions': len(target_actions),
                'actions_distribution': dict(Counter(Y_final))
            },
            #INFO DI DEBUG
            'debug_info': {
                'zupt_removed_indices': sorted(list(zupt_removed_indices)),
                'indices_after_zupt': indices_after_zupt[:10],  # primi 10 per debug
                'final_indices_in_filtered': final_indices[:10],  # primi 10 per debug
                'final_indices_in_original': final_indices_global[:10]  # primi 10 per debug
            }
        }

        # Salva in un file JSON più leggibile per debug
        indices_json_path = os.path.join(seed_dir, f'{args.toy}_filtered_indices_seed{seed}.json')
        import json
        def convert_to_json_serializable(obj):
            """
            Converte ricorsivamente tutti i tipi numpy in tipi Python nativi
            """
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_json_serializable(item) for item in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_json_serializable(obj.tolist())
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj


        with open(indices_json_path, 'w') as f:
            json_compatible = convert_to_json_serializable(filtered_indices_for_har)
            json.dump(json_compatible, f, indent=2)


        indices_pkl_path = os.path.join(seed_dir, f'{args.toy}_filtered_indices_seed{seed}.pkl')
        with open(indices_pkl_path, 'wb') as f:
            pickle.dump(filtered_indices_for_har, f)

        logger.info(f"Indici filtrati salvati per run_har_experiment.py (seed={seed}):")
        logger.info(f" JSON (human-readable): {indices_json_path}")  
        logger.info(f" PKL (fast loading): {indices_pkl_path}")
        logger.info(f" Finestre da mantenere: {len(final_indices_global)}/{len(X)}")

        # AGGIORNA ANCHE IL RISULTATO COMPLETO CON GLI INDICI CORRETTI
        results = {
            'toy_name': args.toy,
            'target_actions': target_actions,
            'original_windows': len(X),
            'zupt_filtered_windows': len(X_filtered),
            'final_windows': len(X_final),
            'windows_to_keep_indices': final_indices_global,  
            'pause_analysis': pause_analysis,
            'mapping_results': mapping_results,
            'parameters': {
                'pause_threshold': args.pause_threshold,
                'max_windows_per_kid_action': args.max_windows,
                'seed': seed
            }
        }

        
        # Salva in pickle per caricare nel script principale
        results_path = os.path.join(seed_dir, f'{args.toy}_zupt_filtering_results_seed{seed}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Risultati salvati in: {results_path}")
        
        # Salva anche un report testuale
        report_path = os.path.join(reports_dir, f'{args.toy}_zupt_report_seed{seed}.txt')
        with open(report_path, 'w') as f:
            f.write(f"ZUPT PREPROCESSING REPORT - {args.toy.upper()} (SEED={seed})\n")
            f.write("="*50 + "\n\n")
            f.write(f"Parametri:\n")
            f.write(f"  - Soglia pause: {args.pause_threshold*100}%\n")
            f.write(f"  - Max finestre per bambino/azione: {args.max_windows}\n\n")
            f.write(f"Risultati:\n")
            f.write(f"  - Finestre originali: {len(X)}\n")
            f.write(f"  - Finestre dopo filtraggio ZUPT: {len(X_filtered)}\n")
            f.write(f"  - Finestre finali: {len(X_final)}\n")
            f.write(f"  - Finestre rimosse per pause: {len(X) - len(X_filtered)}\n")
            f.write(f"  - Finestre rimosse per downsampling: {len(X_filtered) - len(X_final)}\n")
            f.write(f"  - Percentuale finale mantenuta: {len(X_final)/len(X)*100:.1f}%\n")
        
        logger.info(f"Report salvato in: {report_path}")
        
        logger.info(f"\n=== PREPROCESSING COMPLETATO PER {args.toy.upper()} ===")
        for seed in downsampling_results.keys():
            results_path = os.path.join(args.output_dir, f'{args.toy}_zupt_filtering_results_seed{seed}.pkl')
            logger.info(f"Per usare questi risultati, carica il file: {results_path}")



if __name__ == "__main__":
    main()
    # Per debug: ispeziona il file del primo seed (es. seed=42)
    default_seed = 42
    pkl_path = f"zupt_preprocessing_output/car_zupt_filtering_results_seed{default_seed}.pkl"
    if os.path.exists(pkl_path):
        inspect_zupt_results(pkl_path)


