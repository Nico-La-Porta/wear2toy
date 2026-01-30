import os
import sys
import shutil
from typing import Optional, Tuple
from typing import List, Dict
import numpy as np
from scipy.signal import resample
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history as optuna_plot_history
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'HumanActivityRecognition')))
sys.path.append(os.path.abspath('HumanActivityRecognition'))
from src.app_config import PROJ_ROOT, REPORTS_DIR, FIGURES_DIR, MODELS_DIR,  OUTPUT_WISDIM_DATA_DIR, SELECTED_ACTIVITIES, FS_TARGET, FS_ORIGINAL


import matplotlib.pyplot as plt
from src.utils.transformations import *
from src.utils.transformations_utils import *
from collections import defaultdict


from src.utils import data_processing
from src.utils import init_weights, train_with_cm
from src.utils.log_config import logger
from src.models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn, create_weighted_sampler
from src.pretraining.figures import plot_CM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_JSON_DIR = os.path.join(OUTPUT_WISDIM_DATA_DIR,  'json_export_train')
TEST_JSON_DIR  = os.path.join(OUTPUT_WISDIM_DATA_DIR,  'json_export_test')

BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"best_deepconvlstm_wisdm_meanstd_aug_sampler_seed_{SEED}.pt")
BEST_SCORE_PATH = os.path.join(MODELS_DIR, f"best_deepconvlstm_wisdm_meanstd_aug_sampler_seed_{SEED}.score.txt")

best_hyperparams_file = os.path.join(REPORTS_DIR, f'best_hyperparameters_wisdm_meanstd_aug_sampler_seed_{SEED}.csv')


NB_SENSOR_CHANNELS = 6 #x,y,z di accelerometro e giroscopio
SLIDING_WINDOW_LENGTH = 100 #samples (2s a 100Hz)
SLIDING_WINDOW_STEP = 50 #samples (1s a 100Hz)



os.makedirs(OUTPUT_WISDIM_DATA_DIR, exist_ok=True)
label_to_idx = {lab: i for i, lab in enumerate(SELECTED_ACTIVITIES)}
idx_to_label = {i: lab for lab, i in label_to_idx.items()}
import json

def build_dataset_from_json(path: str):
    """
    Legge tutti i <trace>.json in `path`, carica i corrispondenti .npy
    e ritorna una lista di tracce nello stesso schema atteso da
    apply_sliding_window_wisdm().
    """
    files = [f for f in os.listdir(path) if f.lower().endswith(".json")]
    files.sort()
    dataset_traces = []

    for jf in files:
        jpath = os.path.join(path, jf)
        with open(jpath, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # metadata
        base = meta.get("trace_id", os.path.splitext(jf)[0])

        # carica array (portabile tra versioni numpy)
        npy_name = meta["signals"]["data_path"]
        npy_path = os.path.join(path, npy_name)
        data = np.load(npy_path, mmap_mode="r")  # efficiente in RAM
        chs  = meta["signals"].get("chs", [])
        fs   = meta["signals"].get("fs", None)

        # annotazioni (possono mancare)
        ann = meta.get("annotations", None)
        if ann is None or "labels" not in ann or "idx" not in ann:
            print(f"[WARN] {jf}: nessuna annotazione valida; salto.")
            continue

        labels = ann["labels"]
        idx    = ann["idx"]

        trace = {
            "TraceID": base,
            "signals": {
                "data": data,    # (T, C)
                "chs": chs,
                "fs": fs
            },
            "annotations": {
                "labels": labels,
                "idx": idx
            }
        }
        dataset_traces.append(trace)

    print(f"[OK] Caricate {len(dataset_traces)} tracce JSON da {path}")
    return dataset_traces

def extract_subject_id_from_name(filename: str) -> str:
    """
    Estrae l'ID soggetto dal nome file WISDM, es. 'Dwisdm_S1600_R1_T0_sig.npz' -> 'S1600'
    """
    parts = filename.split('_')
    for p in parts:
        if p.startswith('S'):
            return p
    return "Unknown"

def plot_wisdm_subject_activity_distribution(WISDIM_DATA_DIR, FIGURES_DIR):
    """
    Plotta distribuzione delle attività per ciascun soggetto
    del dataset WISDM e salva i grafici in FIGURES_DIR/wisdm_subject_activities
    """
    output_dir = os.path.join(FIGURES_DIR, "wisdm_subject_activities")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Directory di output creata: {output_dir}")

    sig_files = [f for f in os.listdir(WISDIM_DATA_DIR) if f.endswith("_sig.npz")]
    ann_files = [f for f in os.listdir(WISDIM_DATA_DIR) if f.endswith("_ann.npz")]

    subjects_data = defaultdict(lambda: defaultdict(float))  # {subj: {act: seconds}}

    for sig_file in sig_files:
        base = sig_file.replace("_sig.npz", "")
        ann_file = base + "_ann.npz"
        if ann_file not in ann_files:
            print(f"[WARN] Annotazioni mancanti per {sig_file}")
            continue

        try:
            sig = np.load(os.path.join(WISDIM_DATA_DIR, sig_file), allow_pickle=True)
            ann = np.load(os.path.join(WISDIM_DATA_DIR, ann_file), allow_pickle=True)
            
            fs = int(sig["fs"])
            labels = ann["activities"].item()["labels"]
            indices = ann["activities"].item()["idx"]

            subj_id = extract_subject_id_from_name(sig_file)

            # calcola durata di ciascuna attività
            for i, act in enumerate(labels):
                start = indices[i]
                end = indices[i+1] if i+1 < len(indices) else sig["data"].shape[0]
                dur_s = (end - start) / fs
                subjects_data[subj_id][act] += dur_s

        except Exception as e:
            print(f"[ERRORE] {sig_file}: {e}")
            continue

    # tutte le attività trovate
    all_activities = sorted({a for d in subjects_data.values() for a in d})
    subjects = sorted(subjects_data.keys())

    # --- overview heatmap ---
    mat = np.zeros((len(subjects), len(all_activities)))
    for i, subj in enumerate(subjects):
        for j, act in enumerate(all_activities):
            mat[i, j] = subjects_data[subj].get(act, 0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(subjects)*0.35)))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(all_activities)))
    ax.set_xticklabels(all_activities, rotation=45, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    plt.colorbar(im, ax=ax, label="Durata (s)")
    ax.set_title("WISDM - Distribuzione Attività per Soggetto", fontsize=14, fontweight="bold")
    plt.tight_layout()

    overview_path = os.path.join(output_dir, "overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Overview salvata: {overview_path}")

    # --- grafici individuali ---
    indiv_dir = os.path.join(output_dir, "individual_subjects")
    os.makedirs(indiv_dir, exist_ok=True)

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_activities)))

    for subj, acts in subjects_data.items():
        acts = {a: d for a, d in acts.items() if d > 0}
        if not acts:
            continue
        labels, durations = zip(*sorted(acts.items(), key=lambda x: x[1], reverse=True))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # bar chart
        bars = ax1.bar(labels, durations, 
                       color=[colors[all_activities.index(a)] for a in labels],
                       edgecolor="black")
        ax1.set_ylabel("Durata (s)")
        ax1.set_title(f"{subj} - Durata per Attività")
        ax1.tick_params(axis='x', rotation=45)
        for bar, dur in zip(bars, durations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{dur:.1f}", ha="center", va="bottom", fontsize=8)

        # pie chart
        ax2.pie(durations, labels=labels, autopct="%1.1f%%",
                colors=[colors[all_activities.index(a)] for a in labels])
        ax2.set_title(f"{subj} - Percentuale Attività")

        plt.tight_layout()
        save_path = os.path.join(indiv_dir, f"{subj}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[OK] Grafici individuali salvati in: {indiv_dir}")

def resample_signal(signal: np.ndarray, fs_original: float, fs_target: float, axis: int=0) -> np.ndarray:
    """
    Resampling di un segnale da fs_original a fs_target.

    Parameters:
    - signal (numpy.ndarray): array 1D o 2D (T, C) dove T=tempo, C=canali.
    - fs_original (float): frequenza di campionamento originale (Hz).
    - fs_target (float): frequenza di campionamento desiderata (Hz).

    Returns:
    - resampled_signal (numpy.ndarray): segnale ricampionato.
    """

    T = signal.shape[axis]
    # Numero di campioni nel segnale ricampionato
    num_samples_target = int(len(signal) * fs_target / fs_original)

    # Resampling con Fourier method
    resampled_signal = resample(signal, num_samples_target, axis=0)

    return resampled_signal

def collect_wisdm_npz_pairs(data_dir: str) -> List[Dict[str, str]]:
    """
    Colleziona tutte le coppie (sig, ann) disponibili in data_dir.
    Ritorna una lista di dict: {"subject", "sig_path", "ann_path"}
    Se l'ann mancante, ann_path=None (si può comunque usare solo il segnale).
    """
    sig_files = [f for f in os.listdir(data_dir) if f.endswith("_sig.npz")]
    ann_files = set([f for f in os.listdir(data_dir) if f.endswith("_ann.npz")])

    pairs = []
    for sig_file in sorted(sig_files):
        base = sig_file.replace("_sig.npz", "")
        ann_file = base + "_ann.npz"
        subject = extract_subject_id_from_name(sig_file)
        sig_path = os.path.join(data_dir, sig_file)
        ann_path = os.path.join(data_dir, ann_file) if ann_file in ann_files else None
        pairs.append({"subject": subject, "sig_path": sig_path, "ann_path": ann_path})
    return pairs

def calculate_user_threshold(user_ids: List[str], train_ratio: float = 0.8, seed: int = 44) -> List[str]:
    """
    Seleziona casualmente la lista di utenti per il train in base al train_ratio.
    """
    random.seed(seed)
    user_ids = sorted(set(user_ids))
    total_users = len(user_ids)
    threshold = int(round(train_ratio * total_users))
    print(f"Total users: {total_users}")
    print(f"Users for training: {threshold}")
    selected_users = random.sample(user_ids, k=threshold)
    print(f"Selected user IDs for training: {sorted(selected_users)}")
    return selected_users

def split_dataset_npz(
    pairs: List[Dict[str, str]],
    train_ratio: float = 0.8,
    seed: int = 44,
    include_user: Optional[str] = None,
    exclude_user: Optional[str] = None
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split per soggetto dei file NPZ (sig/ann).
    - pairs: lista da collect_wisdm_npz_pairs
    - include_user: se specificato, forza l'utente nel train
    - exclude_user: se specificato, rimuove l'utente dal train (andranno nel test)
    Ritorna (train_set, test_set) come liste di dict {subject, sig_path, ann_path}.
    """
    user_ids = sorted(set(p["subject"] for p in pairs if p["subject"] != "Unknown"))
    if include_user and include_user not in user_ids:
        raise ValueError(f"User {include_user} not found in dataset.")
    if exclude_user and exclude_user not in user_ids:
        raise ValueError(f"User {exclude_user} not found in dataset.")

    selected_users = calculate_user_threshold(user_ids, train_ratio=train_ratio, seed=seed)

    # forza include_user nel train
    if include_user and include_user not in selected_users:
        selected_users.append(include_user)

    # esclude dal train (finirà nel test)
    if exclude_user:
        selected_users = [u for u in selected_users if u != exclude_user]

    selected_users = sorted(set(selected_users))

    # build split
    train_set, test_set = [], []
    train_users = set(selected_users)
    for item in pairs:
        subj = item["subject"]
        if subj in train_users:
            train_set.append(item)
        else:
            test_set.append(item)

    # sanity check: ogni soggetto interamente in uno dei due
    assert set(x["subject"] for x in train_set).isdisjoint(set(y["subject"] for y in test_set)), \
        "Uno stesso soggetto è finito in entrambi gli split! Controlla la logica di split."

    print(f"\n[SUMMARY] Train users: {len(set(x['subject'] for x in train_set))} | "
          f"Test users: {len(set(x['subject'] for x in test_set))}")
    print(f"Train files: {len(train_set)} | Test files: {len(test_set)}")
    return train_set, test_set
def scale_indices_preserve_end(idx_list, old_len: int, new_len: int):
    """
    Scala gli indici delle annotazioni dal vecchio al nuovo dominio campioni.
    Usa ratio=new_len/old_len per stabilità numerica, arrotonda al più vicino intero,
    forza ultimo indice = new_len, clamp [0,new_len], garantisce monotonia.
    """
    if old_len <= 0:
        raise ValueError("old_len deve essere > 0")
    ratio = new_len / float(old_len)
    idx_arr = np.asarray(idx_list, dtype=float)
    idx_scaled = np.rint(idx_arr * ratio).astype(int)

    if idx_scaled.size > 0:
        idx_scaled[-1] = new_len  # chiusura segmento finale

    # clamp
    idx_scaled = np.clip(idx_scaled, 0, new_len)

    # monotonia non decrescente
    for i in range(1, idx_scaled.size):
        if idx_scaled[i] < idx_scaled[i-1]:
            idx_scaled[i] = idx_scaled[i-1]

    return idx_scaled.tolist()
def save_split_to_folders(train_set, test_set, output_dir: str):
    """
    Copia i file NPZ in due cartelle: train/ e test/
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for item in train_set:
        shutil.copy(item["sig_path"], train_dir)
        if item["ann_path"]:
            shutil.copy(item["ann_path"], train_dir)

    for item in test_set:
        shutil.copy(item["sig_path"], test_dir)
        if item["ann_path"]:
            shutil.copy(item["ann_path"], test_dir)

    print(f"[OK] Train files copied: {len(train_set)} subjects → {train_dir}")
    print(f"[OK] Test files copied: {len(test_set)} subjects → {test_dir}")
#FUNZIONE PRINCIPALE PER RICAMPIONARE TUTTI I FILE
def process_all_wisdm_fft(data_dir: str, out_dir: str, fs_target: float):
    sig_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_sig.npz")])
    ann_files = set([f for f in os.listdir(data_dir) if f.endswith("_ann.npz")])

    print(f"[INFO] Trovati {len(sig_files)} file segnali in {data_dir}")
    os.makedirs(out_dir, exist_ok=True)

    for k, sig_file in enumerate(sig_files, start=1):
        base = sig_file.replace("_sig.npz", "")
        ann_file = base + "_ann.npz"
        sig_path = os.path.join(data_dir, sig_file)
        ann_path = os.path.join(data_dir, ann_file)

        print(f"\n[{k}/{len(sig_files)}] {sig_file}")

        # Carica segnale
        sig_npz = np.load(sig_path, allow_pickle=True)
        data = sig_npz["data"]          # (T, C) oppure (T,)
        chs = sig_npz["chs"]
        fs_original = float(sig_npz["fs"])

        old_len = data.shape[0]

        # Ricampionamento (asse tempo = 0)
        data_rs = resample_signal(data, fs_original, fs_target, axis=0)
        new_len = data_rs.shape[0]

        # Carica e scala annotazioni 
        activities_out = None
        if ann_file in ann_files:
            ann_npz = np.load(ann_path, allow_pickle=True)
            # activities può essere dict o array object -> uniformO
            activities = ann_npz["activities"].item() if isinstance(ann_npz["activities"], np.ndarray) else ann_npz["activities"]
            labels = list(activities["labels"])
            idx = list(activities["idx"])

            idx_scaled = scale_indices_preserve_end(idx, old_len=old_len, new_len=new_len)
            activities_out = {"labels": labels, "idx": idx_scaled}
        else:
            print(f"[WARN] Annotazioni mancanti per {sig_file} → salvo solo segnale.")

        # Salvataggio
        out_sig_path = os.path.join(out_dir, sig_file)
        out_ann_path = os.path.join(out_dir, ann_file)

        np.savez(out_sig_path, data=data_rs.astype(np.float32), chs=chs, fs=np.array(fs_target, dtype=np.float32))
        if activities_out is not None:
            np.savez(out_ann_path, activities=activities_out)

        print(f"[OK] Segnale: {old_len} → {new_len} campioni, fs: {fs_original} → {fs_target} Hz")
        if activities_out is not None:
            ok_end = (activities_out["idx"][-1] == new_len)
            print(f"[OK] Annotazioni salvate. Ultimo idx={activities_out['idx'][-1]} | len={new_len} | match={ok_end}")
            # check coerenza segmenti (labels deve avere len = len(idx)-1)
            if len(activities_out["labels"]) != len(activities_out["idx"]) - 1:
                print(f"[ATTENZIONE] labels={len(activities_out['labels'])} ma idx={len(activities_out['idx'])} (atteso idx = labels+1)")

    print(f"\n[FINITO] Output in: {out_dir}")


def build_dataset(path):
    """
    Carica tutte le coppie (signal, annotation) da una cartella WISDM
    e ritorna una lista di tracce.
    """
    files = os.listdir(path)
    sig_files = [f for f in files if f.endswith("_sig.npz")]
    ann_files = set(f for f in files if f.endswith("_ann.npz"))

    dataset_traces = []

    for sig_file in sig_files:
        base = sig_file.replace("_sig.npz", "")
        ann_file = base + "_ann.npz"

        if ann_file not in ann_files:
            print(f"[WARN] Missing annotation for {sig_file}")
            continue

        # Carica segnali
        sig = np.load(os.path.join(path, sig_file), allow_pickle=True)
        data = sig["data"]        # (T, C)
        chs = sig["chs"]          # nomi canali
        fs = sig["fs"].item() if hasattr(sig["fs"], "item") else sig["fs"]

        # Carica annotazioni
        ann = np.load(os.path.join(path, ann_file), allow_pickle=True)
        activities = ann["activities"].item() if isinstance(ann["activities"], np.ndarray) else ann["activities"]
        labels = activities["labels"]
        idx = activities["idx"]

        trace = {
            "TraceID": base,
            "signals": {
                "data": data,
                "chs": chs,
                "fs": fs
            },
            "annotations": {
                "labels": labels,
                "idx": idx
            }
        }
        dataset_traces.append(trace)

    print(f"[OK] Caricate {len(dataset_traces)} tracce da {path}")
    return dataset_traces


def apply_sliding_window_wisdm(dataset, sliding_window_length, sliding_window_step, nb_sensor_channels=6, selected_activities=None):
    """
    Applica sliding window a tutto il dataset WISDM.
    Ogni finestra riceve l'etichetta corrispondente al segmento (activities).
    Filtra solo le attività selezionate (se specificato).
    
    Params:
    - dataset: lista di tracce (da build_dataset)
    - sliding_window_length: lunghezza finestra (in campioni)
    - sliding_window_step: step finestra (in campioni)
    - nb_sensor_channels: numero canali sensore (WISDM = 6)
    
    Returns:
    - X: np.array di shape (N_finestre, sliding_window_length, nb_sensor_channels)
    - Y: np.array di shape (N_finestre,) con le etichette
    """
    X, Y = [], []

    for trace in dataset:
        data = trace["signals"]["data"]   # (T, 6)
        labels = trace["annotations"]["labels"]
        idx = trace["annotations"]["idx"]

        for i, label in enumerate(labels):
            if selected_activities and label not in selected_activities:
                continue
            start = idx[i]
            end = idx[i+1] if i+1 < len(idx) else data.shape[0]
            segment = data[start:end]  # segmento relativo a questa attività

            if segment.shape[0] < sliding_window_length:
                # troppo corto per una finestra → viene gestito da sliding_window col padding
                pass

            # X_windows: (n_finestre, sliding_window_length, nb_sensor_channels)
            X_windows = data_processing.sliding_window(
                segment,
                ws=(sliding_window_length, segment.shape[1]),
                ss=(sliding_window_step, segment.shape[1]),
                flatten=False
            )

            if X_windows.shape[0] == 0:
                continue

            #rimuove canali extra se presenti
            if X_windows.ndim == 4 and X_windows.shape[1] == 1:
                X_windows = X_windows[:, 0, :, :]   # (num_windows, win_len, n_channels)

            # assegna la stessa label a tutte le finestre del segmento
            Y_windows = np.array([label] * X_windows.shape[0])

            print(f"Trace {trace['TraceID']} - Segment {i} ({label}) "
                  f"len={segment.shape[0]} → windows={X_windows.shape[0]}")

            X.append(X_windows)
            Y.append(Y_windows)

    # concatena tutte le finestre
    if len(X) == 0:
        return np.array([]), np.array([])

    X = np.vstack(X).astype(np.float32)  # (N, L, C)
    Y = np.concatenate(Y)                # (N,)

    return X, Y


def encode_labels(y_np):
    # y_np può essere array di stringhe ('A'..'E') o già interi
    if y_np.dtype.kind in ("U","S","O"):  # string-like/object
        return np.array([label_to_idx[str(v)] for v in y_np], dtype=np.int64)
    return y_np.astype(np.int64, copy=False)

def to_channels_first(X_np):
    #  X: (N, L, C) -> (N, C, L)
    if X_np.ndim != 3:
        raise ValueError(f"Atteso X a 3 dimensioni (N,L,C); trovato {X_np.shape}")
    X_np = X_np.astype(np.float32, copy=False)
    return np.transpose(X_np, (0, 2, 1))


def compute_mean_std_from_windows(X: np.ndarray):
    """
    Calcola media e std per canale a partire da finestre (N,L,C).
    """
    reshaped = X.reshape(-1, X.shape[2])   # (N*L, C)
    means = np.mean(reshaped, axis=0)
    stds  = np.std(reshaped, axis=0)
    stds[stds == 0] = 1e-6
    return means, stds

def normalize_with_mean_std(X: np.ndarray, means: np.ndarray, stds: np.ndarray):
    """
    Normalizza finestre (N,L,C) rispetto a media/std per canale.
    """
    return (X - means) / stds
def main():
    
    #------PLOTTAGGIO DISTRIBUZIONE ATTIVITA' PER SOGGETTO
    #plot_wisdm_subject_activity_distribution(WISDIM_DATA_DIR, FIGURES_DIR)
    
    
    #------RICAMPIONAMENTO TUTTI I FILE + SALVATAGGIO IN NUOVA CARTELLA
    #process_all_wisdm_fft(WISDIM_DATA_DIR, OUTPUT_WISDIM_DATA_DIR, FS_TARGET)


    #------SPLIT TRAIN/TEST PER SOGGETTO
    #pairs= collect_wisdm_npz_pairs(OUTPUT_WISDIM_DATA_DIR)
    #train_set, test_set = split_dataset_npz(pairs, train_ratio=0.8, seed=44, include_user=None, exclude_user=None)
    #save_split_to_folders(train_set, test_set, OUTPUT_WISDIM_DATA_DIR)

    # --- STEP 3: Caricamento dataset ---
    # === Caricamento da JSON ===
    train_path = TRAIN_JSON_DIR  # oppure os.path.join(OUTPUT_WISDIM_DATA_DIR, "train")
    test_path  = TEST_JSON_DIR  


    print("\nCaricamento dataset TRAIN (JSON+NPY)...")
    train_dataset = build_dataset_from_json(train_path)

    print("\nCaricamento dataset TEST (JSON+NPY)...")
    test_dataset = build_dataset_from_json(test_path)

    print(f"\n[SUMMARY] Train set: {len(train_dataset)} tracce | Test set: {len(test_dataset)} tracce")

    # --- STEP 4: Sliding window ---

    X_train, Y_train = apply_sliding_window_wisdm(train_dataset, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, nb_sensor_channels=NB_SENSOR_CHANNELS, selected_activities=SELECTED_ACTIVITIES)
    X_test,  Y_test  = apply_sliding_window_wisdm(test_dataset, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, nb_sensor_channels=NB_SENSOR_CHANNELS, selected_activities=SELECTED_ACTIVITIES)

    print("Train:", X_train.shape, Y_train.shape)
    print("Test:", X_test.shape, Y_test.shape)

    # === Calcolo media/std sul TRAIN ===
    means, stds = compute_mean_std_from_windows(X_train)
    print("Means:", means)
    print("Stds:", stds)

    # === Normalizzazione train e test con quelle statistiche ===
    X_train = normalize_with_mean_std(X_train, means, stds)
    X_test  = normalize_with_mean_std(X_test,  means, stds)

    #Transformazioni finali
    Y_train_enc = encode_labels(Y_train)
    Y_test_enc  = encode_labels(Y_test)
    X_train_cf  = to_channels_first(X_train)
    X_test_cf   = to_channels_first(X_test)

    #Assert per verificare correttezza
    assert X_train_cf.shape[1] == NB_SENSOR_CHANNELS, f"Canali attesi {NB_SENSOR_CHANNELS}, trovati {X_train_cf.shape[1]}"
    assert X_train_cf.shape[2] == SLIDING_WINDOW_LENGTH, f"L finestra attesa {SLIDING_WINDOW_LENGTH}, trovata {X_train_cf.shape[2]}"
    assert X_train_cf.shape[0] == Y_train_enc.shape[0], "Mismatch N finestre train"
    assert X_test_cf.shape[0]  == Y_test_enc.shape[0],  "Mismatch N finestre test"

    #DATA AUGUMENTATION
    transform_funcs = [
        # transformations.scaling_transform_vectorized, # Use Scaling trasnformation
        noise_transform_vectorized, # Use rotation trasnformation
        scaling_transform_vectorized,
        #rotation_transform_vectorized,
        #axis_angle_to_rotation_matrix_3d_vectorized,
        negate_transform_vectorized,
        time_flip_transform_vectorized,
        intra_sensor_channel_shuffle_transform_vectorized,
        #time_segment_permutation_transform_improved,
        #get_cubic_spline_interpolation,
        time_warp_transform_improved,
        time_warp_transform_low_cost,
    ]
    transformation_function = generate_composite_transform_function_simple(transform_funcs)

    tranform_1 = transformation_function(X_train_cf)
    X_train_cf.shape, tranform_1.shape

    print(f"Dimensioni di X_Train: {X_train_cf.shape}")
    print(f"Dimensioni di Y_Train: {X_train_cf.shape}")
    X_Train_augmented = np.concatenate((X_train_cf, tranform_1), axis=0)
    Y_Train_augmented = np.concatenate((Y_train_enc, Y_train_enc), axis=0)
    print(f"Dimensioni di X_Train_augmented: {X_Train_augmented.shape}")
    print(f"Dimensioni di Y_Train_augmented: {Y_Train_augmented.shape}")

    


    #Creazione dataset
    train_dataset = HARDataset(X_Train_augmented, Y_Train_augmented, class_names=SELECTED_ACTIVITIES)
    test_dataset  = HARDataset(X_test_cf,  Y_test_enc,  class_names=SELECTED_ACTIVITIES)

    train_sampler = create_weighted_sampler(Y_Train_augmented)

    if os.path.exists(best_hyperparams_file):
        logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
        best_hyperparameters = pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
        best_lr = float(best_hyperparameters['lr'])
        best_batch_size = int(best_hyperparameters['batch_size'])
    else:
        logger.info("Nessun file di iperparametri trovato, eseguo l'ottimizzazione")

        def objective(trial):
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler,  num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
            batch_data = next(iter(train_loader))
            print("Shape batch input:", batch_data[0].shape)  # dovrebbe stampare (64, 6, 100)

            # Creo il modello con gli iperparametri suggeriti
            net = DeepConvLSTM(nb_sensor_channels=NB_SENSOR_CHANNELS, n_classes=len(SELECTED_ACTIVITIES))

            # Eseguo l'allenamento
            best_f1_score, _ = train_with_cm.train(
                net,
                train_loader,
                test_loader,
                exp_figures_dir=FIGURES_DIR,
                exp_reports_dir=REPORTS_DIR,
                epochs=100,
                lr=lr
            )

            # salva se migliore
            is_better = True
            if os.path.exists(BEST_SCORE_PATH):
                with open(BEST_SCORE_PATH, "r") as f:
                    try:
                        best_score_so_far = float(f.read())
                        is_better = best_f1_score > best_score_so_far
                    except:
                        is_better = True
            if is_better:
                torch.save(net.state_dict(), BEST_MODEL_PATH)
                with open(BEST_SCORE_PATH, "w") as f:
                    f.write(str(best_f1_score))
                logger.info(f"Nuovo miglior modello salvato in: {BEST_MODEL_PATH} con F1 score: {best_f1_score:.4f}")

            return best_f1_score
        
        study = optuna.create_study(direction='maximize',
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10))
        study.optimize(objective, n_trials=50)
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Highest F1-score: {study.best_value}")

        # salva best iperparametri
        best_hyperparameters = study.best_params
        best_hyperparameters['best_f1_score'] = study.best_value
        pd.DataFrame([best_hyperparameters]).to_csv(best_hyperparams_file, index=False)

        best_lr = float(study.best_params['lr'])
        best_batch_size = int(study.best_params['batch_size'])

    
    plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    n_classes=len(SELECTED_ACTIVITIES),
    X=X_train,     # (N,L,C)
    Y=Y_train_enc,     # interi 0..len(SELECTED_ACTIVITIES)-1
    batch_size=best_batch_size,
    figure_name=f"cm_train_best_wisdm_meanstd_aug_sampler_seed_{SEED}"
)

    plot_CM(
        mdl_class=DeepConvLSTM,
        mdl_weights=BEST_MODEL_PATH,
        n_classes=len(SELECTED_ACTIVITIES),
        X=X_test,
        Y=Y_test_enc,
        batch_size=best_batch_size,
        figure_name=f"cm_test_best_wisdm_meanstd_aug_sampler_seed_{SEED}"
    )


if __name__ == "__main__":
    main()



    


