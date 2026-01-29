import os
import json
import numpy as np
from typing import Optional, Dict, Any

DATA_DIR = r"C:\codes\HumanActivityRecognition\data\WISDM_100Hz"  # sorgente
OUT_DIR  = os.path.join(DATA_DIR, "json_export")                  # destinazione
os.makedirs(OUT_DIR, exist_ok=True)

def load_npz_safe(path: str) -> Dict[str, Any]:
    """
    Carica un .npz senza dipendere dal pickle (tranne quando 'activities' è un oggetto dict salvato).
    Restituisce un dizionario Python.
    """
    npz = np.load(path, allow_pickle=True)  # allow_pickle serve solo per 'activities' quando è numpy.object
    out = {}
    for k in npz.files:
        v = npz[k]
        # Normalizza 'activities' eventualmente come dict python
        if k == "activities":
            if isinstance(v, np.ndarray) and v.dtype == object:
                v = v.item()
        out[k] = v
    return out

def validate_pair(sig: Dict[str, Any], ann: Optional[Dict[str, Any]]) -> None:
    """Controlli minimi di coerenza; solleva ValueError se qualcosa non torna."""
    if "data" not in sig or "fs" not in sig or "chs" not in sig:
        raise ValueError("File _sig.npz privo di una o più chiavi richieste: 'data', 'fs', 'chs'.")

    data = sig["data"]
    if data.ndim != 2:
        raise ValueError(f"'data' deve essere 2D (T, C). Trovato: {data.shape}")

    if ann is not None:
        activities = ann.get("activities", None)
        if not isinstance(activities, dict) or "labels" not in activities or "idx" not in activities:
            raise ValueError("File _ann.npz privo di 'activities' con chiavi 'labels' e 'idx'.")
        labels = list(activities["labels"])
        idx    = list(activities["idx"])
        if len(labels) != len(idx) - 1:
            raise ValueError(f"Incoerenza labels/idx: len(labels)={len(labels)} vs len(idx)={len(idx)} (atteso: idx = labels+1).")

        # L'ultimo indice deve combaciare con T (numero di righe di data)
        T = data.shape[0]
        if idx[-1] != T:
            raise ValueError(f"L'ultimo indice in 'idx' ({idx[-1]}) non combacia con la lunghezza dei dati ({T}).")

def convert_one(base_name: str, data_dir: str, out_dir: str) -> None:
    """
    Converte una coppia base_name_{sig,ann}.npz in:
      - <base_name>.npy (solo data)
      - <base_name>.json (metadati + riferimenti)
    """
    sig_path = os.path.join(data_dir, base_name + "_sig.npz")
    ann_path = os.path.join(data_dir, base_name + "_ann.npz")

    if not os.path.exists(sig_path):
        print(f"[SKIP] Mancante: {sig_path}")
        return

    sig = load_npz_safe(sig_path)
    ann = load_npz_safe(ann_path) if os.path.exists(ann_path) else None

    # Validazione
    try:
        validate_pair(sig, ann)
    except Exception as e:
        print(f"[ERRORE] {base_name}: {e}")
        return

    data: np.ndarray = sig["data"]
    chs  = [str(c) for c in np.array(sig["chs"]).ravel().tolist()]
    fs   = float(np.array(sig["fs"]).item() if hasattr(sig["fs"], "item") else sig["fs"])

    # Percorsi out
    npy_path   = os.path.join(out_dir, base_name + ".npy")
    json_path  = os.path.join(out_dir, base_name + ".json")

    # Salva array grandi in .npy (portabile tra versioni numpy)
    np.save(npy_path, data.astype(np.float32), allow_pickle=False)

    # Prepara metadati JSON
    meta = {
        "trace_id": base_name,
        "signals": {
            "data_path": os.path.basename(npy_path),
            "shape": list(data.shape),      # [T, C]
            "dtype": "float32",
            "chs": chs,                     # es. ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
            "fs": fs                        # 100.0
        },
        "annotations": None
    }

    if ann is not None:
        activities = ann["activities"]
        labels = list(activities["labels"])
        idx    = [int(i) for i in activities["idx"]]   # int puri per JSON pulito
        meta["annotations"] = {
            "labels": labels,
            "idx": idx
        }

    # Salva JSON (leggibile)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Log di verifica
    print(f"[OK] {base_name}")
    print(f"     data: {data.shape} -> {os.path.basename(npy_path)}")
    print(f"     fs:   {fs} Hz | chs: {chs}")
    if meta["annotations"]:
        L = len(meta["annotations"]["labels"])
        I = len(meta["annotations"]["idx"])
        print(f"     ann:  labels={L}, idx={I} (ultimo idx={meta['annotations']['idx'][-1]})")
    else:
        print("     ann:  (nessuna annotazione)")

def batch_convert(data_dir: str, out_dir: str) -> None:
    files = [f for f in os.listdir(data_dir) if f.endswith("_sig.npz")]
    files.sort()
    print(f"[INFO] Trovati {len(files)} segnali in {data_dir}")
    for sig_file in files:
        base = sig_file.replace("_sig.npz", "")
        convert_one(base, data_dir, out_dir)
    print(f"[FINITO] JSON+NPY in: {out_dir}")

if __name__ == "__main__":
    batch_convert(DATA_DIR, OUT_DIR)
    # sotto-cartelle split
    for split in ["train", "test"]:
        in_dir  = os.path.join(DATA_DIR, split)
        out_dir = os.path.join(DATA_DIR, f"json_export_{split}")
        os.makedirs(out_dir, exist_ok=True)
        batch_convert(in_dir, out_dir)