import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, classification_report

# === CONFIGURAZIONE ===
BASE_DIR = r"D:\RISULTATI\EVENTS_RESULTS\BALL"     
MAJORITY_THR = 0.5
RESPECT_CONSEC = True


def build_segments(df_file, respect_consec=True):
    """Spezza finestre in segmenti consecutivi di stessa true_label."""
    rows = df_file.reset_index(drop=True)
    if rows.empty:
        return []

    segs = []
    s = 0
    for i in range(1, len(rows)):
        new_seg = (rows.loc[i, "true_label"] != rows.loc[i - 1, "true_label"])
        if not new_seg and respect_consec and "is_consecutive" in rows:
            if not rows.loc[i, "is_consecutive"]:
                new_seg = True
        if new_seg:
            segs.append((s, i - 1))
            s = i
    segs.append((s, len(rows) - 1))
    return segs


def analyze_predictions(pred_csv, npz_path, output_dir, majority_thr=0.5, respect_consec=True):
    os.makedirs(output_dir, exist_ok=True)

    # --- Leggi predizioni ---
    dfp = pd.read_csv(pred_csv)
    gid = dfp["global_window_id"].astype(int).values

    # --- Leggi NPZ ---
    data = np.load(npz_path, allow_pickle=True)
    original_files = data["original_files"].astype(str)
    consec = data["all_consecutivity"].astype(bool) if "all_consecutivity" in data else np.ones(len(original_files), dtype=bool)

    # --- Costruisci df combinato ---
    df = pd.DataFrame({
        "global_window_id": gid,
        "true_label": dfp["true_label"].values,
        "predicted_label": dfp["predicted_label"].values,
        "is_correct": dfp["correct"].astype(bool).values,
        "is_consecutive": consec[gid],
        "original_file": original_files[gid],
        "true_action": dfp.get("true_action", dfp["true_label"]).values,
        "predicted_action": dfp.get("predicted_action", dfp["predicted_label"]).values
    })

    # --- Segmenti ---
    seg_rows = []
    for of, g in df.groupby("original_file"):
        g = g.sort_values("global_window_id")
        segs = build_segments(g, respect_consec)
        for sid, (a, b) in enumerate(segs, start=1):
            chunk = g.iloc[a:b + 1]
            n = len(chunk)
            n_ok = chunk["is_correct"].sum()
            acc = n_ok / n if n > 0 else 0
            seg_rows.append({
                "original_file": of,
                "segment_id": sid,
                "true_action": chunk["true_action"].iloc[0],
                "n_windows": n,
                "n_correct": int(n_ok),
                "perc_correct": round(acc * 100, 2),
                "segment_correct": acc > majority_thr,
                "global_window_ids": chunk["global_window_id"].tolist(),
                "window_pred_actions": chunk["predicted_action"].tolist()
            })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(os.path.join(output_dir, "segments_summary.csv"), index=False)

    # --- Metriche ---
    y_true = seg_df["true_action"].values
    y_pred = []
    for _, row in seg_df.iterrows():
        if row["segment_correct"]:
            y_pred.append(row["true_action"])
        else:
            # fallback: maggioranza predetta nelle finestre di quel segmento
            maj = Counter(row["window_pred_actions"]).most_common(1)[0][0]
            y_pred.append(maj)

    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")

    metrics = {
        "n_segments": len(seg_df),
        "macro_f1": macro,
        "weighted_f1": weighted,
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{os.path.basename(pred_csv)}] Macro-F1={macro:.3f} Weighted-F1={weighted:.3f}")


if __name__ == "__main__":
    # loop sugli esperimenti (Tball_*)
    for exp in sorted(os.listdir(BASE_DIR)):
        exp_path = os.path.join(BASE_DIR, exp)
        if not os.path.isdir(exp_path):
            continue

        print(f"\n=== Esperimento: {exp} ===")
        # loop sui seed
        for seed_dir in sorted(os.listdir(exp_path)):
            seed_path = os.path.join(exp_path, seed_dir)
            if not os.path.isdir(seed_path) or not seed_dir.startswith("seed_"):
                continue

            print(f"--- Seed: {seed_dir} ---")
            # npz corrispondente
            seed_num = seed_dir.replace("seed_", "")
            npz_path = os.path.join(
                r"C:\codes\HumanActivityRecognition\HumanActivityRecognition\zupt_preprocessing_output\ball",
                seed_dir,
                f"ball_processed_windows_seed{seed_num}.npz"
            )
            if not os.path.exists(npz_path):
                print(f"[SKIP] NPZ non trovato: {npz_path}")
                continue

            csvs = sorted(glob.glob(os.path.join(seed_path, "predictions*.csv")))
            print(f"CSV trovati ({len(csvs)}): {csvs}")
            for csv in csvs:
                outdir = os.path.join(seed_path, "event_eval", os.path.splitext(os.path.basename(csv))[0])
                analyze_predictions(csv, npz_path, outdir, majority_thr=MAJORITY_THR, respect_consec=RESPECT_CONSEC)