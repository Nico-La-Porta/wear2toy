#!/usr/bin/env python3
from pathlib import Path
import re
import argparse
import numpy as np
import csv
from typing import Dict, List, Tuple

# pattern: f1-score__test_<EXPERIMENTO>_seed_<SEED>[.txt]
FNAME_RE = re.compile(r"^f1-score__test_(?P<exp>.+?)_seed_(?P<seed>\d+)(?:\.[^.]+)?$")

# estrae Macro/Micro/Weighted dal contenuto del file
MACRO_RE   = re.compile(r"F1-Score\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)")
MICRO_RE   = re.compile(r"F1-Score\s*\(Micro\)\s*:\s*([0-9]*\.?[0-9]+)")
WEIGHT_RE  = re.compile(r"F1-Score\s*\(Weighted\)\s*:\s*([0-9]*\.?[0-9]+)")

def parse_metrics(text: str) -> Tuple[float, float, float]:
    def grab(rx):
        m = rx.search(text)
        if not m:
            raise ValueError(f"Campo non trovato: {rx.pattern}")
        return float(m.group(1))
    return grab(MACRO_RE), grab(MICRO_RE), grab(WEIGHT_RE)

def main():
    ap = argparse.ArgumentParser(description="Raggruppa file f1-score per esperimento e calcola media/std.")
    ap.add_argument("--dir", type=Path, default=Path("."), help="Cartella con i file (default: .)")
    ap.add_argument("--prefix", type=str, default="f1-score__test_", help="Prefisso dei file (default: f1-score__test_)")
    ap.add_argument("--csv", type=Path, default=None, help="(Opzionale) salva risultati in CSV a questo percorso")
    args = ap.parse_args()

    if not args.dir.exists() or not args.dir.is_dir():
        raise SystemExit(f"Directory non valida: {args.dir}")

    # individua i candidati e filtra col regex per avere exp/seed puliti
    files = []
    for p in args.dir.iterdir():
        if not p.is_file():
            continue
        if not p.name.startswith(args.prefix):
            continue
        m = FNAME_RE.match(p.name)
        if m:
            files.append((p, m.group("exp"), int(m.group("seed"))))

    if not files:
        print("Nessun file trovato con il formato atteso.")
        return

    print("File trovati:")
    for p, exp, seed in sorted(files, key=lambda x: (x[1], x[2], x[0].name)):
        print(f"  - {p.name}")

    # raggruppa per esperimento
    groups: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for p, exp, seed in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            macro, micro, weighted = parse_metrics(text)
            groups.setdefault(exp, []).append((seed, macro, micro, weighted))
        except Exception as e:
            print(f"[ATTENZIONE] Salto {p.name}: {e}")

    esperimenti = sorted(groups.keys())
    print("\nNumero di esperimenti (tipi diversi):", len(esperimenti))

    # stampa riepilogo per esperimento
    rows_for_csv = [("experiment", "num_seeds",
                     "macro_mean", "macro_std",
                     "micro_mean", "micro_std",
                     "weighted_mean", "weighted_std",
                     "seeds")]
    print("\nRisultati per esperimento (media ± std):")
    for exp in esperimenti:
        seeds, macros, micros, weights = [], [], [], []
        for seed, macro, micro, weighted in groups[exp]:
            seeds.append(seed); macros.append(macro); micros.append(micro); weights.append(weighted)
        macros = np.array(macros, dtype=float)
        micros = np.array(micros, dtype=float)
        weights = np.array(weights, dtype=float)

        macro_mean, macro_std = float(np.mean(macros)), float(np.std(macros))
        micro_mean, micro_std = float(np.mean(micros)), float(np.std(micros))
        weight_mean, weight_std = float(np.mean(weights)), float(np.std(weights))

        seeds_str = ",".join(map(str, sorted(seeds)))
        print(f"\n[{exp}]")
        print(f"  seeds ({len(seeds)}): {seeds_str}")
        print(f"  Macro    : {macro_mean:.6f} ± {macro_std:.6f}")
        print(f"  Micro    : {micro_mean:.6f} ± {micro_std:.6f}")
        print(f"  Weighted : {weight_mean:.6f} ± {weight_std:.6f}")

        rows_for_csv.append((
            exp, len(seeds),
            f"{macro_mean:.6f}", f"{macro_std:.6f}",
            f"{micro_mean:.6f}", f"{micro_std:.6f}",
            f"{weight_mean:.6f}", f"{weight_std:.6f}",
            seeds_str
        ))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows_for_csv)
        print(f"\nCSV salvato in: {args.csv}")

if __name__ == "__main__":
    main()
