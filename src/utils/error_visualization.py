import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
import glob
import shutil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import src.utils.sliding_window_on_data as sliding_window_on_data


def plot_sensor_channels_with_colored_line_and_windows(file_path, signal_cols, max_rows=1000, save_dir=None):
    """
    Plotta i 9 canali sensoriali in subplot, colorando la linea in blu (corretto) o rosso (sbagliato).
    Mostra anche la finestra di appartenenza come heatmap sotto ogni segnale.
    """
    

    df = pd.read_csv(file_path)
    df = df.head(max_rows) #Limito il dataframe ai primi 1000 campioni

    #converto le colonne window_ids e correct_windows da stringa a lista
    df["window_ids"] = df["window_ids"].apply(ast.literal_eval) 
    df["correct_windows"] = df["correct_windows"].apply(ast.literal_eval)

    # Determino correttezza per ogni riga
    correct_per_row = []
    window_id_per_row = []
    #Per ogni riga 
    for idx, row in df.iterrows():
        #Prendo il primo ID Finestra se esiste , Determino la correttezza: 
        # True se tutte le finestre sono corrette, False se almeno una è sbagliata, None se non ci sono dati
        window_id_per_row.append(row["window_ids"][0] if row["window_ids"] else None)
        if not row["correct_windows"]:
            correct_per_row.append(None)
        elif all(x is True for x in row["correct_windows"]):
            correct_per_row.append(True)
        elif any(x is False for x in row["correct_windows"]):
            correct_per_row.append(False)
        else:
            correct_per_row.append(None)

    #Creo subplot per ogni canale sesnoriale
    n_cols = len(signal_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 2.2 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    #per ogni canale: estraggo i dati, per ogni segmento tra due punti coloro la linea in base alla correttezza, imposto titolo ed etichetta
    for i, col in enumerate(signal_cols):
        y = df[col].values
        x = np.arange(len(y))
        for j in range(1, len(y)):
            if correct_per_row[j] is True:
                color = 'blue'
            elif correct_per_row[j] is False:
                color = 'red'
            else:
                color = 'gray'
            axes[i].plot([x[j-1], x[j]], [y[j-1], y[j]], color=color, linewidth=1)
        axes[i].set_ylabel(col)
        axes[i].set_title(col)

        # Creo un asse gemello per la heatmap degli ID finestra sotto il segnale.
        ax2 = axes[i].twinx()
        # Sostituisco None con -1 per la heatmap
        window_id_numeric = [wid if wid is not None else -1 for wid in window_id_per_row]
        ax2.imshow(
            np.array(window_id_numeric)[None, :],
            aspect="auto",
            cmap="viridis",
            alpha=0.3,
            extent=[0, len(window_id_numeric), np.min(y), np.max(y)]
        )
        ax2.set_yticks([])
        ax2.set_ylabel("window_id", fontsize=8, color="gray")
        ax2.set_ylim(np.min(y), np.max(y))

    axes[-1].set_xlabel("Indice riga")

    # Legenda
    legend_patches = [
        mpatches.Patch(color='blue', label='Corretto'),
        mpatches.Patch(color='red', label='Sbagliato'),
        mpatches.Patch(color='gray', label='Nessuna finestra')
    ]
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.13, n_cols))
    plt.tight_layout()
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_channels.png")
        plt.savefig(save_path)
        print(f"Figura salvata in: {save_path}")
    #plt.show()

def plot_all_enriched_actions(signal_cols, toy_name,max_rows=1000,save_dir=None):
    folder = rf'C:\codes\HumanActivityRecognition\data\downstream_data\enriched_actions_{toy_name}'
    files = glob.glob(os.path.join(folder, f'df_{toy_name}_action_*.csv'))
    print(f"Trovati {len(files)} file in {folder}.")
    for file_path in files:
        print(f"Visualizzo: {file_path}")
        plot_sensor_channels_with_colored_line_and_windows(
            file_path, signal_cols, max_rows=max_rows, save_dir=save_dir
        )


def plot_activity_timeline_broken_axis(file_path, signal_col, pause_threshold=200, save_dir=None):
    """
    Plotta la timeline di un segnale spezzando l'asse X
    per tagliare via le pause lunghe, mantenendo la scala temporale originale.
    """
    print(f"\n--- Generazione timeline con asse spezzato per '{signal_col}' ---")
    df = pd.read_csv(file_path)
    activity_col = 'action'
    
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except:
            return []

    df["correct_windows"] = df["correct_windows"].apply(safe_eval) # Converte le stringhe in liste


    # costruzione etichette stato (correct, wrong, pause)
    correct_per_row = []
    for _, row in df.iterrows():
        correct = row.get("correct_windows", [])
        status = "pause"
        if correct:
            status = "correct" if correct[0] is True else "wrong"
        correct_per_row.append(status)

    # Costruisce la lista dei segmenti consecutivi con lo stesso stato (correct/wrong/pause)
    timeline = []
    current_status = correct_per_row[0]
    start = 0
    for i, status in enumerate(correct_per_row[1:], 1):
        if status != current_status:
            timeline.append((start, i - 1, current_status))
            start = i
            current_status = status
    timeline.append((start, len(correct_per_row) - 1, current_status))
    
    ## individua le pause lunghe
    long_pauses = [(s, e, st) for s, e, st in timeline if st == "pause" and (e - s) >= pause_threshold]

    # se non ci sono pause lunghe → grafico unico
    if not long_pauses:
        print("Nessuna pausa abbastanza lunga per spezzare l'asse. Genero un grafico unico.")
        fig, ax = plt.subplots(figsize=(20, 6))
        axes = [ax]
        intervals = [(0, len(df)-1)]
    else:
        # costruisci gli intervalli "buoni" da plottare
        intervals = []
        last_end = 0
        for (s, e, _) in long_pauses:
            if last_end < s:
                intervals.append((last_end, s))  # prima della pausa
            last_end = e
        if last_end < len(df)-1:
            intervals.append((last_end, len(df)-1))  # dopo ultima pausa

        # crea tanti subplot quanti sono gli intervalli
        fig, axes = plt.subplots(1, len(intervals), figsize=(20, 6), sharey=True,
                                 gridspec_kw={'wspace': 0.05})

        if len(intervals) == 1:
            axes = [axes]  # nel caso matplotlib restituisca un singolo Axes

        # aggiungi le barrette di rottura
        d = .015
        for i in range(len(axes)-1):
            kwargs = dict(transform=axes[i].transAxes, color='k', clip_on=False)
            axes[i].plot((1-d, 1+d), (-d, +d), **kwargs)
            axes[i].plot((1-d, 1+d), (1-d, 1+d), **kwargs)

            kwargs.update(transform=axes[i+1].transAxes)
            axes[i+1].plot((-d, +d), (1-d, 1+d), **kwargs)
            axes[i+1].plot((-d, +d), (-d, +d), **kwargs)

    colors = {'correct': '#2E86AB', 'wrong': '#F24236', 'pause': '#CCCCCC'}
    
    # plot effettivo
    for ax, (x_start, x_end) in zip(axes, intervals):
        for start, end, status in timeline:
            if end < x_start or start > x_end:
                continue  # fuori intervallo
            seg_start = max(start, x_start)
            seg_end = min(end, x_end)
            segment_df = df.iloc[seg_start:seg_end+1]
            ax.plot(segment_df.index, segment_df[signal_col], color=colors[status], linewidth=2)

            if status != 'pause':
                activity = df.loc[seg_start, activity_col]
                mid_point = seg_start + (seg_end - seg_start) // 2
                ax.text(mid_point, ax.get_ylim()[1], f'{activity}', ha='center', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax.set_xlim(x_start, x_end)
        ax.grid(True, alpha=0.3)

    # titoli, labels e legenda
    fig.suptitle(f'{signal_col} - Timeline Activities', fontsize=16)
    axes[0].set_ylabel(signal_col, fontsize=12)
    fig.text(0.5, 0.04, 'Row Indices', ha='center', va='center', fontsize=12)

    legend_elements = [
        mpatches.Patch(color=colors['correct'], label='Corretto'),
        mpatches.Patch(color=colors['wrong'], label='Sbagliato'),
        mpatches.Patch(color=colors['pause'], label='Pausa ')
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_{signal_col}_timeline.png")
        plt.savefig(save_path)
        print(f"Figura salvata in: {save_path}")

    #plt.show()


def radar_activity_duration(file_path, activity_col='action', save_dir=None):
    """
    Radar Plot — Durata totale per attività: Mostra quanto il bambino ha fatto ciascuna attività in termini di numero di campioni.
    """
    print("\n--- Generazione radar della durata delle attività ---")
    df = pd.read_csv(file_path)
    df[activity_col] = df[activity_col].fillna("Pausa").replace('', "Pausa")

    counts = df[activity_col].value_counts().sort_index()
    if "Pausa" in counts.index:
        counts = counts.drop("Pausa")
    
    if counts.empty:
        print("Nessuna attività trovata nel file per il radar plot.")
        return

    labels = counts.index.tolist()
    values = counts.values.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='darkorange')
    ax.fill(angles, values, color='orange', alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_title("Durata Totale per Attivita' (in campioni)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_radar_duration_activity.png")
        plt.savefig(save_path)
        print(f"Figura salvata in: {save_path}")
    #plt.show()


def radar_activity_sessions(file_path, activity_col='action', save_dir=None):
    """
    Radar Plot — Numero di sessioni (cambi attività): Conta quante transizioni ci sono state per ciascuna attività.
    Utile per sapere quante volte un’attività è iniziata (indipendentemente dalla durata).
    """
    print("\n--- Generazione radar del numero di sessioni per attività ---")
    df = pd.read_csv(file_path)
    df[activity_col] = df[activity_col].fillna("Pausa").replace('', "Pausa")

    # Trova le transizioni: ogni volta che cambia attività
    activity_series = df[activity_col].values
    sessions = []
    last = None
    for act in activity_series:
        if act != last and act != "Pausa":
            sessions.append(act)
        last = act

    # Conta quante sessioni per ciascuna attività
    session_counts = Counter(sessions)
    if not session_counts:
        print("Nessuna sessione trovata per il radar plot.")
        return

    labels = list(session_counts.keys())
    values = list(session_counts.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='darkgreen')
    ax.fill(angles, values, color='limegreen', alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_title("Numero di Sessioni per Attività (transizioni)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_radar_sessions_activity.png")
        plt.savefig(save_path)
        print(f"Figura salvata in: {save_path}")
    #plt.show()

def barplot_activity_mean_duration_and_kids(file_path, activity_col='action', kid_col='kid_id', save_dir=None):
    """
    Bar Plot — Durata media dell'attività e numero di bambini che l'hanno svolta.
    Per ogni attività mostra la durata media (in campioni) e quanti bambini l'hanno svolta.
    """
    print("\n--- Generazione barplot durata media attività e numero bambini ---")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[activity_col, kid_col])
    df[activity_col] = df[activity_col].replace('', 'Pausa').fillna('Pausa')

    # Raggruppa per attività e bambino, conta i campioni per ogni (attività, bambino)
    group = df.groupby([activity_col, kid_col]).size().reset_index(name='duration')
    # Calcola durata media per attività
    mean_duration = group.groupby(activity_col)['duration'].mean()
    # Conta quanti bambini hanno fatto ciascuna attività
    n_kids = group.groupby(activity_col)[kid_col].nunique()

    # Ordina per durata media decrescente
    activities = mean_duration.sort_values(ascending=False).index.tolist()
    mean_duration = mean_duration[activities]
    n_kids = n_kids[activities]

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.bar(activities, mean_duration, color='skyblue', label='Durata media (campioni)')
    ax2.plot(activities, n_kids, color='orange', marker='o', label='Numero bambini')

    ax1.set_ylabel('Durata media attività (campioni)', color='skyblue', fontsize=12)
    ax2.set_ylabel('Numero bambini', color='orange', fontsize=12)
    ax1.set_xticklabels(activities, rotation=45, ha='right')
    ax1.set_title("Durata media attività e numero bambini che l'hanno svolta", fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_barplot_mean_duration_nkids.png")
        plt.savefig(save_path)
        print(f"Figura salvata in: {save_path}")
    #plt.show()

def stats_activity_duration_and_kids(file_path, activity_col='action', kid_col='kid_id', save_dir=None):
    """
    Calcola media, std della durata delle attività e numero di bambini che l'hanno svolta.
    Salva una tabella CSV e stampa a schermo.
    """
    print("\n--- Statistiche durata attività e numero bambini ---")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[activity_col, kid_col])
    df[activity_col] = df[activity_col].replace('', 'Pausa').fillna('Pausa')

    # Raggruppa per attività e bambino, conta i campioni per ogni (attività, bambino)
    group = df.groupby([activity_col, kid_col]).size().reset_index(name='duration')
    # Calcola media e std per attività
    mean_duration = group.groupby(activity_col)['duration'].mean()
    std_duration = group.groupby(activity_col)['duration'].std()
    n_kids = group.groupby(activity_col)[kid_col].nunique()

    stats_df = pd.DataFrame({
        'mean_duration': mean_duration,
        'std_duration': std_duration,
        'n_kids': n_kids
    }).sort_values('mean_duration', ascending=False)

    print(stats_df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_stats_duration_nkids.csv")
        stats_df.to_csv(save_path)
        print(f"Tabella salvata in: {save_path}")




def choose_experiment_interactively(base_reports_dir):
    """Scansiona e permette di scegliere un esperimento in modo interattivo."""
    print("="*50 + "\nRicerca degli esperimenti disponibili...\n" + "="*50)
    try:
        # Scansiona le cartelle degli esperimenti
        all_exp_dirs = sorted([d for d in os.listdir(base_reports_dir) if os.path.isdir(os.path.join(base_reports_dir, d))])
    except FileNotFoundError:
        print(f"ERRORE: Cartella base non trovata: {base_reports_dir}")
        return None, None
    if not all_exp_dirs:
        print("Nessuna cartella di esperimento trovata.")
        return None, None

    print("Scegli un esperimento da analizzare:")
    for i, dir_name in enumerate(all_exp_dirs):
        print(f"  [{i+1}] {dir_name}")

    while True:
        try:
            choice_str = input(f"\nInserisci il numero (1-{len(all_exp_dirs)}): ")
            choice_int = int(choice_str)
            # Controlla se la scelta è valida
            if 1 <= choice_int <= len(all_exp_dirs):
                # Seleziona la cartella dell'esperimento
                selected_dir_name = all_exp_dirs[choice_int - 1]
                # Estrai il nome del giocattolo
                toy_name = selected_dir_name.split('_')[0][1:].lower()
                return os.path.join(base_reports_dir, selected_dir_name), toy_name
            else:
                print("Numero non valido.")
        except ValueError:
            print("Input non valido.")
        except (KeyboardInterrupt, EOFError):
            print("\nAnalisi annullata.")
            return None, None


def load_prediction_results(experiment_dir):
    """Carica i risultati da tutti i file di predizione"""
    window_correct_map = {}
    
    # Cerca file che iniziano con "predictions_TEST" 
    prediction_files = glob.glob(os.path.join(experiment_dir, "predictions_TEST*.csv"))
    
    if not prediction_files: 
        print(f"ATTENZIONE: Nessun file predictions_TEST*.csv trovato in {experiment_dir}")
        
        
    print(f"\n 1. Caricamento risultati da {len(prediction_files)} file di predizione...")
    
    for file in prediction_files:
        print(f"  -> Leggendo: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file)


            for _, row in df.iterrows():
                try:
                    # Controlla se c'è la colonna window_index 
                    if 'window_index' in df.columns:
                        window_info = ast.literal_eval(row["window_index"])
                        global_id = window_info['global_window_id']
                    else:
                        # Se c'è global_window_id diretto
                        global_id = row["global_window_id"]

                    # Mappa l'ID della finestra globale con il valore corretto
                    window_correct_map[global_id] = row["correct"]
                except Exception as e:
                    print(f"    Errore alla riga {row.name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ERRORE nel leggere {file}: {e}")
            continue
    
    print(f"-> Caricate informazioni per {len(window_correct_map)} finestre uniche.")
    return window_correct_map
    

def create_window_mappings(data_dir, toy_name):
    """
    Replica la logica di sliding window per costruire la mappatura fondamentale.
    """
    print(f"\n 2.Ricostruzione mappatura finestre per '{toy_name}'...")
    
    toy_prefix_map = {'ball': 'BA', 'car': 'C', 'doll': 'DO', 'spoon': 'SP', 'elephant': 'E'}
    toy_prefix = toy_prefix_map.get(toy_name, toy_name.upper())
    
    #Costriusco il percorso da trovare
    non_null_file = os.path.join(data_dir, f'df_{toy_prefix}_non_null.csv')
    if not os.path.exists(non_null_file):
        print(f"ERRORE: File {non_null_file} non trovato!")
        return {}, None
    
    df_non_null = pd.read_csv(non_null_file)
    print(f"  -> Caricato df con {len(df_non_null)} righe")

    temp_action_dir = os.path.join(data_dir, "temp_actions_for_analysis")
    os.makedirs(temp_action_dir, exist_ok=True)
    
    # Crea file temporanei per ogni azione
    for action_id in df_non_null['action_id'].unique():
        df_action = df_non_null[df_non_null["action_id"] == action_id]
        # Usa il nome del toy coerente
        df_action.to_csv(os.path.join(temp_action_dir, f'df_{toy_name}_action_{int(action_id)}.csv'), index=False)
        
    all_window_indices = []
    global_window_id = 0
    action_files = sorted(glob.glob(os.path.join(temp_action_dir, f'df_{toy_name}_action_*.csv')))

    print(f"  -> Elaborando {len(action_files)} file di azione...")
    for file_path in action_files:
        action_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
        try:
            _, _, _, _, window_row_indices = sliding_window_on_data.process_csv(file_path, 9, 100, 50)
            for i in range(len(window_row_indices)):
                all_window_indices.append({
                    'global_window_id': global_window_id, 
                    'action_id': action_id, 
                    'row_indices': window_row_indices[i].tolist()
                })
                global_window_id += 1
        except Exception as e:
            print(f"    Errore nel processare {file_path}: {e}")
            continue
            
    # Pulisci i file temporanei
    shutil.rmtree(temp_action_dir)
    
    # Crea la mappatura riga -> finestre
    row_to_window_map = defaultdict(list)
    for window_info in all_window_indices:
        action_id = window_info['action_id']
        file_name = f'df_{toy_name}_action_{action_id}.csv'
        for row_idx in window_info['row_indices']:
            if row_idx != -1:
                row_to_window_map[(file_name, row_idx)].append(window_info['global_window_id'])
    
    print(f"-> Mappatura creata. Trovate {len(all_window_indices)} finestre in totale.")
    return row_to_window_map, df_non_null

def enrich_action_files(data_dir, toy_name, df_non_null, row_to_window_map, window_correct_map):
    """
    Arricchisce i file per azione con window_ids e correct_windows.
    """
    print(f"\nArricchimento dei file divisi per azione...")
    output_dir = os.path.join(data_dir, f"enriched_actions_{toy_name}")
    os.makedirs(output_dir, exist_ok=True)

    for action_id in df_non_null['action_id'].unique():
        action_file_name = f'df_{toy_name}_action_{int(action_id)}.csv'
        df_action = df_non_null[df_non_null['action_id'] == action_id].copy()
        df_action.reset_index(drop=True, inplace=True)

        # Aggiungi window_ids per ogni riga
        df_action["window_ids"] = [row_to_window_map.get((action_file_name, i), []) for i in range(len(df_action))]
        
        # Aggiungi correct_windows basato sui window_ids
        df_action["correct_windows"] = df_action["window_ids"].apply(
            lambda ids: [window_correct_map.get(wid, None) for wid in ids]
        )
        
        # Aggiungi colonne di sintesi
        df_action["correct_summary"] = df_action["correct_windows"].apply(
            lambda c_list: "no_windows" if not c_list else (
                "all_correct" if all(c is True for c in c_list) else (
                    "some_incorrect" if any(c is False for c in c_list) else "unknown"
                )
            )
        )
        df_action["any_incorrect"] = df_action["correct_windows"].apply(
            lambda lst: any(x is False for x in lst)
        )

        output_path = os.path.join(output_dir, action_file_name)
        df_action.to_csv(output_path, index=False)
        print(f"  -> Salvato: {action_file_name}")
        
    print(f"-> Creati file arricchiti in: {output_dir}")
    return output_dir


def create_final_enriched_files(data_dir, toy_name, enriched_actions_dir):
    """
    Esegue i merge finali per creare i file KID_ID_with_correctness.csv.
    """
    print(f"\n Creazione dei file finali con i risultati del merge...")
    
    toy_prefix_map = {'ball': 'BA', 'car': 'C', 'doll': 'DO', 'spoon': 'SP', 'elephant': 'E'}
    toy_prefix = toy_prefix_map.get(toy_name, toy_name.upper())

    # --- Primo Merge: Crea df_{toy_name}_non_null_with_correctness.csv ---
    enriched_action_files = glob.glob(os.path.join(enriched_actions_dir, "*.csv"))
    if not enriched_action_files:
        print(f"ERRORE: Nessun file trovato in {enriched_actions_dir}")
        return []
        
    df_correct_all = pd.concat([pd.read_csv(f) for f in enriched_action_files], ignore_index=True)
    df_non_null = pd.read_csv(os.path.join(data_dir, f'df_{toy_prefix}_non_null.csv'))

    # Colonne per il merge (escludi 'action' per evitare duplicati)
    key_cols = [col for col in df_non_null.columns if col not in ['action']]
    extra_cols = ['window_ids', 'correct_windows', 'correct_summary', 'any_incorrect']
    
    #USO COME CHIAVE TUTTE LE COLONNE TRANNE ACTION
    df_non_null_enriched = pd.merge(
        df_non_null, 
        df_correct_all[key_cols + extra_cols], 
        on=key_cols, 
        how='left'
    )
    
    # Salva il risultato
    non_null_enriched_path = os.path.join(data_dir, f'df_{toy_name}_non_null_with_correctness.csv')
    df_non_null_enriched.to_csv(non_null_enriched_path, index=False)
    print(f"-> Creato file aggregato: {os.path.basename(non_null_enriched_path)}")

    # --- Secondo Merge: Arricchisci i file dei singoli bambini ---
    raw_kid_files = glob.glob(os.path.join(data_dir, f"*_{toy_prefix}.csv"))
    output_dir = os.path.join(data_dir, f"visualizations_{toy_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    final_file_paths = []
    print(f"  -> Processando {len(raw_kid_files)} file bambini...")
    #Prende il file originali die bambini, per ogni bambino, leggo il suo file grezzo, estraggo solo le righe del merge globale (fatto prima) che riguardano quel kid_id
    for kid_file_path in raw_kid_files:
        kid_id = int(os.path.basename(kid_file_path).split('_')[0])
        df_kid_raw = pd.read_csv(kid_file_path)
        
        # Filtra per il bambino corrente
        df_correct_subset = df_non_null_enriched[df_non_null_enriched['kid_id'] == kid_id]
        
        if df_correct_subset.empty:
            print(f"    ATTENZIONE: Nessun dato per kid_id {kid_id}")
            continue
        
        #merge finale sempre sulla chiave (tutte le colonne tranne action)
        df_final_kid = pd.merge(
            df_kid_raw,
            df_correct_subset[key_cols + extra_cols],
            on=key_cols,
            how='left'
        )
        
        # Gestisci i NaN nelle colonne lista
        def safe_eval_list(x):
            if pd.isna(x) or x == '' or x == 'nan':
                return []
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return []
            return x if isinstance(x, list) else []
        
        df_final_kid['window_ids'] = df_final_kid['window_ids'].apply(safe_eval_list)
        df_final_kid['correct_windows'] = df_final_kid['correct_windows'].apply(safe_eval_list)
        
        output_path = os.path.join(output_dir, os.path.basename(kid_file_path).replace('.csv', '_with_correctness.csv'))
        df_final_kid.to_csv(output_path, index=False)
        final_file_paths.append(output_path)
        print(f"    -> Salvato: {os.path.basename(output_path)}")
        
    print(f"-> Creati {len(final_file_paths)} file per visualizzazione in: {output_dir}")
    return final_file_paths

def main():
    """Funzione principale che orchestra l'intera pipeline di post-analisi."""
    BASE_REPORTS_DIR = r"C:\codes\HumanActivityRecognition\reports"
    BASE_DATA_DIR = r"C:\codes\HumanActivityRecognition\data\downstream_data"
    BASE_FIGURES_DIR = r"C:\codes\HumanActivityRecognition\reports\figures"
    
    # 1.Scegli l'esperimento interattivamente
    experiment_dir, toy_name = choose_experiment_interactively(BASE_REPORTS_DIR)
    if not experiment_dir: 
        return

    experiment_name = os.path.basename(experiment_dir)
    figures_dir = os.path.join(BASE_FIGURES_DIR, experiment_name)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Esperimento selezionato: {experiment_dir}")
    print(f"Giocattolo: {toy_name}")

    # 2. Carica i risultati delle predizioni
    window_correct_map = load_prediction_results(experiment_dir)
    if not window_correct_map: 
        print("ERRORE: Impossibile caricare le predizioni.")
        return
    
    # 3. Ricostruisci la mappatura delle finestre
    row_to_window_map, df_non_null = create_window_mappings(BASE_DATA_DIR, toy_name)
    if not row_to_window_map or df_non_null is None:
        print("ERRORE: Impossibile creare la mappatura delle finestre.")
        return

    # 4. Arricchisci i file per azione
    enriched_actions_dir = enrich_action_files(BASE_DATA_DIR, toy_name, df_non_null, row_to_window_map, window_correct_map)

    # 5. Arricchisci i file dei singoli bambini
    final_files_to_plot = create_final_enriched_files(BASE_DATA_DIR, toy_name, enriched_actions_dir)
    
    if not final_files_to_plot:
        print("ERRORE: Nessun file finale creato per la visualizzazione.")
        return
    
    # 6. Lancia le visualizzazioni
    print(f"\n Avvio della visualizzazione per {len(final_files_to_plot)} file...")
    signal_cols = ["Accel_LN_X", "Accel_LN_Y", "Accel_LN_Z"]

    plot_all_enriched_actions(signal_cols, toy_name, save_dir=figures_dir)

    for file_path in final_files_to_plot:
        print(f"\n{'='*20} ANALISI DI: {os.path.basename(file_path)} {'='*20}")
        
        # Verifica che il file abbia dati
        try:
            df_check = pd.read_csv(file_path)
            if df_check.empty:
                print(f"  File vuoto, saltato.")
                continue
                
            for signal_col in signal_cols:
                plot_activity_timeline_broken_axis(file_path, signal_col, pause_threshold=200, save_dir=figures_dir)
            
            radar_activity_duration(file_path, activity_col="action")
            radar_activity_sessions(file_path, activity_col="action", save_dir=figures_dir)
            barplot_activity_mean_duration_and_kids(file_path, activity_col="action", kid_col="kid_id", save_dir=figures_dir)
            stats_activity_duration_and_kids(file_path, activity_col="action", kid_col="kid_id", save_dir=figures_dir)
            
        except Exception as e:
            print(f"  ERRORE nella visualizzazione di {file_path}: {e}")
            continue


if __name__ == "__main__":
    main()