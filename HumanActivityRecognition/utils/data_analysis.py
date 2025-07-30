import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


def plot_sensor_channels_with_colored_line_and_windows(file_path, signal_cols, max_rows=1000):
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
        #Prendo il primo ID Finestra se esiste , Determino la correttezza: T
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
    plt.show()



def plot_clean_activity_timeline_compress(
    file_path,
    signal_cols,
    min_activity_length=100,
    pause_compression_factor=10,
    debug=True
):


    # Carico il csv e stympo il numero di righe
    df = pd.read_csv(file_path)
    print(f"Dataset caricato: {len(df)} righe")

    # Cerco la colonna che rappresenta l'attività
    activity_col = next((col for col in ['activity', 'Activity', 'label', 'Label', 'class', 'Class', 'action', 'Action'] if col in df.columns), None)
    if activity_col:
        print(f"Colonna attività trovata: {activity_col}")
    else:
        print("Nessuna colonna attività trovata")
    
    #Creo una funzione di utilità per convertire stringhe in liste 
    def safe_eval(x):
        if pd.isna(x) or x in ['', '[]']:
            return []
        try:
            return ast.literal_eval(x.strip()) if isinstance(x, str) else x
        except:
            return []

    # Applico la conversione alla colonna correct_windows
    df["correct_windows"] = df.get("correct_windows", []).apply(safe_eval)

    #Inizializzo le liste per la correttezza e l’attività per ogni riga.
    correct_per_row = []
    activity_per_row = []

    #Per ogni riga: determino lo stato (corretto, sbagliato, pausa) in base a correct windows e prendo l'attività associata
    for _, row in df.iterrows():
        correct = row.get("correct_windows", [])
        status = "pause"
        if isinstance(correct, list) and correct:
            valids = [x for x in correct if x is not None]
            if valids:
                status = "correct" if valids[0] is True else "wrong"
        correct_per_row.append(status)

        act = row.get(activity_col, "Pausa") if activity_col else "Pausa"
        activity_per_row.append(str(act) if pd.notna(act) else "Pausa")

    #Segmento la timeline in blocchi di stato uguale (pause/corretto/sbagliato).
    timeline = []
    current_status = correct_per_row[0]
    start = 0

    for i, status in enumerate(correct_per_row[1:], 1):
        if status != current_status:
            timeline.append((start, i - 1, current_status))
            start = i
            current_status = status
    timeline.append((start, len(correct_per_row) - 1, current_status))

    # Definisco i colori 
    colors = {'correct': '#2E86AB', 'wrong': '#F24236', 'pause': '#CCCCCC'}

    # Per ogni canale, preparo il grafico e le variabili di supporto.
    for col in signal_cols:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        x_plot, y_plot, color_plot = [], [], []
        x_pos = 0
        label_positions = []
        #Per ogni segmento se è una pausa comprimo il segmento, se è un attività mantengo la lunghezza originale e preparo la posizione dell'etichetta
        for start, end, status in timeline:
            seg_len = end - start + 1
            real_y = df[col].iloc[start:end+1].values
            real_colors = [status] * seg_len
            seg_activities = activity_per_row[start:end+1]

            if status == "pause":
                compressed_len = max(1, seg_len // pause_compression_factor)
                indices = np.linspace(0, seg_len - 1, compressed_len).astype(int)
                seg_y = real_y[indices]
                seg_colors = [status] * compressed_len
            else:
                seg_y = real_y
                seg_colors = real_colors

                # Etichetta attività
                most_common_act = Counter(seg_activities).most_common(1)[0][0]
                label_mid_x = x_pos + len(seg_y) // 2
                label_positions.append((label_mid_x, most_common_act, start, end))

            seg_x = np.arange(x_pos, x_pos + len(seg_y))
            x_plot.extend(seg_x)
            y_plot.extend(seg_y)
            color_plot.extend(seg_colors)
            x_pos += len(seg_y)

        # Disegno delle linee colorate
        for i in range(len(x_plot) - 1):
            if not np.isnan(y_plot[i]) and not np.isnan(y_plot[i+1]):
                c = colors.get(color_plot[i], '#CCCCCC')
                alpha = 0.9 if color_plot[i] != 'pause' else 0.4
                lw = 2 if color_plot[i] != 'pause' else 1
                ax.plot([x_plot[i], x_plot[i+1]], [y_plot[i], y_plot[i+1]],
                        color=c, linewidth=lw, alpha=alpha)

        # Etichette attività
        for mid_x, act, orig_start, orig_end in label_positions:
            # Calcolo la y massima nei dintorni del centro dell'attività per evitare sovrapposizione
            local_window = y_plot[max(0, mid_x - 20):min(len(y_plot), mid_x + 20)]
            mid_y = np.nanmax(local_window) if len(local_window) > 0 else ax.get_ylim()[1]
            text_y = mid_y + (0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]))

            # Inserisco il testo sopra il segmento di attività senza bordo
            ax.text(mid_x, text_y, f'{act}\n{orig_start}-{orig_end}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')


        # Finalizzo grafico
        ax.set_title(f'{col} - Timeline con Pause Compresse e Attività', fontsize=14)
        ax.set_xlabel('Campioni (timeline compressa)', fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True, alpha=0.3)
        legend_elements = [
            mpatches.Patch(color=colors['correct'], label='Corretto'),
            mpatches.Patch(color=colors['wrong'], label='Sbagliato'),
            mpatches.Patch(color=colors['pause'], label='Pausa compressa')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.show()

        # Debug
        if debug:
            correct_count = correct_per_row.count('correct')
            wrong_count = correct_per_row.count('wrong')
            pause_count = correct_per_row.count('pause')
            total_activity = correct_count + wrong_count
            print(f"\n{col} - Statistiche:")
            print(f"  Corretto: {correct_count}")
            print(f"   Sbagliato: {wrong_count}")
            print(f"   Pause: {pause_count}")
            print(f"   Accuratezza: {correct_count / total_activity * 100:.1f}%" if total_activity > 0 else "   Accuratezza: N/A")



#Radar Plot — Durata totale per attività: Mostra quanto il bambino ha fatto ciascuna attività in termini di numero di campioni
"""def radar_activity_duration(file_path, activity_col='activity'):
    df = pd.read_csv(file_path)
    df[activity_col] = df[activity_col].fillna("Pausa")
    df[activity_col] = df[activity_col].replace('', "Pausa")

    counts = df[activity_col].value_counts().sort_index()
    
    labels = counts.index.tolist()
    values = counts.values.tolist()
    # Chiudi il cerchio
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Chiudi il cerchio

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='darkorange')
    ax.fill(angles, values, color='orange', alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_title("Durata Totale per Attività (in campioni)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()"""

def radar_activity_duration(file_path, activity_col='activity'):
    df = pd.read_csv(file_path)
    #Sostituisco i valori NaN e vuoti con "Pausa"
    df[activity_col] = df[activity_col].fillna("Pausa")
    df[activity_col] = df[activity_col].replace('', "Pausa")

    # Conto quante volte appare ciascuna attività, ordinate per nome
    counts = df[activity_col].value_counts().sort_index()
    # Se tra le attività c'è "Pausa", la elimino dal conteggio
    if "Pausa" in counts.index:
        counts = counts.drop("Pausa")
    
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
    ax.set_title("Durata Totale per Attività (senza Pausa)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


#Radar Plot — Numero di sessioni (cambi attività): Conta quante transizioni ci sono state per ciascuna attività. 
#Utile per sapere quante volte un’attività è iniziata (indipendentemente dalla durata).

#Durata media dell'attività (Quanti bambini hanno fatto quell attività)
#media, std, 


