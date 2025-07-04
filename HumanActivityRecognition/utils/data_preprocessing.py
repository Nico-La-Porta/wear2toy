import os
import numpy as np
import json
from utils.log_config import logger
from typing import List, Dict, Any
#from scipy.stats import ks_1samp, norm
from app_config import FIGURES_DIR
#from scipy import stats
#from sklearn.preprocessing import StandardScaler, RobustScaler
#funzioni per:
    # 1)restituire una lista di file con una specifica estensione
    # 2)Caricare annotazioni e segnali da file .npz
    # 3)Costruire dataset a partire da file .npz (annotazioni e segnali)
    # 4) Raggruppare dati per attività  

def get_files_in_directory(path, extension):

    return [f for f in os.listdir(path) if f.endswith(extension)]


def load_trace_data(path, annotations_file, signals_file):

    annotations_data = np.load(os.path.join(path, annotations_file))
    signals_data = np.load(os.path.join(path, signals_file))
    annotations = annotations_data['annotations']
    signals = signals_data['signals']
    
    #logger.debug(f"Loaded annotations and signals for TraceID: {annotations_file.replace('_ann.npz', '')}")
    #logger.debug(f"Annotations shape: {annotations.shape}")
    #logger.debug(f"Signals shape: {signals.shape}") 
    return annotations, signals

def build_dataset(path):

    # Ottengo file di annotazioni e segnali
    annotations_files = get_files_in_directory(path, '_ann.npz')
    signals_files = get_files_in_directory(path, '_sig.npz')
    
    dataset_traces = []  # Lista per contenere i dati delle tracce

    for ann_file in annotations_files:
        # Estrai il TraceID
        traceID = ann_file.replace('_ann.npz', '')

        # Costruisco il nome del file dei segnali corrispondente
        sig_file = traceID + '_sig.npz'

        # Verifico che il file di segnali esista
        if sig_file in signals_files:
            # Carico i dati delle annotazioni e segnali
            annotations, signals = load_trace_data(path, ann_file, sig_file)

            # Aggiungo i dati al dataset
            trace = {
                'TraceID': traceID,
                'TraceData': signals  
            }
            dataset_traces.append(trace)
        else:
            logger.warning(f"Missing signals file for TraceID: {traceID}")
    
    return dataset_traces

# Funzione per aggiungere etichette (activity) ai dati
def add_labels_to_dataset(dataset):

    new_dataset_labeled = []  # Lista per memorizzare il nuovo dataset con le etichette

    for trace in dataset:
        # Estrazione del numero dell'attività
        activity = int(trace['TraceID'].split('_')[2].replace("Activity", ""))  # Eseguo parsing dell'activity
        activity = activity - 1  #sottraggo 1 perchè successivamene mi servono etichette da 0 a 28 (e non da 1 a 29)
        
        original_data = trace['TraceData']

        # Creo la nuova colonna activity
        activity_column = np.full((original_data.shape[0], 1), activity)
        new_data = np.hstack((original_data, activity_column))  # Aggiungo l'activity_column ai dati originali

        # Aggiungo una nuova traccia al nuovo dataset
        new_trace = trace.copy()  # Creo una copia del dizionario
        new_trace['TraceData'] = new_data  # Aggiorno TraceData
        new_dataset_labeled.append(new_trace)

    return new_dataset_labeled



def remove_classes(dataset, classes_to_remove):
    """
    Rimuove le tracce appartenenti alle classi specificate dal dataset.
    
    Args:
    - dataset: Lista di dizionari contenente le tracce del dataset.
    - classes_to_remove: Lista di interi che rappresentano le classi da rimuovere.
    
    Returns:
    - Nuovo dataset con le tracce delle classi rimosse.
    """
    logger.info("Inizio rimozione classi. Classi da rimuovere: %s", classes_to_remove)
    new_dataset = []  # Lista per memorizzare il dataset senza le classi rimosse
    removed_count = 0 # Contatore per tracce rimosse
    
    for trace in dataset:
        # Estrazione del numero dell'attività
        activity = int(trace['TraceID'].split('_')[2].replace("Activity", ""))  # Eseguo parsing dell'activity
        activity = activity - 1  # Sottraggo 1 per avere etichette da 0 a 28
        
        if activity in classes_to_remove:
            removed_count += 1
            continue  # Salta la traccia se l'attività è una di quelle da rimuovere
        
        
        # Aggiungo la traccia al nuovo dataset
        new_dataset.append(trace)
    
    logger.info("Rimozione completata. Numero tracce rimosse: %d", removed_count)
    logger.info("Nuovo dataset contiene %d tracce.", len(new_dataset))

    return new_dataset




# Funzione per raggruppare il dataset per attività
def group_by_activity(dataset: List[Dict[str, Any]]) -> Dict[int, List[np.ndarray]]:
    """
    Raggruppa il dataset per attività (activity).
    
    :param dataset: Lista di dizionari, ciascuno contenente 'TraceData' (un array NumPy) 
    :return: Dizionario con attività come chiavi, e tracce di dati come valori
            (una lista di arrays per ciascuna attività).
    """
    activity_groups = {}

    # Raggruppo le tracce per attività
    for trace in dataset:
        # Estraggo l'attività dalla colonna 'TraceData' (ultima colonna)
        activity = trace['TraceData'][:, -1].astype(int)[0]  

        if activity not in activity_groups:
            activity_groups[activity] = []

        # Aggiungo i dati della traccia al gruppo corrispondente
        activity_groups[activity].append(trace['TraceData'][:, :-1])  # Rimuovo l'activity dalla traccia

    return activity_groups

# Funzione per combinare train e test e raggruppare per attività
def combine_and_group(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> Dict[int, List[np.ndarray]]:
    """
    Combina i dataset di train e test e raggruppa per attività.
    
    :param train_data: Dataset di train (lista di dizionari).
    :param test_data: Dataset di test (lista di dizionari).
    :return: Dizionario con attività come chiavi e tracce di dati come valori
            (una lista di arrays per ciascuna attività).
    """
    # Combino i dataset di train e test
    combined_dataset = train_data + test_data
    
    # Raggruppo per attività
    return group_by_activity(combined_dataset)



"""# Funzione per salvare i risultati in file JSON
def save_results_to_json(results, figures_dir):
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    for activity, activity_results in results.items():
        file_path = os.path.join(figures_dir, f"Activity_{activity}_results.json")
        with open(file_path, 'w') as f:
            json.dump(activity_results, f, indent=4)"""


# Funzione per eseguire il test di normalità e salvare i risultati
"""def check_normality_and_save_by_activity(activity_data, figures_dir):
    results = {}
    for activity, data_list in activity_data.items():
        combined_data = np.vstack(data_list) ## Combino i dati per l'attività
        activity_results = {}
        for feature_idx in range(combined_data.shape[1]):  # Itero su ogni canale (feature)
            feature_data = combined_data[:, feature_idx] ## Dati per una singola caratteristica
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            ks_stat, ks_p = stats.kstest(feature_data, 'norm', args=(mean, std))
            mean_val = mean
            std_val = std
            median_val = np.median(feature_data)
            iqr_val = stats.iqr(feature_data)
            activity_results[f"Channel_{feature_idx+1}"] = {
                "KS_H": ks_stat,
                "KS_p": ks_p,
                "Mean": mean_val,
                "Std": std_val,
                "Median": median_val,
                "IQR": iqr_val
            }
        results[activity] = activity_results
    save_results_to_json(results, figures_dir)"""


def apply_scalers_to_dataset(dataset, json_results, figures_dir):
    """
    Applica gli scaler ai dati già etichettati in base ai risultati del test di normalità.
    
    :param dataset: Dataset da scalare (lista di dizionari con 'TraceData' già etichettati).
    :param json_results: Risultati del test di normalità per ogni attività (dal file JSON).
    :param figures_dir: Directory dove sono salvati i risultati (per il salvataggio).
    :return: Nuovo dataset con i dati scalati.
    """
    new_dataset = []  # Lista per il nuovo dataset scalato

    for trace in dataset:
        # Estraggo dati
        trace_data = trace['TraceData']
        
        
        activity = trace_data[0, -1].astype(int)  # Estraggo l'attività dalla prima riga, ultima colonna
        
        # Ottengo i risultati del test di normalità per questa attività dal JSON
        activity_results = json_results.get(str(activity), {})
        
        # Creo una copia dei dati per applicare gli scalers
        scaled_trace_data = np.copy(trace_data)
        
        # Applico lo scaler per ciascun canale esclusa l'ultima colonna (l'etichetta)
        for feature_idx in range(trace_data.shape[1] - 1):  # Ignoro l'ultima colonna (activity)
            channel_key = f"Channel_{feature_idx+1}"
            stats_dict = activity_results.get(channel_key, {})
            ks_p_value = float(stats_dict.get("KS_p", 1.0))  # Valore di default se non trovato

            feature_data = trace_data[:, feature_idx]
            logger.info(f"p-value: {ks_p_value}")

            # Se il p-value è maggiore di 0.05, applica StandardScaler, altrimenti RobustScaler
            if ks_p_value > 0.05:
                # StandardScaler
                logger.info(f"Attività {activity}, Canale {channel_key}: p-value = {ks_p_value} (StandardScaler)")
                mean = float(stats_dict.get("Mean", np.mean(feature_data)))
                std = float(stats_dict.get("Std", np.std(feature_data)))
                if std == 0: std = 1e-8  # Evito divisione per zero
                scaled_feature = (feature_data - mean) / std
            else:
                logger.info(f"Attività {activity}, Canale {channel_key}: p-value = {ks_p_value} (RobustScaler)")
                median = float(stats_dict.get("Median", np.median(feature_data)))
                iqr = float(stats_dict.get("IQR", np.percentile(feature_data, 75) - np.percentile(feature_data, 25)))
                if iqr == 0: iqr = 1e-8
                scaled_feature = (feature_data - median) / iqr
            

            scaled_trace_data[:, feature_idx] = scaled_feature
            
        
        # Aggiungo la traccia scalata al nuovo dataset
        new_trace = trace.copy()
        new_trace['TraceData'] = scaled_trace_data
        new_dataset.append(new_trace)
    
    return new_dataset


# Carica i risultati dal JSON (già calcolati e salvati in precedenza)
def load_json_results(figures_dir):
    results = {}
    for filename in os.listdir(figures_dir):
        if filename.endswith('_results.json'):
            activity = filename.split('_')[1]
            with open(os.path.join(figures_dir, filename), 'r') as f:
                activity_results = json.load(f)
                results[activity] = activity_results
    return results




# Controlli dopo lo scaling
def check_scaled_data(original_dataset, scaled_dataset, dataset_name):
    """Controlla se lo scaling è stato applicato correttamente su una traccia e una feature a caso,
    mostrando la media e deviazione standard prima e dopo lo scaling."""
    
    num_traces = len(scaled_dataset)
    random_trace_idx = np.random.randint(0, num_traces)  # Scelgo una traccia a caso
    random_feature_idx = np.random.randint(0, scaled_dataset[random_trace_idx]['TraceData'].shape[1] - 1)  # Scelgo una feature a caso

    logger.info(f"Controllo {dataset_name}: Trace {random_trace_idx}, Feature {random_feature_idx}")

    original_values = original_dataset[random_trace_idx]['TraceData'][:, random_feature_idx]  # Valori originali
    scaled_values = scaled_dataset[random_trace_idx]['TraceData'][:, random_feature_idx]  # Valori scalati

    # Calcolo della media e deviazione standard prima e dopo lo scaling
    original_mean = np.mean(original_values)
    original_std = np.std(original_values)
    scaled_mean = np.mean(scaled_values)
    scaled_std = np.std(scaled_values)

    # Log delle informazioni
    logger.info(f"[PRIMA] {dataset_name} - Trace {random_trace_idx}, Feature {random_feature_idx}: {original_values[:5]}")
    logger.info(f"Media PRIMA: {original_mean}, Deviazione Standard PRIMA: {original_std}")
    
    logger.info(f"[DOPO] {dataset_name} - Trace {random_trace_idx}, Feature {random_feature_idx}: {scaled_values[:5]}")
    logger.info(f"Media DOPO: {scaled_mean}, Deviazione Standard DOPO: {scaled_std}")



"""def main():
    
    #Funzione principale per eseguire il processo.
    
    # Percorso della cartella contenente i file .npz
    path = PROCESSED_DATA_DIR_TRAIN

    # Costruisco il dataset
    dataset_traces_train = build_dataset(path)

    # Aggiungo le etichette al dataset
    dataset_traces_labeled = add_labels_to_dataset(dataset_traces_train)

    # Verifica ris
    print(f"Total number of traces in datasetTrain: {len(dataset_traces_train)}")
    

    # Esempio di stampa del primo elemento
    if dataset_traces_train:
        first_trace = dataset_traces_train[0]
        print(f"First trace ID: {first_trace['TraceID']}")
        print(f"Shape of first trace data: {first_trace['TraceData'].shape}")

    if dataset_traces_labeled:
        first_trace = dataset_traces_labeled[0]
        print(f"First trace ID labeled: {first_trace['TraceID']}")
        print(f"Shape of first trace data labeled: {first_trace['TraceData'].shape}")
if __name__ == "__main__":
    main()"""
