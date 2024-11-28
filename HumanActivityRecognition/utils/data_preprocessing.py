import os
import numpy as np

#from HumanActivityRecognition.app_config import PROCESSED_DATA_DIR_TRAIN

#funzioni per:
    # 1)restituire una lista di file con una specifica estensione
    # 2)Caricare annotazioni e segnali da file .npz
    # 3)Costruire dataset a partire da file .npz (annotazioni e segnali) 

def get_files_in_directory(path, extension):

    return [f for f in os.listdir(path) if f.endswith(extension)]

def load_trace_data(path, annotations_file, signals_file):

    annotations_data = np.load(os.path.join(path, annotations_file))
    signals_data = np.load(os.path.join(path, signals_file))
    annotations = annotations_data['annotations']
    signals = signals_data['signals']
    
    print(f"Loaded annotations and signals for TraceID: {annotations_file.replace('_ann.npz', '')}")
    print(f"Annotations shape: {annotations.shape}")
    print(f"Signals shape: {signals.shape}")
    
    return annotations, signals

def build_dataset(path):

    # Ottieni file di annotazioni e segnali
    annotations_files = get_files_in_directory(path, '_ann.npz')
    signals_files = get_files_in_directory(path, '_sig.npz')
    
    dataset_traces = []  # Lista per contenere i dati delle tracce

    for ann_file in annotations_files:
        # Estrai il TraceID
        traceID = ann_file.replace('_ann.npz', '')

        # Costruisci il nome del file dei segnali corrispondente
        sig_file = traceID + '_sig.npz'

        # Verifica che il file di segnali esista
        if sig_file in signals_files:
            # Carica i dati delle annotazioni e segnali
            annotations, signals = load_trace_data(path, ann_file, sig_file)

            # Aggiungi i dati al dataset
            trace = {
                'TraceID': traceID,
                'TraceData': signals  # Puoi aggiungere annotazioni come richiesto
            }
            dataset_traces.append(trace)
        else:
            print(f"Missing signals file for TraceID: {traceID}")
    
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


def main():
    """
    Funzione principale per eseguire il processo.
    """
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
    main()
