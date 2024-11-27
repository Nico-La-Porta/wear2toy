import os
import numpy as np

#funzioni per:
    # 1)estituire una lista di file con una specifica estensione
    # 2)Caricare annotazioni e segnali da file .npz
    # 3)Costruire dataset a partire da file .npz (annotazioni e segnali) 

def get_files_in_directory(path, extension):
    """
    Restituisce una lista di file con una specifica estensione in una directory.
    """
    return [f for f in os.listdir(path) if f.endswith(extension)]

def load_trace_data(path, annotations_file, signals_file):
    """
    Carica i dati di annotazioni e segnali da file .npz.
    """
    annotations_data = np.load(os.path.join(path, annotations_file))
    signals_data = np.load(os.path.join(path, signals_file))
    annotations = annotations_data['annotations']
    signals = signals_data['signals']
    
    print(f"Loaded annotations and signals for TraceID: {annotations_file.replace('_ann.npz', '')}")
    print(f"Annotations shape: {annotations.shape}")
    print(f"Signals shape: {signals.shape}")
    
    return annotations, signals

def build_dataset(path):
    """
    Costruisce il dataset a partire dai file .npz di annotazioni e segnali.
    """
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

def main():
    """
    Funzione principale per eseguire il processo.
    """
    # Percorso della cartella contenente i file .npz
    path = 'C:\\codes\\HumanActivityRecognition\\data\\raw\\train'

    # Costruisci il dataset
    dataset_traces_train = build_dataset(path)

    # Verifica il risultato
    print(f"Total number of traces in datasetTrain: {len(dataset_traces_train)}")
    
    # Esempio di stampa del primo elemento
    if dataset_traces_train:
        first_trace = dataset_traces_train[0]
        print(f"First trace ID: {first_trace['TraceID']}")
        print(f"Shape of first trace data: {first_trace['TraceData'].shape}")

if __name__ == "__main__":
    main()
