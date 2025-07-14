import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from utils.data_preprocessing import remove_classes_from_car_data, get_class_distribution

from optuna.pruners import MedianPruner

from figures import plot_CM, create_class_distribution_bar_chart, create_side_by_side_bar_chart, create_single_distribution_bar_chart
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from utils import mapping_activity
from collections import Counter

#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\downstream_data'
path_1= 'C:\codes\HumanActivityRecognition\data\downstream_data\car_3009_test'
print(path)

#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_CAR_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    print(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_car = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "C*" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_C*.csv"))
    print("Files:", files)
    
    kid_car, kid_car_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_car = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        print("Original shape:", df.shape)
        kid_car.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        print("Filtered shape:", df.shape)
        kid_car_no_null.append(df['kid_id'].unique())

        print("Columns:", df.columns)
        print("Action counts:\n", df['action'].value_counts())
        print("Toy counts:\n", df['toy_id'].value_counts())
        print("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_car_no_null)/len(kid_car)}")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella cartella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha BE*.csv
        if file.split('_')[-1].startswith('C') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_car.append(df_temp)

    df_car = pd.concat(df_list_car)
    print("Dimensioni del df_car con tutte le attività non nulle")
    print(df_car.shape)
    print(df_car.columns)
    print(df_car['action'].value_counts())

    # salvo il dataframe
    df_car.to_csv(os.path.join(path,'df_CAR_non_null.csv'),index=False)

#filtro le classi per trovare solo quelle che mi interessano
logger.info("FILTRAGGIO CLASSI DAL DATASET CAR")

# Mostra distribuzione originale
logger.info("Distribuzione action_id originale:")
original_counts = df_car['action_id'].value_counts().sort_index()
for action_id, count in original_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


# Rimuovo le classi che non ti interessano
classes_to_remove = [2, 3, 5, 9, 11, 12, 14, 16, 18, 19, 27, 28, 29, 31, 32, 37, 38, 39, 40]
logger.info(f"Rimozione action_id: {classes_to_remove}")

df_car_filtered = df_car[~df_car['action_id'].isin(classes_to_remove)] # filtro il dataframe per rimuovere le classi che non mi interessano

logger.info(f"Righe prima del filtro: {len(df_car)}")
logger.info(f"Righe dopo il filtro: {len(df_car_filtered)}")
logger.info(f"Righe rimosse: {len(df_car) - len(df_car_filtered)}")


logger.info("Distribuzione action_id dopo filtro:")
filtered_counts = df_car_filtered['action_id'].value_counts().sort_index()
for action_id, count in filtered_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


#MI VADO A TENERE DA PARTE IL KID PER IL TEST
KID_TEST = 3009

# Divido il dataframe in due: uno per il training/validazione e uno per il test finale
df_kid_test = df_car_filtered[df_car_filtered['kid_id'] == KID_TEST]

# Tieni i dati di tutti gli altri per training e tuning
df_car_filtered = df_car_filtered[df_car_filtered['kid_id'] != KID_TEST]

#salvo
df_kid_test.to_csv(os.path.join(path_1, f'df_test_kid_{KID_TEST}.csv'), index=False)





logger.info(f"Bambino {KID_TEST} escluso dal training. Dati riservati per la fase finale.")
logger.info(f"Righe training/val: {df_car_filtered.shape[0]}")
logger.info(f"Righe test finale: {df_kid_test.shape[0]}")

#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_car_filtered['action_id'].unique():
    df_action = df_car_filtered[df_car_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_car_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_car_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_1,f'df_car_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_car_action_{action_id}.csv")

for action_id in df_kid_test['action_id'].unique():
    df_action = df_kid_test[df_kid_test["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_kid_test_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_kid_test_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_1,f'df_kid_test_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_kid_test_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo car
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}


for action_file in [f for f in os.listdir(path_1) if f.endswith('.csv') and f.split('_')[1] == 'car']:
    file_path = os.path.join(path_1, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"car_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni car_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts[f'car_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Tipo di Y: {Y.dtype}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")


if Y.ndim > 1:
    logger.info(f"Y ha {Y.ndim} dimensioni ({Y.shape}), lo appiattisco...")
    Y = Y.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y dopo flatten: {Y.shape}")



logger.info(f"Shape finale di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")


logger.info("=== INIZIO ELABORAZIONE TEST KID 3009 ===")

X_Test, Y_Test = [], []
kid_action_counts_test = {}

#applico sliding window sul test 
for action_file in [f for f in os.listdir(path_1) if f.endswith('.csv') and f.split('_')[1] == 'kid' and f.split('_')[2] == 'test']:
    file_path = os.path.join(path_1, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows_test, Y_windows_test, kid_id_action_dict_test = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
    X_Test.append(X_windows_test)
    Y_Test.append(Y_windows_test)
    logger.info(f"Numero totale di finestre per l'azione {action} nel test: {len(X_windows_test)}")
    kid_action_counts[f"kid_test_action_{action}"] = kid_id_action_dict
logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts[f'kid_test_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


X_Test = np.concatenate(X_Test, axis=0)
Y_Test = np.concatenate(Y_Test, axis=0)

# Stampo le dimensioni di X_Test e Y_Test
logger.info(f"Dimensioni di X_Test finale: {X_Test.shape}")
logger.info(f"Dimensioni di Y_Test finale: {Y_Test.shape}")
logger.info(f"Tipo di Y_Test: {Y_Test.dtype}")
logger.info(f"Shape di Y_Test: {Y_Test.shape}")


if Y_Test.ndim > 1:
    logger.info(f"Y_Test ha {Y_Test.ndim} dimensioni ({Y_Test.shape}), lo appiattisco...")
    Y_Test = Y_Test.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y_Test dopo flatten: {Y_Test.shape}")

logger.info(f"Shape finale di Y_Test: {Y_Test.shape}")
logger.info(f"Primi 5 elementi di Y_Test: {Y_Test[:5]}")
    




#remap delle etichette per renderle consecutive
unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

logger.info(f"Mappatura etichette: {label_mapping}")

# Ora il remapping funzionerà correttamente
Y_remapped = np.array([label_mapping[label] for label in Y])

#rempappo anche le etichette del test
Y_Test_remapped = np.array([label_mapping[label] for label in Y_Test])


logger.info("Distribuzione finale delle classi con etichette rimappate:")
get_class_distribution(Y_remapped, "Car Dataset con Etichette Rimappate")
get_class_distribution(Y_Test_remapped, "Car Dataset Test con Etichette Rimappate")

#uso le Y_remapped per il resto del codice
Y = Y_remapped
Y_Test = Y_Test_remapped


# Trovo le classi presenti nel test
test_class_distribution = Counter(Y_Test)
train_class_distribution = Counter(Y)

# Classi assenti nel test
missing_classes = [label for label in np.unique(Y) if label not in test_class_distribution]

logger.info(f"Classi mancanti nel test: {missing_classes}")


# Calcolo quante finestre spostare per ogni classe mancante (15% arrotondato per difetto)
windows_to_move = {}
for missing_class in missing_classes:
    total_windows = train_class_distribution[missing_class]
    windows_to_move[missing_class] = int(round(total_windows * 0.15))  # int() arrotondo per eccesso
    logger.info(f"Classe {missing_class}: {total_windows} finestre totali -> {windows_to_move[missing_class]} finestre da spostare")


# Creo maschere per identificare le finestre da spostare
indices_to_move = []
np.random.seed(42)  # Per riproducibilità


for missing_class in missing_classes:
    # Trovo tutti gli indici delle finestre di questa classe
    class_indices = np.where(Y == missing_class)[0]
    
    # Seleziono casualmente le finestre da spostare
    n_to_move = windows_to_move[missing_class] #mi prendo il numero di finestre da spostare per questa classe
    if n_to_move > 0:
        selected_indices = np.random.choice(class_indices, size=n_to_move, replace=False) #seleziono senza ripetizione n_to_move finestre da spostare
        indices_to_move.extend(selected_indices) #aggiungo gli indici selezionati alla lista degli indici da spostare

# Converto in array numpy
indices_to_move = np.array(indices_to_move)

logger.info(f"Totale finestre da spostare: {len(indices_to_move)}")

# Sposto le finestre dal train al test
if len(indices_to_move) > 0:
    # Estraggo le finestre da spostare
    X_to_move = X[indices_to_move]
    Y_to_move = Y[indices_to_move]
    
    # Rimuovo le finestre dal training set
    mask_keep = np.ones(len(X), dtype=bool)
    mask_keep[indices_to_move] = False
    
    X_train_updated = X[mask_keep]
    Y_train_updated = Y[mask_keep]
    
    # Aggiungo le finestre al test set
    X_Test_updated = np.concatenate([X_Test, X_to_move], axis=0)
    Y_Test_updated = np.concatenate([Y_Test, Y_to_move], axis=0)
    
    # Aggiorno le variabili
    X = X_train_updated
    Y = Y_train_updated
    X_Test = X_Test_updated
    Y_Test = Y_Test_updated
    
    logger.info(f"Finestre spostate con successo!")
    logger.info(f"Nuovo training set: {X.shape}")
    logger.info(f"Nuovo test set: {X_Test.shape}")
    
    # Verifico la nuova distribuzione
    updated_train_distribution = Counter(Y)
    updated_test_distribution = Counter(Y_Test)
    
    logger.info("Nuova distribuzione training:")
    for class_label in sorted(updated_train_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_train_distribution[class_label]} finestre")
    
    logger.info("Nuova distribuzione test:")
    for class_label in sorted(updated_test_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_test_distribution[class_label]} finestre")
    
    # Verifico che ora tutte le classi siano presenti nel test
    missing_classes_after = [label for label in np.unique(Y) if label not in updated_test_distribution]
    logger.info(f"Classi ancora mancanti nel test: {missing_classes_after}")
    
else:
    logger.info("Nessuna finestra da spostare")

# Aggiorno anche le distribuzioni per il resto del codice
train_class_distribution = Counter(Y)
test_class_distribution = Counter(Y_Test)




car_mapping = mapping_activity.CAR_ACTION_MAPPING
Y_names = [car_mapping["encoded_to_name"][y] for y in Y]
Y_Test_names = [car_mapping["encoded_to_name"][y] for y in Y_Test]



create_single_distribution_bar_chart(
    Y=Y_names,
    toy_name="CAR",
    title_suffix="Train Dataset",
    save_path=os.path.join(FIGURES_DIR, "car_train_distribution"),
    use_class_prefix=False  
)

create_single_distribution_bar_chart(
    Y=Y_Test_names,
    toy_name="CAR",
    title_suffix="Test Dataset",
    save_path=os.path.join(FIGURES_DIR, "car_test_distribution"),
    use_class_prefix=False  
)


#concateno train e test
X_combined = np.concatenate([X, X_Test], axis=0)
Y_combined = np.concatenate([Y, Y_Test], axis=0)


car_mapping = mapping_activity.CAR_ACTION_MAPPING
Y_combined_names = [car_mapping["encoded_to_name"][y] for y in Y_combined]

# Stampo bar chart della distribuzione combinata
create_single_distribution_bar_chart(
    Y=Y_combined_names,
    toy_name="CAR",
    title_suffix="",
    save_path=os.path.join(FIGURES_DIR, "car_combined_distribution"),
    use_class_prefix=False  
)