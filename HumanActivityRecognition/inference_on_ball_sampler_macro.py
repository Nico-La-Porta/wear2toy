import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger

#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\pdd_data'
print(path)

#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_BA_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    print(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_ball = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "BA" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_BA*.csv"))
    print("Files:", files)
    
    kid_ball, kid_ball_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_ball = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        print("Original shape:", df.shape)
        kid_ball.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        print("Filtered shape:", df.shape)
        kid_ball_no_null.append(df['kid_id'].unique())

        print("Columns:", df.columns)
        print("Action counts:\n", df['action'].value_counts())
        print("Toy counts:\n", df['toy_id'].value_counts())
        print("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_ball_no_null)/len(kid_ball)}")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella balltella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha BE*.csv
        if file.split('_')[-1].startswith('C') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_ball.append(df_temp)

    df_ball = pd.concat(df_list_ball)
    print("Dimensioni del df_ball con tutte le attività non nulle")
    print(df_ball.shape)
    print(df_ball.columns)
    print(df_ball['action'].value_counts())

    # salvo il dataframe
    df_ball.to_csv(os.path.join(path,'df_BA_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_ball['action_id'].unique():
    df_action = df_ball[df_ball["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_ball_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_ball_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_ball_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_ball_action_{action_id}.csv")

#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.split('_')[1] == 'ball']:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"ball_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni ball_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")

#split ratio  (70% nel train e 30% nel test)
split_ratio = 0.7

# Split the data
X_train = []
Y_train = []
X_test = []
Y_test = []

"""for action in unique_actions:

    #trovo gli indici delle finestre corrispondenti a ciascuna azione
    action_indices = np.where(Y == action)[0]
    print(action_indices)

    # Calcolo il numero di finestre da usare per il train e per il test
    num_windows = len(action_indices)
    num_train = int(num_windows * split_ratio)
    num_test = num_windows - num_train
    
    # Divido gli indici delle finestre in train e test
    train_indices = action_indices[:num_train]
    test_indices = action_indices[-num_test:]
    
    # Aggiungo le finestre al train e al test set
    X_train.append(X[train_indices])
    Y_train.append(Y[train_indices])
    X_test.append(X[test_indices])
    Y_test.append(Y[test_indices])

# Concateno tutti i dati in un unico array
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(Y_test, axis=0)

print("Training set shape:", X_train.shape, Y_train.shape)
print (Y_train)
print("Test set shape:", X_test.shape, Y_test.shape)
print(Y_test)

#stampo il tipo di valore che contiene x_train e y_train(se int float ecc)
print("Tipo di X_train:", X_train.dtype)
print("Tipo di Y_train:", Y_train.dtype)

#stampo il tipo di x_train y train x test e y test
print("Tipo di X_train:", type(X_train))
print("Tipo di Y_train:", type(Y_train))
print("Tipo di X_test:", type(X_test))
print("Tipo di Y_test:", type(Y_test))

#rimuovo colonne in eccesso in x_train e x_test (rimangono solo le prime 9 colonne)
X_train = X_train[:, :, :9]
X_test = X_test[:, :, :9]
print("Dimensioni di X_train e X_test dopo aver rimosso le colonne in eccesso:")
print(X_train.shape, X_test.shape)

Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print("Etichette train:", Y_train)
print("Etichette test:", Y_test)

#tipo
print("Tipo di Y_train:", type(Y_train))
print("Tipo di Y_test:", type(Y_test))

# Trova tutte le etichette uniche presenti nei dati
unique_labels = np.unique(Y_train)

# Crea un dizionario che mappa ogni etichetta originale a un valore consecutivo
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Stampa il dizionario per vedere il mapping
print("Mapping delle etichette:", label_mapping)

# Applica il mapping ai dataset di train e test
Y_train_mapped = np.array([label_mapping[y] for y in Y_train])
Y_test_mapped = np.array([label_mapping[y] for y in Y_test])

# Controllo finale
print("Nuove etichette train:", np.unique(Y_train_mapped))
print("Nuove etichette test:", np.unique(Y_test_mapped))


# Creo i dataset per il training e il test
train_dataset = HARDataset(X_train, Y_train_mapped)
test_dataset = HARDataset(X_test, Y_test_mapped)

 #stampo il dataset di training e di test a livello di dimensioni
print("Dataset di training:", len(train_dataset))
print("Dataset di test:", len(test_dataset))

print("Tipo di train_dataset:", type(train_dataset))
print("Tipo di test_dataset:", type(test_dataset))


train_sampler=create_weighted_sampler(Y_train_mapped)







def objective(trial):
    # Definisci gli iperparametri da ottimizzare
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 12])

    # Crea i DataLoader con il batch_size suggerito
    #runno di nuovo il train con 5 secondi di finestra 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Crea il modello con gli iperparametri suggeriti# Carica il modello preaddestratocd
    model = DeepConvLSTM()

        # Rimuovi la testa originale, in modo da non caricare i pesi associati
    model.load_state_dict(torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_sampler.pth'), strict=False)


    # Ora sostituisco la testa del modello con la nuova dimensione di classi (4)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classi
    model.set_n_classes(4)


    # Congelo tutti i parametri tranne quelli della testa (fully connected)
    for param in model.parameters():
        param.requires_grad = False  # Congela tutti i pesi

    # Sblocca i parametri della testa (fully connected)
    for param in model.fc.parameters():
        param.requires_grad = True  # Solo i pesi della testa saranno addestrabili

    # Esegui l'allenamento
    best_f1_score = train.train(model, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr, f1_average="macro")

    return best_f1_score

# Creazione studio Optuna ottimizza, nel senso di minimizzare la loss in 100 prove
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Highest F1-score: ", study.best_value)


#salvo i best hyperparameters
best_hyperparameters = study.best_params
best_hyperparameters['best_f1_score'] = study.best_value
best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_ball_inference_with_sampler_macro.csv'), index=False)









    # Crea il modello con gli iperparametri suggeriti# Carica il modello preaddestratocd
model = DeepConvLSTM()

        # Rimuovi la testa originale, in modo da non caricare i pesi associati
model.load_state_dict(torch.load('best_model_dl.pth'), strict=False)

    # Ora sostituisci la testa del modello con la nuova dimensione di classi (4)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classi
model.set_n_classes(4)

    # Congela tutti i parametri tranne quelli della testa (fully connected)
for param in model.parameters():
        param.requires_grad = False  # Congela tutti i pesi

    # Sblocca i parametri della testa (fully connected)
for param in model.fc.parameters():
    param.requires_grad = True  # Solo i pesi della testa saranno addestrabili

for name, param in model.named_parameters():
    print(f"{name} requires_grad={param.requires_grad}")

# Ora crea e allena il modello con i migliori iperparametri
best_lr = study.best_params['lr']
best_batch_size = study.best_params['batch_size']

# Crea i DataLoader con i migliori iperparametri
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)


best_f1_score = train_with_cm.train(model, train_loader, test_loader, epochs=100,batch_size=best_batch_size, lr=best_lr, figure_name="model_ball_inference_with_sampler_macro", patience=7, f1_average='macro')
print(f"Best F1 score: {best_f1_score}")"""