import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn, create_weighted_sampler
import optuna
import optuna.visualization as vis
import train_toys
#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\pdd_data'
print(path)


df=pd.read_csv(os.path.join(path,'3002_BA.csv'))
print(df.shape)
df = df[df['action_id'] != 0]
print(df.shape)
print(df.columns)
#stampa il contenuto della colonna 'action'
print(df['action'].value_counts())
print(df['toy_id'].value_counts())


df=pd.read_csv(os.path.join(path,'3005_BA.csv'))
print(df.shape)
df = df[df['action_id'] != 0]
print(df.shape)
print(df.columns)
#stampa il contenuto della colonna 'action'
print(df['action'].value_counts())
print(df['toy_id'].value_counts())


#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path

df_list_ball = [] #lista vuota per appendere i dataframe con attività non nulla
for file in os.listdir(path):
    if file.endswith('.csv'):
        #leggo solo i file che dopo l'undescore ha BA
        if file.split('_')[1]=='BA.csv':
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_ball.append(df_temp)

df_ball = pd.concat(df_list_ball)
print("Dimensioni del df_ball con tutte le attività non nulle")
print(df_ball.shape)
print(df_ball.columns)
print(df_ball['action'].value_counts())

#salvo il dataframe
df_ball.to_csv(os.path.join(path,'df_BA_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_ball['action_id'].unique():
    df_action = df_ball[df_ball["action_id"] == action_id] #filtro il dataframe in base all'attività
    print(f"Dimensioni del dataframe df_ball_action_{action_id}")
    print(df_action.shape) #stampo le dimensioni del dataframe
    print(df_action.columns) #stampo le colonne del dataframe
    print(df_action['action'].value_counts()) #stampo il conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_ball_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    print(f"Salvato il dataframe df_ball_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
#definisco i parametri
nb_sensor_channels = 13
sliding_window_length = 100
sliding_window_step = 20

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al toy palla
#e poi concateno tutto in un unica x_train, y_train

X= []
Y= []

for file in os.listdir(path):
    if file.endswith('.csv') and file.split('_')[1] == 'ball':
        file_path = os.path.join(path, file)
        print(file_path)
        X_windows, Y_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
        X.append(X_windows)
        Y.append(Y_windows)

# Concatenate all the windows into a single array
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

#NUMERO TOTALE DI FINESTRE PER LA PALLA
print("Numero totale di finestre per il giocattolo ball:")
print(X.shape)
print(Y.shape)

#verifca
print(X)
print(Y)


# estraggo gli id dei bambini per vedere quanti ne ho
kid_ids = X[:, :, -2]  # Assuming kid_id is the third last column

# Get unique kid_ids
unique_kid_ids = np.unique(kid_ids)
print("Unique kid_ids in X:")
print(unique_kid_ids)



#Splitto il dataset in base al numero di azioni eseguite per avere congruenza temporale tra train e test e per 
#cercare di bilanciare le finestre in train e test

#filtro per ogni
#filtro per contare il numero di finestre per ogni azione
unique_actions, counts = np.unique(Y, return_counts=True)
print(unique_actions)
action_counts = dict(zip(unique_actions, counts))

print("Numero di finestre per ogni azione:")
for action, count in action_counts.items():
    print(f"Azione {action}: {count} finestre")


#split ratio  (70% nel train e 30% nel test)
split_ratio = 0.7

# Split the data
X_train = []
Y_train = []
X_test = []
Y_test = []

for action in unique_actions:

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



# Creazione sampler pesato per il dataset di training
train_sampler = create_weighted_sampler(Y_train_mapped)









    # Crea i DataLoader con il batch_size suggerito
    #runno di nuovo il train con 5 secondi di finestra 
train_loader = DataLoader(train_dataset, batch_size=8,shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)

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

file_path = os.path.join(REPORTS_DIR, 'best_hyperparameters_ball.csv')
loaded_params_df = pd.read_csv(file_path)
loaded_params = loaded_params_df.iloc[0].to_dict()  # Convertola prima riga in un dizionario

# Estrai i valori
lr = loaded_params["lr"]
bs = int(loaded_params["batch_size"])  # Assicurati che sia un intero

print(loaded_params)
best_f1_score = train_toys.train(model, train_loader, test_loader, epochs=100, batch_size= bs, lr=lr)
print(f"Best F1 score: {best_f1_score}")

