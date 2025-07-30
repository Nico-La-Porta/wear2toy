from torch import nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_LENGTH
import numpy as np
from torch.utils.data import WeightedRandomSampler


from utils import  check_gpu 

import torch
from torch.utils.data import Dataset

class HARDataset(Dataset):
    def __init__(self, data, labels, window_indices=None, consecutivity=None, class_names=None):
        """
        Args:
            data (numpy.ndarray): Input data, forma (n_samples, n_features, seq_length).
            labels (numpy.ndarray): Output labels, forma (n_samples,).
            window_indices (list): Indici originali delle finestre
            consecutivity (numpy.ndarray): Array booleano per consecutività
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.classes = class_names if class_names is not None else [str(i) for i in range(len(np.unique(labels)))]
        
        if window_indices is not None:
            self.window_indices = window_indices
        else:
            self.window_indices = list(range(len(labels)))
            
        if consecutivity is not None:
            self.consecutivity = consecutivity
        else:
            self.consecutivity = [True] * len(labels)  # Default: tutte consecutive

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.data[idx], 
                self.labels[idx], 
                self.window_indices[idx], 
                self.consecutivity[idx])
    
    
train_on_gpu=check_gpu.check_gpu_availability()

#prende in input un batch di dati
def collate_fn(batch):
    #batch è ubna lista di tuple ritornare da __getitem__ del dataset
    #ogni tupla contiene (data, label, index, consecutivity)
    data_batch, labels_batch, indices_batch, consecutivity_batch = zip(*batch)
    
    X_batch = torch.stack([x.clone().detach() for x in data_batch])
    Y_batch = torch.tensor(labels_batch, dtype=torch.long)
    indices_batch = list(indices_batch)
    consecutivity_batch = list(consecutivity_batch)
    
    return X_batch, Y_batch, indices_batch, consecutivity_batch


#prende in input le eutichette y
"""def create_weighted_sampler(Y):
    #conta quante volte ciascuna etichetta appare
    unique_labels, counts = np.unique(Y, return_counts=True)
    weights = 1.0 / counts #calcola pesi inversamente proporzionali al numero di occorrenze delle etichette
    sample_weights = np.array([weights[label] for label in Y]) #arrai dove ogni elemento è il peso corrispondente all'etichetta in y 
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),replacement=True) #seleziona i campioni in modo causale ma con una probabilità proporzionale ai pesi specificati
    return sampler""" #ritorna sampler che può essere utilizzato nel dataloder per bilanciare il dataset durante l'addestramento


def create_weighted_sampler(Y):
    # Conto le occorrenze di ciascuna etichetta
    unique_labels, counts = np.unique(Y, return_counts=True)

    # Calcolo i pesi come radice quadrata (per evitare overiffing)
    weights = np.sqrt(1.0 / counts)

    #normalizzo i pesi in modo da avere una somma pari al numero di classi, questo mi permette di avere un bilanciamento 
    weights = weights / np.sum(weights) * len(unique_labels)

    weights_dict = {label: weight for label, weight in zip(unique_labels, weights)} # creo un dizionario che associa a ciascuna etichetta il suo peso
    sample_weights = np.array([weights_dict[label] for label in Y]) #creo un array di pesi per ciascun campione in Y =>ASSOCIA a ciascun campione il peso corrispondente alla sua etichetta
    # Crea il sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    return sampler

class DeepConvLSTM(nn.Module):
    
    #n_hidden=numero di unità nei livelli di LSTM che determina capacità della LSTM di memorizzare informazioni temporali
    #n_layers= numero di strati della LSTM
    #n_filters=numero filtri covoluzionali applicati a ciascun livello consoluzionale
    #num_classes= numero attività da classificare
    #filter_size=dimensione dei kernel convoluzionale
    #drop_prob: probabilità di dropout, utilizzata per ridurre il rischio di overfitting
    
    #INPUT= un tensoore di dimensioni [batchsize, NB_SENSOR_CHANNELS.sequence_lenght]
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=29, filter_size=5, drop_prob=0.5,
                 nb_sensor_channels=9, sliding_window_length=100,
                 single_fc = True):
        super(DeepConvLSTM, self).__init__() #inizializza iperparametri del modello
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.nb_sensor_channels = nb_sensor_channels
        self.sliding_window_length = sliding_window_length
             
        #PRENDE IN INGRESSO il numrro di canali, applica n filtri di dimensione filtersize
        #L'OUTPUT sarà (batcsize, nfilters, output_lenght)
        self.conv1 = nn.Conv1d(self.nb_sensor_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        
        #livello completamente connesso che mappa le n_hidden unità nascoste alla dimensione di output n_classes
        #l'output ha dimensioni (batchsize, n_classes) che rappresenta la probabilità per ciascuna delle classi di attività
        if single_fc:
            self.classification_head = nn.Linear(n_hidden, n_classes)
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(n_hidden, n_hidden/2),
                nn.ReLU(),
                nn.Linear(n_hidden/2, n_classes)
            )

        #applica una probabilità di drop_prob per "spegnere" casualmente alcune attività durante l'allenamento riducendo così il rischio di overfittin
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, x, hidden, batch_size, single_fc=True):

        #print(f"Input iniziale al modello: {x.shape}")
        # Se  feature aggiuntive (indici, consecutività), estraggo solo i sensori
        if x.shape[2] > self.nb_sensor_channels:
            # Prendo solo le prime nb_sensor_channels feature (i sensori)
            x_sensors = x[:, :, :self.nb_sensor_channels]
            print(f"Shape dopo estrazione sensori: {x_sensors.shape}")
            x = x_sensors
        else:
            x_sensors = x
        #-1 sta calcolando automaticamente la dimensione rimanente (batchsize),
        x = x.reshape(-1, self.nb_sensor_channels, self.sliding_window_length)
        #print(f"Dopo reshape per convoluzione: {x.shape}")
        x = F.relu(self.conv1(x))
        #print(f"Dopo conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"Dopo conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        #print(f"Dopo conv3: {x.shape}")
        x = F.relu(self.conv4(x))
        #print(f"Dopo conv4: {x.shape}")
        #x = x.reshape(5, -1, self.n_filters)
        # Per evitare problemi con la dimensione
        x = x.transpose(1, 2)
        
        #print(f"Dopo reshape per LSTM: {x.shape}")
        #print(f"La forma del tensore dopo il view è: {x.shape}")
  
        #si passa x attraverso due livelli LSTM uno dopo l'altro 
        x, hidden = self.lstm1(x, hidden)
        #print(f"Dopo LSTM1: {x.shape}")
        x, hidden = self.lstm2(x, hidden)
        #print(f"Dopo LSTM2: {x.shape}")
        
        

        #riorganizza x in un vettore 2D (bacthsize*sequence lenght, n_hidden) per adattarsi al layer fully connected
        x = x.contiguous().view(-1, self.n_hidden)
        #print(f"Dopo view per fully connected: {x.shape}")
        x = self.dropout(x)
        #print(f"Dopo dropout: {x.shape}")
        
        #passa l'output attraverso il layer fully connected, mappando x a dimensioni [batchsize*sequence_lenght, n_classes]
        x = self.classification_head(x)
        
        #ridimensiona l'output per ottenere [batchsize, lunghezzasequenza, n_classes] e mantiene solo l'ultimo step temporale (-1,:)
        out = x.reshape(batch_size, -1, self.n_classes)[:,-1,:]
        """# Se hai classification_head:
        x_last = x[:, -1, :]  # shape: (batch_size, n_hidden)
        x_last = self.dropout(x_last)

        if hasattr(self, "classification_head"):
            out = self.classification_head(x_last)
        else:
            out = self.fc(x_last)"""

        return out, hidden
    

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
        #restituisce una tupla contenente lo stato nascosto e lo stato della cella
    
    def set_n_classes(self, n_classes):
        self.n_classes = n_classes 
    
    #per definire numero di sensor_channels
    def set_nb_sensor_channels(self, nb_sensor_channels):
        self.nb_sensor_channels = nb_sensor_channels
    

    #per definire sliding window lenght
    def set_sliding_window_length(self, sliding_window_length):
        self.sliding_window_length = sliding_window_length


