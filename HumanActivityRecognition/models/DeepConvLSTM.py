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
    def __init__(self, data, labels):
        """
        Args:
            data (numpy.ndarray): Input data, forma (n_samples, n_features, seq_length).
            labels (numpy.ndarray): Output labels, forma (n_samples,).
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
train_on_gpu=check_gpu.check_gpu_availability()

#prende in input un batch di dati
def collate_fn(batch):
    X_batch, Y_batch = zip(*batch) #utilizziamo zip per separare i dati dalle etichette all'interno del batch
    #scompone il batch in due tupple (una contentente tutti gli elementi di x e l'altra gli el di y)
    X_batch = torch.stack([x.clone().detach() for x in X_batch]) #converte ogni elemento di x_batch in un tensore e poi li impila lungo una nuova dimensione per creare un unico tensore 3d
    Y_batch = torch.tensor(Y_batch, dtype=torch.long) #converte y_batch in un tensore
    return X_batch, Y_batch


#prende in input le eutichette y
def create_weighted_sampler(Y):
    #conta quante volte ciascuna etichetta appare
    unique_labels, counts = np.unique(Y, return_counts=True)
    weights = 1.0 / counts #calcola pesi inversamente proporzionali al numero di occorrenze delle etichette
    sample_weights = np.array([weights[label] for label in Y]) #arrai dove ogni elemento è il peso corrispondente all'etichetta in y 
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),replacement=True) #seleziona i campioni in modo causale ma con una probabilità proporzionale ai pesi specificati
    return sampler #ritorna sampler che può essere utilizzato nel dataloder per bilanciare il dataset durante l'addestramento

class DeepConvLSTM(nn.Module):
    
    #n_hidden=numero di unità nei livelli di LSTM che determina capacità della LSTM di memorizzare informazioni temporali
    #n_layers= numero di strati della LSTM
    #n_filters=numero filtri covoluzionali applicati a ciascun livello consoluzionale
    #num_classes= numero attività da classificare
    #filter_size=dimensione dei kernel convoluzionale
    #drop_prob: probabilità di dropout, utilizzata per ridurre il rischio di overfitting
    
    #INPUT= un tensoore di dimensioni [batchsize, NB_SENSOR_CHANNELS.sequence_lenght]
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=29, filter_size=5, drop_prob=0.5):
        super(DeepConvLSTM, self).__init__() #inizializza iperparametri del modello
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
             
        #PRENDE IN INGRESSO il numrro di canali, applica n filtri di dimensione filtersize
        #L'OUTPUT sarà (batcsize, nfilters, output_lenght)
        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        
        #livello completamente connesso che mappa le n_hidden unità nascoste alla dimensione di output n_classes
        #l'output ha dimensioni (batchsize, n_classes) che rappresenta la probabilità per ciascuna delle classi di attività
        self.fc = nn.Linear(n_hidden, n_classes)

        #applica una probabilità di drop_prob per "spegnere" casualmente alcune attività durante l'allenamento riducendo così il rischio di overfittin
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, x, hidden, batch_size):

        #print(f"Input iniziale al modello: {x.shape}")
        #-1 sta calcolando automaticamente la dimensione rimanente (batchsize),
        x = x.reshape(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
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
        x = self.fc(x)
        #print(f"Dopo fully connected: {x.shape}")
        
        #ridimensiona l'output per ottenere [batchsize, lunghezzasequenza, n_classes] e mantiene solo l'ultimo step temporale (-1,:)
        out = x.reshape(batch_size, -1, self.n_classes)[:,-1,:]
        
        
        # il risultato è un tensore di dimensioni batchsize, n_classes che rappresenta la previsione finale per ciascuna classe di attività ne batch
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
