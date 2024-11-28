from torch import nn
import torch.nn.functional as F
from HumanActivityRecognition import run_config

from HumanActivityRecognition.utils import  check_gpu 

train_on_gpu=check_gpu.check_gpu_availability()
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
        self.conv1 = nn.Conv1d(run_config.NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        #livello completamente connesso che mappa le n_hidden unità nascoste alla dimensione di output n_classes
        #l'output ha dimensioni (batchsize, n_classes) che rappresenta la probabilità per ciascuna delle classi di attività
        self.fc = nn.Linear(n_hidden, n_classes)

        #applica una probabilità di drop_prob per "spegnere" casualmente alcune attività durante l'allenamento riducendo così il rischio di overfittin
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, x, hidden, batch_size):

        
        #-1 sta calcolando automaticamente la dimensione rimanente (batchsize),
        x = x.reshape(-1, run_config.NB_SENSOR_CHANNELS, run_config.SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        #x = x.reshape(5, -1, self.n_filters)
        # Per evitare problemi con la dimensione
        x = x.reshape(x.size(0), -1, self.n_filters)
        #print(f"La forma del tensore dopo il view è: {x.shape}")

        #si passa x attraverso due livelli LSTM uno dopo l'altro 
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        

        #riorganizza x in un vettore 2D (bacthsize*sequence lenght, n_hidden) per adattarsi al layer fully connected
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        
        #passa l'output attraverso il layer fully connected, mappando x a dimensioni [batchsize*sequence_lenght, n_classes]
        x = self.fc(x)
        
        #ridimensiona l'output per ottenere [batchsize, lunghezzasequenza, n_classes] e mantiene solo l'ultimo step temporale (-1,:)
        out = x.reshape(batch_size, -1, self.n_classes)[:,-1,:]
        
        # il risultato è un tensore di dimensioni batchsize, n_classes che rappresenta la previsione finale per ciascuna classe di attività ne batch
        return out, hidden
    

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        
        #parametro arbitrario del modello per conoscere il dispositivo su cui è presente il modello (GPU O CPU)
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
    #restituisce una tupla contenente lo stato nascosto e lo stato della cella