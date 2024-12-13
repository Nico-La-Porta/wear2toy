import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from HumanActivityRecognition.utils import data_processing, check_gpu 
from HumanActivityRecognition.utils import plot

train_on_gpu=check_gpu.check_gpu_availability()

#(net=modello da allenare)
def train(net, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=16, lr=0.01):

    #ottimizzatore stocastico del gradiente per aggiornare i pesi del modello, momentum aiuta a accelrare la discesa
    #del gradinete nella dimensione desiderata e ridurre le oscillazioni
    #wight_decay: regolarizzazione L2 per prevenire overfitting
    
    
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #opt = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, weight_decay=1e-4, momentum=0.9)
    #opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if train_on_gpu:
        net.cuda()


    train_loss_history = [] 
    val_loss_history = [] 
    val_accuracy_history = []

    #per ogni epoca il modello viene allenato su tutti i dati di train
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        X_train, Y_train = shuffle(X_train, Y_train) # Shuffle dei dati all'inizio di ogni epoca
        # initialize hidden state: crea uno stato nascosto di dimensione (n_layer, batchsize,n_hidden)
        train_losses = []
        net.train()  # Imposta la rete in modalità allenamento

        #suddividiamo i dati di addestramento in minibatch, ogni bach ha dimensione (batchsize, n_features) per x e (batchsize,) per y 
        for batch in data_processing.iterate_minibatches(X_train, Y_train, batch_size):
            x, y = batch
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history


            h = tuple([each.data for each in h])

            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output, h = net(inputs, h, batch_size)
            loss = criterion(output, targets.long())
            train_losses.append(loss.item())
            loss.backward()
            opt.step()

        train_loss_history.append(np.mean(train_losses))
        # Valutazione sui dati di validazione
        val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy = 0
        f1score = 0
        net.eval()
        with torch.no_grad():
            for batch in data_processing.iterate_minibatches(X_test, Y_test, batch_size):
                x, y = batch
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                val_h = tuple([each.data for each in val_h])
                if train_on_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()



                output, val_h = net(inputs, val_h,batch_size)
                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='weighted')
        

        val_loss_history.append(np.mean(val_losses)) 
        val_accuracy_history.append(accuracy / (len(X_test) // batch_size))
        
        net.train()
        print(f"Epoch: {e+1}/{epochs}... "
              f"Train Loss: {np.mean(train_losses):.4f}... "
              f"Val Loss: {np.mean(val_losses):.4f}... "
              f"Val Acc: {accuracy / (len(X_test) // batch_size):.4f}... "
              f"F1-Score: {f1score / (len(X_test) // batch_size):.4f}")
        
    plot.plot_learning_curves(train_loss_history, val_loss_history, val_accuracy_history)

    return np.mean(val_losses)