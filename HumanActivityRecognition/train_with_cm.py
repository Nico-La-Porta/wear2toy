import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from app_config import MODELS_DIR, FIGURES_DIR
from utils import check_gpu
from utils.log_config import logger

train_on_gpu=check_gpu.check_gpu_availability()

"""class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
 
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False"""



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1score = -float('inf')  # F1-score migliore inizializzato a -inf

    def early_stop(self, f1score):
        if f1score > self.best_f1score + self.min_delta:  # Se l'F1-score migliora
            self.best_f1score = f1score
            self.counter = 0  # Reset del contatore
        else:
            self.counter += 1  # Inizia a contare quando l'F1-score non migliora
            if self.counter >= self.patience:  # Se non migliora per 'patience' epoche
                return True  # Fermati
        return False

def train(net, train_loader, test_loader, epochs=10, batch_size=16, lr=0.01, patience=7, figure_name="figure", f1_average='macro'):

    os.makedirs(FIGURES_DIR, exist_ok=True)  # Assicura che la cartella per le figure esista



    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    if train_on_gpu:
        net.cuda()

    #storicizzo i valori di loss e accuracy
    train_loss_history = [] 
    val_loss_history = [] 
    val_accuracy_history = []
    val_f1score_history = []
    train_f1score_history = []

    #liste perr le predizioni e i target
    all_train_preds = []
    all_train_labels = []
    all_test_preds = []
    all_test_labels = []

    best_f1score = 0
    early_stopper = EarlyStopper(patience=patience, min_delta=0.001)

    for e in range(epochs):
        train_losses = []
        net.train()

        #calcolo dell'f1-score di train
        train_f1score = 0 #variabile per l'f1-score di train
        for inputs, targets in train_loader: 
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size(0)
            h = net.init_hidden(batch_size)
            opt.zero_grad()
            output, h = net(inputs, h, batch_size)
            loss = criterion(output, targets.long())
            train_losses.append(loss.item())

            loss.backward()
            opt.step()

            #salvo le predizioni e i target per la confusion matrix di training
            _, predicted = torch.max(output, 1) #prendo la classe con probabilità maggiore
            all_train_preds.extend(predicted.cpu().numpy()) #aggiungo le predizioni alla lista
            all_train_labels.extend(targets.cpu().numpy()) #aggiungo i target alla lista

            #calcolo l'f1-score
            train_f1score += f1_score(predicted.cpu(), targets.cpu(), average=f1_average)
        
        train_loss_history.append(np.mean(train_losses))
        train_f1score_history.append(train_f1score / len(train_loader)) #calcolo l'f1-score medio

        val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy = 0
        f1score = 0
        net.eval()

        with torch.no_grad():
            for inputs, targets in test_loader:
                batch_size = inputs.size(0)
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = net(inputs, val_h, batch_size)
                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                f1score += f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average=f1_average)

                #salvo le predizioni e i target per la confusion matrix di test
                all_test_preds.extend(top_class.cpu().numpy().flatten()) # flatten() per avere un array 1D
                all_test_labels.extend(targets.cpu().numpy())

        val_loss_history.append(np.mean(val_losses))
        val_accuracy_history.append(accuracy / len(test_loader))
        val_f1score_history.append(f1score / len(test_loader))

        logger.info(f"Epoch: {e+1}/{epochs}... "
                f"Train Loss: {np.mean(train_losses):.4f}... "
                f"Val Loss: {np.mean(val_losses):.4f}... "
                f"Val Acc: {accuracy / len(test_loader):.4f}... "
                f"F1-Score: {f1score / len(test_loader):.4f}")
        
        # Aggiorna il miglior F1-score e salva il modello con il nome univoco
        current_f1score = f1score / len(test_loader)

        if current_f1score > best_f1score:
            best_f1score = current_f1score


        # Early stopping basato sull'F1-score
        if early_stopper.early_stop(current_f1score):
            net.eval()
            break
    
    # Matrice di confusione per il training set
    cm_train = confusion_matrix(all_train_labels, all_train_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens", xticklabels=train_loader.dataset.classes, yticklabels=train_loader.dataset.classes)
    plt.title("Confusion Matrix - Train Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}_train_confusion_matrix.png"))
    plt.close()


    # Confusion Matrix - Test Set
    cm_test = confusion_matrix(all_test_labels, all_test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}_test_confusion_matrix.png"))
    plt.close()




    # Curve di train e validazione F1-score
    plt.figure(figsize=(10, 6))
    plt.plot(train_f1score_history, label="Train F1-Score", color="blue")
    plt.plot(val_f1score_history, label="Test F1-Score", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.title("F1-Score (Train vs Test)")
    plt.legend()

    # Trovo l'epoch con il valore massimo di F1-score
    max_f1_epoch = np.argmax(val_f1score_history)
    max_f1_value = val_f1score_history[max_f1_epoch]
    max_train_f1_epoch = np.argmax(train_f1score_history)
    max_train_f1_value = train_f1score_history[max_train_f1_epoch]


    # Aggiungi i punti e le annotazioni nel grafico per il massimo F1-score
    plt.scatter(max_f1_epoch, max_f1_value, color="red", label=f"Max Val F1: {max_f1_value:.6f}")
    plt.text(max_f1_epoch, max_f1_value, f"{max_f1_value:.6f}", fontsize=12, verticalalignment='bottom', color="red")
    
    plt.scatter(max_train_f1_epoch, max_train_f1_value, color="orange", label=f"Max Train F1: {max_train_f1_value:.6f}")
    plt.text(max_train_f1_epoch, max_train_f1_value, f"{max_train_f1_value:.6f}", fontsize=12, verticalalignment='bottom', color="orange")

    plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}_f1score_curve.png"))
    plt.close()  # Chiude la figura

    return best_f1score
