import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from sklearn import metrics
from utils import check_gpu


train_on_gpu = check_gpu.check_gpu_availability()

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
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
        return False

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def train(net, train_loader, test_loader, epochs=10, batch_size=16, lr=0.01, patience=7, cm_filename="confusion_matrix.png"): 
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if train_on_gpu:
        net.cuda()

    train_loss_history = [] 
    val_loss_history = [] 
    val_accuracy_history = []
    val_f1score_history = []

    best_f1score = 0
    patience_counter = 0
    early_stopper = EarlyStopper(patience=patience, min_delta=0.0001)

    for e in range(epochs):
        train_losses = []
        net.train()
        
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

        train_loss_history.append(np.mean(train_losses))

        val_losses = []
        accuracy = 0
        f1score = 0
        y_true = []
        y_pred = []
        
        net.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                batch_size = inputs.size(0)
                val_h = net.init_hidden(batch_size)

                if train_on_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                output, val_h = net(inputs, val_h, batch_size)
                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))

                y_true.extend(targets.cpu().numpy())
                y_pred.extend(top_class.cpu().numpy())

        val_loss_history.append(np.mean(val_losses)) 
        val_accuracy_history.append(accuracy / len(test_loader))
        f1score = f1_score(y_true, y_pred, average='weighted')
        val_f1score_history.append(f1score)

        print(f"Epoch: {e+1}/{epochs}... "
              f"Train Loss: {np.mean(train_losses):.4f}... "
              f"Val Loss: {np.mean(val_losses):.4f}... "
              f"Val Acc: {accuracy / len(test_loader):.4f}... "
              f"F1-Score: {f1score:.4f}")
        
        if f1score > best_f1score:
            best_f1score = f1score

        if early_stopper.early_stop(np.mean(val_losses)):
            print("Early stopping triggered")
            break

    #Stampa della Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(cm_filename)  
    print(f"Confusion Matrix salvata come {cm_filename}")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    if f1score > best_f1score:
        best_f1score = f1score
        torch.save(net.state_dict(), "best_model.pth")
        print("Modello salvato con F1-score migliore!")

    return best_f1score
