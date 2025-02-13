import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from utils import check_gpu 
from utils import plot

train_on_gpu=check_gpu.check_gpu_availability()

class EarlyStopper:
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
        return False

def train(net, train_loader, test_loader, epochs=10, batch_size=16, lr=0.01, patience=7):

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
            

            h = net.init_hidden(batch_size)
            opt.zero_grad()
            output, h = net(inputs, h, batch_size)
            loss = criterion(output, targets.long())
            train_losses.append(loss.item())
            
            loss.backward()
            opt.step()

        train_loss_history.append(np.mean(train_losses))

        val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy = 0
        f1score = 0
        net.eval()

        with torch.no_grad():
            for inputs, targets in test_loader:
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                output, val_h = net(inputs, val_h, batch_size)
                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='weighted')
        
        val_loss_history.append(np.mean(val_losses)) 
        val_accuracy_history.append(accuracy / len(test_loader))
        val_f1score_history.append(f1score / len(test_loader))

        print(f"Epoch: {e+1}/{epochs}... "
              f"Train Loss: {np.mean(train_losses):.4f}... "
              f"Val Loss: {np.mean(val_losses):.4f}... "
              f"Val Acc: {accuracy / len(test_loader):.4f}... "
              f"F1-Score: {f1score / len(test_loader):.4f}")
        
        # Aggiorna il miglior F1-score
        current_f1score = f1score / len(test_loader)
        if current_f1score > best_f1score:
            best_f1score = current_f1score

        if early_stopper.early_stop(np.mean(val_losses)):
            print("Early stopping triggered")
            break
        
    return best_f1score
