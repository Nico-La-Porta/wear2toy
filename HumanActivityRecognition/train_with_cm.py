import os
import pandas as pd 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from app_config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from utils import check_gpu
from utils.log_config import logger
from sklearn.utils.class_weight import compute_class_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def calculate_class_weights(y_labels, num_classes):
    """
    Calcola i pesi delle classi per bilanciare il dataset
    """
    
    # Calcolo i pesi usando sklearn:  peso_classe_i = n_samples / (n_classes * n_samples_classe_i)
    class_weights = compute_class_weight( 
        'balanced',
        classes=np.arange(num_classes),
        y=y_labels
    )
    
    # Converte in tensor PyTorch
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"Pesi delle classi calcolati: {class_weights}")
    
    return class_weights_tensor

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

def train(net, train_loader, test_loader=None, epochs: int = 10, batch_size: int = 16, lr: float = 0.01, patience: int = 7, figure_name: str = "figure", f1_average: str = 'macro', validate: bool= True, save_confusion_matrix: bool = False, criterion=None):

    if validate and test_loader is None:
        raise ValueError("test_loader è richiesto quando validate è True")
    os.makedirs(FIGURES_DIR, exist_ok=True)  # Assicura che la cartella per le figure esista

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # Usa il criterion passato come parametro, altrimenti usa CrossEntropyLoss di default
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    # Pass the model to the appropriate device (GPU or CPU)
    net.to(device)

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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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

        
        val_losses = []
        accuracy = 0
        f1score = 0

        if validate:
            net.eval()

            with torch.no_grad():
                for inputs, targets in test_loader:
                    batch_size = inputs.size(0)
                    val_h = net.init_hidden(batch_size)
                    #val_h = tuple([each.data for each in val_h])
                    net.to(device)
                    inputs, targets = inputs.to(device), targets.to(device)

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

            # Aggiorna il miglior F1-score e salva il modello con il nome univoco
            current_f1score = f1score / len(test_loader)

            if current_f1score > best_f1score:
                best_f1score = current_f1score

            # Logging
            logger.info(f"Epoch: {e+1}/{epochs}... "
                    f"Train Loss: {np.mean(train_losses):.4f}... "
                    f"Val Loss: {np.mean(val_losses):.4f}... "
                    f"Val Acc: {accuracy / len(test_loader):.4f}... "
                    f"F1-Score: {f1score / len(test_loader):.4f}")
        
            # Early stopping basato sull'F1-score
            if early_stopper.early_stop(current_f1score):
                net.eval()
                break
        else:
            logger.info(f"Epoch: {e+1}/{epochs}... "
                    f"Train Loss: {np.mean(train_losses):.4f}... ")
    if save_confusion_matrix:
    # Matrice di confusione per il training set
        cm_train = confusion_matrix(all_train_labels, all_train_preds)
        plt.figure(figsize=(16, 13))
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens", xticklabels=train_loader.dataset.classes, yticklabels=train_loader.dataset.classes, linewidths=0.5, square=True)
        plt.title("Confusion Matrix - Train Set")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}_train_confusion_matrix.png"))
        plt.close()

        if validate:
            # Confusion Matrix - Test Set
            cm_test = confusion_matrix(all_test_labels, all_test_preds)
            plt.figure(figsize=(16, 13))
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes, linewidths=0.5, square=True)
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



def evaluate_model(net, test_loader, figure_name="evaluation", f1_average='macro', save_confusion_matrix: bool = False, save_f1_score: bool = True, save_predictions_csv: bool = True, criterion=None):
    net.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    all_test_preds = []
    all_test_labels = []

    val_losses = []
    val_accuracy = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            batch_size = inputs.size(0)
            val_h = net.init_hidden(batch_size)
            val_h = tuple([each.data for each in val_h])
            net.to(device)

            inputs, targets = inputs.to(device), targets.to(device)

            output, val_h = net(inputs, val_h, batch_size)
            loss = criterion(output, targets.long())
            val_losses.append(loss.item())

            _, predicted = torch.max(output, 1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(targets.cpu().numpy())

            equals = predicted == targets.long()
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Calcolo del mean_loss e mean_accuracy 
    mean_loss = np.mean(val_losses)
    mean_accuracy = val_accuracy / len(test_loader)
    
    # Calcolo dell'F1 score con diverse strategie
    f1_scores = {
        'macro': f1_score(all_test_labels, all_test_preds, average='macro'),
        'micro': f1_score(all_test_labels, all_test_preds, average='micro'),
        'weighted': f1_score(all_test_labels, all_test_preds, average='weighted')
    }
    
    # F1-score per classe
    f1_per_class = f1_score(all_test_labels, all_test_preds, average=None)

    logger.info(f"Evaluation Results - Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}, F1-Score ({f1_average}): {f1_scores[f1_average]:.4f}")
    
    #DataFrame con true e predicted
    predictions_df = pd.DataFrame({
        'true': all_test_labels,
        'predicted': all_test_preds
    })


    # Salvataggio opzionale del CSV
    if save_predictions_csv:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        pred_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        pred_path = os.path.join(REPORTS_DIR, f"predictions_{pred_filename}.csv")
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to: {pred_path}")


    # Salvataggio F1-score
    if save_f1_score:
        # Assicurati che la directory esista
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Rimuovo le prime due lettere dal nome 
        f1_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        f1_file_txt = os.path.join(REPORTS_DIR, f"f1-score_{f1_filename}.txt")
        
        # Salvo l'F1-score in TXT
        with open(f1_file_txt, 'w') as f:
            f.write(f"F1-Score (Macro): {f1_scores['macro']:.6f}\n")
            f.write(f"F1-Score (Micro): {f1_scores['micro']:.6f}\n")
            f.write(f"F1-Score (Weighted): {f1_scores['weighted']:.6f}\n\n")
            f.write("F1-Score per classe:\n")
            
            # Verifica se le classi sono disponibili nel dataset
            class_names = getattr(test_loader.dataset, 'classes', None)
            
            for i, score in enumerate(f1_per_class):
                class_name = class_names[i] if class_names else f"Class {i}"
                f.write(f"{class_name}: {score:.6f}\n")
        
        logger.info(f"F1-scores saved to: {f1_file_txt}")

    if save_confusion_matrix:
        cm = confusion_matrix(all_test_labels, all_test_preds)
        plt.figure(figsize=(16, 14), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",  # in scala di grigi
                    xticklabels=test_loader.dataset.classes,
                    yticklabels=test_loader.dataset.classes,
                    linewidths=0.3, square=True, annot_kws={"size": 10}, cbar=True)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.title(f"Confusion Matrix - {figure_name}", fontsize=16) 
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        plt.tight_layout()  
        plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}_eval_confusion_matrix.png"), 
                    bbox_inches='tight', dpi=300)  # Salvataggio in alta qualità
        plt.close()


    return mean_loss, mean_accuracy, f1_scores[f1_average]
