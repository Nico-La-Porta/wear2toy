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

def train(net, train_loader, test_loader=None, epochs: int = 10, batch_size: int = 16, 
          lr: float = 0.01, patience: int = 7, figure_name: str = "figure", 
          f1_average: str = 'macro', validate: bool = True, save_confusion_matrix: bool = False, 
          criterion=None, save_final_results: bool = False, is_best_trial: bool = False, 
          figures_dir=None, reports_dir=None):
    

    if validate and test_loader is None:
        raise ValueError("test_loader è richiesto quando validate è True")
    os.makedirs(FIGURES_DIR, exist_ok=True)  # Assicura che la cartella per le figure esista

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # Usa il criterion passato come parametro, altrimenti usa CrossEntropyLoss di default
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    # Pass the model to the appropriate device (GPU or CPU)
    net.to(device)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

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

    all_train_indices = []
    all_test_indices = []

    best_f1score = 0
    early_stopper = EarlyStopper(patience=patience, min_delta=0.001)
    # Variabili per salvare SOLO i risultati finali
    final_train_results = []
    final_test_results = []
    for e in range(epochs):
        train_losses = []
        net.train()
        epoch_train_results = []
        #calcolo dell'f1-score di train
        train_f1score = 0 #variabile per l'f1-score di train
        for batch_data in train_loader:
            if len(batch_data) == 4:  # inputs, targets, indices
                inputs, targets, indices, consecutivity = batch_data
                all_train_indices.extend(indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices)
            else:  # solo inputs, targets
                inputs, targets = batch_data
                indices = list(range(len(targets)))
                consecutivity = [True] * len(targets)
            
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()
            if isinstance(consecutivity, torch.Tensor):
                consecutivity = consecutivity.cpu().numpy().tolist()
                
            # Se indices è ancora problematico, usa range semplice
            if not isinstance(indices, (list, tuple)):
                indices = list(range(len(targets)))
            if not isinstance(consecutivity, (list, tuple)):
                consecutivity = [True] * len(targets)
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            targets = targets.view(-1)
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

            # Salva i risultati di questo batch
            for i in range(len(targets)):
                epoch_train_results.append({
                    'window_index': indices[i],
                    'true_label': targets[i].cpu().item(),
                    'predicted_label': predicted[i].cpu().item(),
                    'is_consecutive': consecutivity[i],
                    'correct': targets[i].cpu().item() == predicted[i].cpu().item(),
                    'epoch': e + 1
                })

            #calcolo l'f1-score
            train_f1score += f1_score(predicted.cpu(), targets.cpu(), average=f1_average)
        
        train_loss_history.append(np.mean(train_losses))
        train_f1score_history.append(train_f1score / len(train_loader)) #calcolo l'f1-score medio

        
        val_losses = []
        accuracy = 0
        f1score = 0
        epoch_test_results = []
        current_f1score = 0

        if validate:
            net.eval()

            with torch.no_grad():
                for batch_data in test_loader:
                    if len(batch_data) == 4:
                        inputs, targets, indices, consecutivity = batch_data
                    else:
                        inputs, targets = batch_data
                        indices = list(range(len(targets)))
                        consecutivity = [True] * len(targets)
                    if isinstance(indices, torch.Tensor):
                        indices = indices.cpu().numpy().tolist()
                    if isinstance(consecutivity, torch.Tensor):
                        consecutivity = consecutivity.cpu().numpy().tolist()
                    
                    # Se indices è ancora problematico, usa range semplice
                    if not isinstance(indices, (list, tuple)):
                        indices = list(range(len(targets)))
                    if not isinstance(consecutivity, (list, tuple)):
                        consecutivity = [True] * len(targets)

                    batch_size = inputs.size(0)
                    val_h = net.init_hidden(batch_size)
                    #val_h = tuple([each.data for each in val_h])
                    net.to(device)
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets.view(-1)
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

                    for i in range(len(targets)):
                        epoch_test_results.append({
                            'window_index': indices[i],
                            'true_label': targets[i].cpu().item(),
                            'predicted_label': top_class[i].cpu().item(),
                            'is_consecutive': consecutivity[i],
                            'correct': targets[i].cpu().item() == top_class[i].cpu().item(),
                            'epoch': e + 1
                        })

            val_loss_history.append(np.mean(val_losses))
            val_accuracy_history.append(accuracy / len(test_loader))
            val_f1score_history.append(f1score / len(test_loader))


                    # Calcola F1 score per early stopping
        if validate:
            epoch_preds = [r['predicted_label'] for r in epoch_test_results]
            epoch_true = [r['true_label'] for r in epoch_test_results]
            current_f1score = f1_score(epoch_true, epoch_preds, average=f1_average)
            
            if current_f1score > best_f1score:
                best_f1score = current_f1score
                # Salva i risultati della migliore epoca
                final_train_results = epoch_train_results.copy()
                final_test_results = epoch_test_results.copy()

        # Early stopping check...
        if early_stopper.early_stop(current_f1score):
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
    
    # Salva i CSV finali SOLO se è il miglior trial
    if save_final_results and is_best_trial:
        save_path = reports_dir if reports_dir else REPORTS_DIR
        os.makedirs(save_path, exist_ok=True)
        
        # DISTINGUI tra K-Fold training e Final training
        is_final_training = "final" in figure_name.lower() and "fold" not in figure_name.lower()
        
        if is_final_training:
            # TRAINING FINALE: salva un unico CSV con tutte le finestre
            logger.info("=== SALVATAGGIO RISULTATI TRAINING FINALE ===")
            
            # CSV Train finale (tutte le 27 finestre)
            train_df = pd.DataFrame(final_train_results)
            train_csv_path = os.path.join(save_path, f"final_train_results_all_data.csv")
            train_df.to_csv(train_csv_path, index=False)
            logger.info(f"Train results salvati: {train_csv_path} ({len(train_df)} finestre)")
            
            # CSV Test finale (tutte le 27 finestre - se presente test_loader)
            if final_test_results:
                test_df = pd.DataFrame(final_test_results)
                test_csv_path = os.path.join(save_path, f"final_test_results_all_data.csv")
                test_df.to_csv(test_csv_path, index=False)
                logger.info(f"Test results salvati: {test_csv_path} ({len(test_df)} finestre)")
            
        else:
            # K-FOLD TRAINING: NON salvare CSV (solo per debug se necessario)
            logger.info(f"K-Fold training - CSV non salvati per {figure_name}")
            # Opzionale: salva solo per debug
            # train_df = pd.DataFrame(final_train_results)
            # train_csv_path = os.path.join(save_path, f"debug_train_{figure_name}.csv")
            # train_df.to_csv(train_csv_path, index=False)

    return best_f1score



def evaluate_model(net, test_loader, figure_name="evaluation", f1_average='macro', save_confusion_matrix: bool = False, save_f1_score: bool = True, save_predictions_csv: bool = True, criterion=None, labels_dict=None, figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR):
    net.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    all_test_preds = []
    all_test_labels = []
    all_test_indices = []
    all_test_consecutivity = []
    val_losses = []
    val_accuracy = 0

    # DEBUG: Verifica dimensioni del DataLoader
    logger.info(f"\n=== DEBUG EVALUATE_MODEL ===")
    logger.info(f"Test loader dataset size: {len(test_loader.dataset)}")
    logger.info(f"Test loader number of batches: {len(test_loader)}")
    logger.info(f"Expected total samples: {len(test_loader.dataset)}")

    total_samples_processed = 0  # DEBUG: Conta finestre processate


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            if len(batch_data) == 4:  # data, labels, indices, consecutivity
                inputs, targets, indices, consecutivity = batch_data
            else:
                inputs, targets = batch_data
                indices = list(range(len(targets)))
                consecutivity = [True] * len(targets)
                
            batch_size = inputs.size(0)
            logger.info(f"Batch {batch_idx + 1}: {batch_size} finestre")
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
            top_p, top_class = output.topk(1, dim=1)
            equals = predicted == targets.long()
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


            all_test_indices.extend(indices)
            all_test_consecutivity.extend(consecutivity)
            logger.info(f"Batch {batch_idx + 1} - Indici salvati: {indices}")
        # DEBUG: Risultati finali
    logger.info(f"Totale finestre processate: {total_samples_processed}")
    logger.info(f"Lunghezza liste finali:")
    logger.info(f"  - all_test_preds: {len(all_test_preds)}")
    logger.info(f"  - all_test_labels: {len(all_test_labels)}")
    logger.info(f"  - all_test_indices: {len(all_test_indices)}")
    logger.info(f"  - all_test_consecutivity: {len(all_test_consecutivity)}")
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
    
    # DataFrame con tutte le informazioni necessarie per l'analisi degli errori
    predictions_df = pd.DataFrame({
        'window_index': all_test_indices,
        'true_label': all_test_labels,
        'predicted_label': all_test_preds,
        'is_consecutive': all_test_consecutivity,
        'correct': [true == pred for true, pred in zip(all_test_labels, all_test_preds)]
    })


    # Aggiungi nomi delle azioni se il mapping è fornito
    if labels_dict:
        predictions_df['true_action'] = [labels_dict.get(label, f'Unknown_{label}') for label in all_test_labels]
        predictions_df['predicted_action'] = [labels_dict.get(label, f'Unknown_{label}') for label in all_test_preds]


    # Salvataggio del CSV con informazioni complete
    if save_predictions_csv:
        os.makedirs(reports_dir, exist_ok=True)
        pred_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        pred_path = os.path.join(reports_dir, f"test_predictions_{pred_filename}.csv")
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Test predictions saved to: {pred_path}")


    # Salvataggio F1-score
    if save_f1_score:
        # Assicurati che la directory esista
        os.makedirs(reports_dir, exist_ok=True)
        
        # Rimuovo le prime due lettere dal nome 
        f1_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        f1_file_txt = os.path.join(reports_dir, f"f1-score_{f1_filename}.txt")
        
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
        
        # Determina le etichette per gli assi
        if labels_dict:
            # Usa i nomi delle azioni
            tick_labels = [labels_dict.get(i, f'Class_{i}') for i in range(len(np.unique(all_test_labels + all_test_preds)))]
        else:
            # Fallback ai nomi del dataset
            tick_labels = getattr(test_loader.dataset, 'classes', [f'Class_{i}' for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",  # in scala di grigi
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    linewidths=0.3, square=True, annot_kws={"size": 10}, cbar=True)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.title(f"Confusion Matrix - {figure_name}", fontsize=16) 
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        plt.tight_layout()  
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f"{figure_name}_eval_confusion_matrix.png"), 
                    bbox_inches='tight', dpi=300)  # Salvataggio in alta qualità
        plt.close()


    return mean_loss, mean_accuracy, f1_scores[f1_average]
