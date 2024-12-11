import matplotlib.pyplot as plt
from HumanActivityRecognition.app_config import FIGURES_DIR
import os
from datetime import datetime


def plot_learning_curves(train_loss, val_loss, val_acc):
    epochs = range(1, len(train_loss) + 1)
    
    # Crea la figura del grafico
    plt.figure(figsize=(14, 5))
    
    # Primo grafico: Train Loss e Val Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Secondo grafico: Val Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, 'g', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()

    # Nome dinamico del file per evitare sovrascrittura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"learning_curves_{timestamp}.png"
    file_path = os.path.join(FIGURES_DIR, file_name)

    # Salva il grafico
    plt.savefig(file_path)
    plt.close()  # Chiude la figura per liberare la memoria

    print(f"Grafico salvato in: {file_path}")