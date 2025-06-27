import os
import sys
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import neptune
from neptune.integrations.optuna import NeptuneCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HumanActivityRecognition.app_config import *

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[8,5,3], dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Handle shortcut connection
        self.use_projection = (in_channels != out_channels)
        if self.use_projection:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        
        # Shortcut path
        if self.use_projection:
            identity = self.shortcut(identity)
        else:
            identity = self.shortcut(identity)
        
        # Ensure same temporal dimension by adjusting if needed
        if out.size(2) != identity.size(2):
            # Take minimum size to handle potential dimension mismatches
            min_size = min(out.size(2), identity.size(2))
            out = out[:, :, :min_size]
            identity = identity[:, :, :min_size]
        
        out += identity
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, base_filters=64, dropout_rate=0.0):
        super().__init__()
        self.block1 = ResidualBlock1D(n_features, base_filters, dropout_rate=dropout_rate)
        self.block2 = ResidualBlock1D(base_filters, base_filters * 2, dropout_rate=dropout_rate)
        self.block3 = ResidualBlock1D(base_filters * 2, base_filters * 2, dropout_rate=dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_filters * 2, n_outputs)

    def forward(self, x):
        # x shape: (batch, timesteps, features)
        x = x.permute(0, 2, 1)  # to (batch, channels, timesteps)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)        # (batch, channels)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, early_stopping_patience=10, neptune_run=None, log_prefix=""):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Log to Neptune
        if neptune_run:
            neptune_run[f"{log_prefix}train/loss"].append(train_loss)
            neptune_run[f"{log_prefix}train/accuracy"].append(train_acc)
            neptune_run[f"{log_prefix}val/loss"].append(val_loss)
            neptune_run[f"{log_prefix}val/accuracy"].append(val_acc)
            neptune_run[f"{log_prefix}learning_rate"].append(optimizer.param_groups[0]['lr'])
        
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if neptune_run:
                    neptune_run[f"{log_prefix}early_stopped_epoch"] = epoch + 1
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss, val_acc

def objective(trial):
    try:
        # Hyperparameters to optimize
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        base_filters = trial.suggest_categorical('base_filters', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
        scheduler_patience = trial.suggest_int('scheduler_patience', 5, 20)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.8)
        
        # Create Neptune run for this trial
        neptune_trial_run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            tags=["optuna_trial", "resnet1d", "hyperparameter_optimization"],
            name=f"Trial_{trial.number}"
        )
        
        # Log trial parameters
        neptune_trial_run["trial/number"] = trial.number
        neptune_trial_run["hyperparameters"] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "base_filters": base_filters,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer_name,
            "scheduler_patience": scheduler_patience,
            "scheduler_factor": scheduler_factor
        }
        
        # Create data loaders
        train_dataset = MyDataset(X_train, y_train)
        val_dataset = MyDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = ResNet1D(
            n_timesteps=X_train.shape[1], 
            n_features=X_train.shape[2], 
            n_outputs=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Log model summary
        neptune_trial_run["model/architecture"] = str(model)
        neptune_trial_run["model/total_params"] = sum(p.numel() for p in model.parameters())
        neptune_trial_run["model/trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:  # SGD
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=1e-6
        )
        
        # Train model
        val_loss, val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            device, epochs=30, early_stopping_patience=8, neptune_run=neptune_trial_run, log_prefix="trial/"
        )
        
        # Log final results
        neptune_trial_run["trial/final_val_loss"] = val_loss
        neptune_trial_run["trial/final_val_accuracy"] = val_acc
        
        # Stop Neptune run
        neptune_trial_run.stop()
        
        return val_acc  # Optimize for validation accuracy
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        if 'neptune_trial_run' in locals():
            neptune_trial_run["trial/error"] = str(e)
            neptune_trial_run.stop()
        raise optuna.TrialPruned()

def run_optimization():
    # Initialize main Neptune run for the optimization study
    neptune_study_run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        tags=["optuna_study", "resnet1d", "hyperparameter_optimization"],
        name=f"ResNet1D_Optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Log experiment metadata
    neptune_study_run["experiment/dataset_info"] = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "n_timesteps": X_train.shape[1],
        "n_features": X_train.shape[2],
        "n_classes": num_classes
    }
    
    neptune_study_run["experiment/user"] = "Nicolò La Porta"
    neptune_study_run["experiment/date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    neptune_study_run["experiment/description"] = "ResNet1D hyperparameter optimization with Optuna"
    
    # Create Neptune callback for Optuna
    neptune_callback = NeptuneCallback(neptune_study_run)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize validation accuracy
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    study.optimize(objective, n_trials=50, timeout=3600, callbacks=[neptune_callback])
    
    # Log study results
    neptune_study_run["study/n_trials"] = len(study.trials)
    neptune_study_run["study/n_pruned"] = len([t for t in study.trials if t.state == TrialState.PRUNED])
    neptune_study_run["study/n_complete"] = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    
    if study.best_trial is not None:
        neptune_study_run["study/best_value"] = study.best_trial.value
        neptune_study_run["study/best_params"] = study.best_trial.params
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Validation Accuracy): {trial.value:.4f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No successful trials completed.")
    
    # Print results
    print("\nOptimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    neptune_study_run.stop()
    return study

def train_final_model(best_params):
    """Train final model with best hyperparameters"""
    print("\nTraining final model with best hyperparameters...")
    
    # Initialize Neptune run for final training
    neptune_final_run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        tags=["final_model", "resnet1d", "best_params"],
        name=f"ResNet1D_Final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Log best hyperparameters
    neptune_final_run["final/best_params"] = best_params
    neptune_final_run["final/user"] = "Nico-La-Porta"
    neptune_final_run["final/date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create data loaders with best batch size
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    test_dataset = MyDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # Create model with best parameters
    model = ResNet1D(
        n_timesteps=X_train.shape[1], 
        n_features=X_train.shape[2], 
        n_outputs=num_classes,
        base_filters=best_params['base_filters'],
        dropout_rate=best_params['dropout_rate']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Log model info
    neptune_final_run["final/model_summary"] = str(model)
    neptune_final_run["final/total_params"] = sum(p.numel() for p in model.parameters())
    neptune_final_run["final/trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Setup training with best parameters
    criterion = nn.CrossEntropyLoss()
    
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], 
                             weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], 
                              weight_decay=best_params['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], 
                            weight_decay=best_params['weight_decay'], momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=best_params['scheduler_factor'], 
        patience=best_params['scheduler_patience'], min_lr=1e-6
    )
    
    # Train for more epochs
    val_loss, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, epochs=100, early_stopping_patience=15, neptune_run=neptune_final_run, log_prefix="final/"
    )
    
    # Test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    test_acc = test_correct / test_total
    
    # Log final results
    neptune_final_run["final/test_accuracy"] = test_acc
    neptune_final_run["final/test_samples"] = test_total
    
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = "resnet1d_optimized.pth"
    torch.save(model.state_dict(), model_path)
    neptune_final_run["final/model_checkpoint"].upload(model_path)
    
    neptune_final_run.stop()
    
    return model, test_acc


# Example usage:
if __name__ == "__main__":
    # Check Neptune configuration
    if not os.getenv("NEPTUNE_API_TOKEN"):
        print("Warning: NEPTUNE_API_TOKEN environment variable not set. Please set it to use Neptune logging.")
        print("You can set it by running: export NEPTUNE_API_TOKEN='your-api-token'")
        exit(1)
    
    # Replace with your actual data
    # X_train, X_val, X_test should have shape (num_samples, n_timesteps, n_features)
    # y_train, y_val, y_test should have shape (num_samples,) with integer class labels
    
    # Example data generation (replace with your data loading)
    np.random.seed(42)
    n_samples, n_timesteps, n_features, num_classes = 1000, 128, 9, 6
    
    X_full = np.random.randn(n_samples, n_timesteps, n_features)
    y_full = np.random.randint(0, num_classes, n_samples)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {num_classes}")
    
    # Run hyperparameter optimization
    study = run_optimization()
    
    # Train final model with best parameters if optimization was successful
    if study.best_trial is not None:
        best_model, final_test_acc = train_final_model(study.best_params)
        print(f"\nOptimization complete! Final test accuracy: {final_test_acc:.4f}")
    else:
        print("Optimization failed. Please check your data and try again.")