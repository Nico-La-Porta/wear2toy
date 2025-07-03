import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
import numpy as np
from sklearn.model_selection import train_test_split

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, early_stopping_patience=10):
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
        
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
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
            device, epochs=30, early_stopping_patience=8  # Reduced for faster optimization
        )
        
        return val_acc  # Optimize for validation accuracy
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.TrialPruned()

def run_optimization():
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize validation accuracy
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    study.optimize(objective, n_trials=50, timeout=3600)  # Reduced trials for faster testing
    
    # Print results
    print("\nOptimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    if study.best_trial is not None:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Validation Accuracy): {trial.value:.4f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No successful trials completed.")
    
    return study

def train_final_model(best_params):
    """Train final model with best hyperparameters"""
    print("\nTraining final model with best hyperparameters...")
    
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
        device, epochs=100, early_stopping_patience=15
    )
    
    # Test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
    
    test_acc = test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "resnet1d_optimized.pth")
    
    return model, test_acc

# Example usage:
if __name__ == "__main__":
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