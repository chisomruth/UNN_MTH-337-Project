
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import json
from datetime import datetime
from MLP.mlp_config import Config, set_seed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

set_seed(42)
config = Config()

def load_and_preprocess_data():
    print("\n Loading and preprocessing data...")
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"Original shape: {df.shape}")
    
    df = df.dropna()
    print(f"After dropping NaN: {df.shape}")

    X = df.drop(['Job Title', 'Salary'], axis=1)
    y = df['Salary'].values
    
    X = pd.get_dummies(X, columns=['Gender', 'Education Level'], drop_first=False)
    
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    print(f" Feature columns: {list(X.columns)}")
    print(f" Number of features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(config.RESULTS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    y_scaler_path = os.path.join(config.RESULTS_DIR, 'y_scaler.pkl')
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    
    print(f" Training samples: {X_train_scaled.shape[0]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    print(f" Salary range: ${y.min():.0f} - ${y.max():.0f}")
    print(f"Data preprocessing complete")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X.shape[1], y_scaler


def create_mlp_model(input_dim, learning_rate, optimizer_name):
    set_seed(config.SEED)
    
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        
        Dense(1)
    ])

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'] 
    )
    
    return model


def train_model(optimizer_name, learning_rate, X_train, y_train, X_test, y_test, input_dim, y_scaler):
    print(f"\n{'='*80}")
    print(f"Training: Optimizer={optimizer_name.upper()}, LR={learning_rate}")
    print(f"{'='*80}")
    set_seed(config.SEED)

    model = create_mlp_model(input_dim, learning_rate, optimizer_name)

    history = model.fit(
        X_train, y_train,
        validation_split=config.VALIDATION_SPLIT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1,
        shuffle=True
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"MSE:  {mse:,.2f}")
    print(f" RMSE: {rmse:,.2f}")
    print(f" MAE:  {mae:,.2f}")
    print(f" R²:   {r2:.4f}")

    model_name = f"{optimizer_name}_lr{learning_rate}"
    model_path = os.path.join(config.RESULTS_DIR, f'{model_name}.keras')
    model.save(model_path)
    print(f" Model saved: {model_path}")
 
    history_path = os.path.join(config.RESULTS_DIR, f'{model_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f" History saved: {history_path}")

    predictions_path = os.path.join(config.RESULTS_DIR, f'{model_name}_predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'y_test': y_test_original, 'y_pred': y_pred}, f)
    
    return history.history, model, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 
                                     'y_test': y_test_original, 'y_pred': y_pred}


def plot_results(all_results):
    print("\n Creating visualizations...")
    for optimizer_name in config.OPTIMIZERS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{optimizer_name.upper()} - Loss Curves', 
                     fontsize=16, fontweight='bold')
        
        for idx, lr in enumerate(config.LEARNING_RATES):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            results = all_results[optimizer_name][lr]
            epochs = range(1, len(results['train_loss']) + 1)
            
            ax.plot(epochs, results['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, results['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('MSE Loss', fontsize=12)
            ax.set_title(f'LR = {lr}', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 
                                 f'{optimizer_name}_loss_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    for optimizer_name in config.OPTIMIZERS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{optimizer_name.upper()} - MAE Curves', 
                     fontsize=16, fontweight='bold')
        
        for idx, lr in enumerate(config.LEARNING_RATES):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            results = all_results[optimizer_name][lr]
            epochs = range(1, len(results['train_mae']) + 1)
            
            ax.plot(epochs, results['train_mae'], 'b-', 
                   label='Train MAE', linewidth=2)
            ax.plot(epochs, results['val_mae'], 'r-', 
                   label='Val MAE', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('MAE', fontsize=12)
            ax.set_title(f'LR = {lr}', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 
                                 f'{optimizer_name}_mae_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(config.LEARNING_RATES))
    width = 0.2
    
    for idx, optimizer_name in enumerate(config.OPTIMIZERS):
        best_losses = [all_results[optimizer_name][lr]['best_val_loss'] 
                       for lr in config.LEARNING_RATES]
        offset = width * (idx - 1.5)
        ax.bar(x + offset, best_losses, width, label=optimizer_name.upper())
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Best Validation Loss by Optimizer and Learning Rate', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr}' for lr in config.LEARNING_RATES])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'best_loss_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, optimizer_name in enumerate(config.OPTIMIZERS):
        r2_scores = [all_results[optimizer_name][lr]['test_r2'] 
                     for lr in config.LEARNING_RATES]
        offset = width * (idx - 1.5)
        ax.bar(x + offset, r2_scores, width, label=optimizer_name.upper())
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set R² Score by Optimizer and Learning Rate', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr}' for lr in config.LEARNING_RATES])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'r2_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = []
    for optimizer_name in config.OPTIMIZERS:
        row = [all_results[optimizer_name][lr]['test_rmse'] 
               for lr in config.LEARNING_RATES]
        heatmap_data.append(row)
    
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.0f',
                xticklabels=[f'{lr}' for lr in config.LEARNING_RATES],
                yticklabels=[opt.upper() for opt in config.OPTIMIZERS],
                cmap='YlOrRd_r',
                cbar_kws={'label': 'Test RMSE'},
                ax=ax)
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_title('Test Set RMSE Heatmap (Lower is Better)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'rmse_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Predicted vs Actual Salary (Best LR for Each Optimizer)', 
                 fontsize=16, fontweight='bold')
    
    for idx, optimizer_name in enumerate(config.OPTIMIZERS):
        best_lr = min(config.LEARNING_RATES, 
                     key=lambda lr: all_results[optimizer_name][lr]['test_rmse'])
        
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        results = all_results[optimizer_name][best_lr]
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        ax.scatter(y_test, y_pred, alpha=0.6, s=50)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Salary ($)', fontsize=12)
        ax.set_ylabel('Predicted Salary ($)', fontsize=12)
        ax.set_title(f'{optimizer_name.upper()} (LR={best_lr})\n' + 
                    f'R²={results["test_r2"]:.4f}, RMSE=${results["test_rmse"]:,.0f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'prediction_scatter.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved!")