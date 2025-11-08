import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, History
from CNN.cnn_config import set_seed, Config

import warnings
warnings.filterwarnings("ignore")
set_seed(42)
config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def create_data_generators(seed):
    train_gen = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=0.1
    )
    val_gen = ImageDataGenerator(rescale=1/255.)
    train_flow_gen = train_gen.flow_from_directory(
        directory=config.TRAIN_DIR,
        class_mode='sparse',
        batch_size=config.BATCH_SIZE,
        target_size=config.IMAGE_SIZE,
        seed=seed,
        shuffle=True
    )
    val_flow_gen = val_gen.flow_from_directory(
        directory=config.VAL_DIR,
        class_mode='sparse',
        batch_size=config.BATCH_SIZE,
        target_size=config.IMAGE_SIZE,
        seed=seed,
        shuffle=False 
    )
    return train_flow_gen, val_flow_gen


def create_cnn_model(num_classes, learning_rate, optimizer_name):
    set_seed(config.SEED)
    model = Sequential([
        Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(0.001), 
               input_shape=(300, 300, 3)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        MaxPooling2D(2,2),
        
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        Conv2D(512, (3,3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        MaxPooling2D(2,2),
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(0.001)),
        tf.keras.layers.LeakyReLU(),
        Dropout(0.4),
        Dense(256, kernel_regularizer=l2(0.001)),
        tf.keras.layers.LeakyReLU(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
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
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(optimizer_name, learning_rate, train_gen, val_gen, num_classes):
    print(f"\n{'='*80}")
    print(f"Training: Optimizer={optimizer_name.upper()}, LR={learning_rate}")
    print(f"{'='*80}")
    set_seed(config.SEED)
    model = create_cnn_model(num_classes, learning_rate, optimizer_name)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        verbose=1
    )
    model_name = f"{optimizer_name}_lr{learning_rate}"
    model_path = os.path.join(config.RESULTS_DIR, f'{model_name}.keras')
    model.save(model_path)
    print(f"Model saved: {model_path}")
    history_path = os.path.join(config.RESULTS_DIR, f'{model_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"History saved: {history_path}")
    return history.history, model


def plot_results(all_results):
    print("\n Creating visualizations...")
    
    for optimizer_name in config.OPTIMIZERS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{optimizer_name.upper()} - Learning Rate Comparison', 
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
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'LR = {lr}', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 
                                 f'{optimizer_name}_loss_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    for optimizer_name in config.OPTIMIZERS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{optimizer_name.upper()} - Accuracy Comparison', 
                     fontsize=16, fontweight='bold')
        
        for idx, lr in enumerate(config.LEARNING_RATES):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            results = all_results[optimizer_name][lr]
            epochs = range(1, len(results['train_accuracy']) + 1)
            
            ax.plot(epochs, results['train_accuracy'], 'b-', 
                   label='Train Accuracy', linewidth=2)
            ax.plot(epochs, results['val_accuracy'], 'r-', 
                   label='Val Accuracy', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'LR = {lr}', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 
                                 f'{optimizer_name}_accuracy_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(config.LEARNING_RATES))
    width = 0.2
    
    for idx, optimizer_name in enumerate(config.OPTIMIZERS):
        best_accs = [all_results[optimizer_name][lr]['best_val_acc'] 
                     for lr in config.LEARNING_RATES]
        offset = width * (idx - 1.5)
        ax.bar(x + offset, best_accs, width, label=optimizer_name.upper())
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Best Validation Accuracy by Optimizer and Learning Rate', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr}' for lr in config.LEARNING_RATES])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'best_accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = []
    for optimizer_name in config.OPTIMIZERS:
        row = [all_results[optimizer_name][lr]['final_val_acc'] 
               for lr in config.LEARNING_RATES]
        heatmap_data.append(row)
    
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.4f',
                xticklabels=[f'{lr}' for lr in config.LEARNING_RATES],
                yticklabels=[opt.upper() for opt in config.OPTIMIZERS],
                cmap='YlOrRd',
                cbar_kws={'label': 'Final Validation Accuracy'},
                ax=ax)
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_title('Final Validation Accuracy Heatmap', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'accuracy_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved!")