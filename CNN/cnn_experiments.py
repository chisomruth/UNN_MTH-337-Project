import os
import numpy as np
import pandas as pd
import pickle
import random
import json
from datetime import datetime
import tensorflow as tf
from CNN.cnn_config import set_seed, Config
from CNN.cnn_utils import create_cnn_model, create_data_generators, train_model, plot_results

import warnings
warnings.filterwarnings("ignore")
set_seed(42)
config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def check_dataset_structure():
    print("\n" + "="*80)
    print("CHECKING DATASET STRUCTURE")
    print("="*80)
    
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {config.TRAIN_DIR}")
    if not os.path.exists(config.VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {config.VAL_DIR}")
    
    train_classes = [d for d in os.listdir(config.TRAIN_DIR) 
                     if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]
    val_classes = [d for d in os.listdir(config.VAL_DIR) 
                   if os.path.isdir(os.path.join(config.VAL_DIR, d))]
    
    print(f"\nTraining directory: {config.TRAIN_DIR}")
    print(f"Classes found: {len(train_classes)}")
    if train_classes:
        print(f"Class names: {train_classes}")
        for cls in train_classes:
            num_images = len([f for f in os.listdir(os.path.join(config.TRAIN_DIR, cls)) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  - {cls}: {num_images} images")
    
    print(f"\nValidation directory: {config.VAL_DIR}")
    print(f"Classes found: {len(val_classes)}")
    if val_classes:
        print(f"Class names: {val_classes}")
        for cls in val_classes:
            num_images = len([f for f in os.listdir(os.path.join(config.VAL_DIR, cls)) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  - {cls}: {num_images} images")
    
    if len(train_classes) < 2:
        print("\n" + "!"*80)
        print("WARNING: Dataset must have at least 2 classes for classification!")
        print("Your dataset structure should be:")
        print("  train_data_V2/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("!"*80)
        raise ValueError("Dataset must contain at least 2 classes")
    
    print("\n Dataset structure is valid")
    print("="*80)
    
    return train_classes, val_classes


def run_experiments():
    print("\n" + "="*80)
    print("CNN APPLIANCE DETECTION - LEARNING RATE STUDY")
    print("="*80)

    train_classes, val_classes = check_dataset_structure()

    print("\nLoading datasets...")
    train_gen, val_gen = create_data_generators(config.SEED)
    num_classes = train_gen.num_classes
    print(f"Detected {num_classes} classes")
    print(f" Training samples: {train_gen.samples}")
    print(f" Validation samples: {val_gen.samples}")

    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes, found {num_classes}")
    
    all_results = {}
    total_experiments = len(config.OPTIMIZERS) * len(config.LEARNING_RATES)
    experiment_count = 0
    
    for optimizer_name in config.OPTIMIZERS:
        all_results[optimizer_name] = {}
        for learning_rate in config.LEARNING_RATES:
            experiment_count += 1
            print(f"\n\n Experiment {experiment_count}/{total_experiments}")
            train_gen, val_gen = create_data_generators(config.SEED)
            
            history, model = train_model(
                optimizer_name, 
                learning_rate, 
                train_gen, 
                val_gen, 
                num_classes
            )
            all_results[optimizer_name][learning_rate] = {
                'train_loss': history['loss'],
                'val_loss': history['val_loss'],
                'train_accuracy': history['accuracy'],
                'val_accuracy': history['val_accuracy'],
                'final_train_loss': history['loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['accuracy'][-1],
                'final_val_acc': history['val_accuracy'][-1],
                'best_val_acc': max(history['val_accuracy']),
                'best_val_acc_epoch': np.argmax(history['val_accuracy']) + 1
            }
            tf.keras.backend.clear_session()
    results_path = os.path.join(config.RESULTS_DIR, 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n All results saved: {results_path}")
    return all_results

def generate_summary_report(all_results):
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_data = []
    
    for optimizer_name in config.OPTIMIZERS:
        for lr in config.LEARNING_RATES:
            results = all_results[optimizer_name][lr]
            summary_data.append({
                'Optimizer': optimizer_name.upper(),
                'Learning_Rate': lr,
                'Final_Train_Loss': results['final_train_loss'],
                'Final_Val_Loss': results['final_val_loss'],
                'Final_Train_Acc': results['final_train_acc'],
                'Final_Val_Acc': results['final_val_acc'],
                'Best_Val_Acc': results['best_val_acc'],
                'Best_Epoch': results['best_val_acc_epoch']
            })
    
    df_summary = pd.DataFrame(summary_data)

    csv_path = os.path.join(config.RESULTS_DIR, 'summary_report.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n Summary report saved: {csv_path}")

    print("\n" + df_summary.to_string(index=False))
 
    best_idx = df_summary['Best_Val_Acc'].idxmax()
    best_config = df_summary.iloc[best_idx]
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Optimizer: {best_config['Optimizer']}")
    print(f"Learning Rate: {best_config['Learning_Rate']}")
    print(f"Best Validation Accuracy: {best_config['Best_Val_Acc']:.4f}")
    print(f"Achieved at Epoch: {best_config['Best_Epoch']}")
    print(f"{'='*80}")
    
    return df_summary

if __name__ == "__main__":
    print("\n GPU Configuration:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("No GPU found, using CPU")
    
    try:
        all_results = run_experiments()
        plot_results(all_results)
        df_summary = generate_summary_report(all_results)
        
        print("\n All experiments completed successfully!")
        print(f" Results saved in: {config.RESULTS_DIR}/")
    
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("\nPlease check your dataset structure and try again.")