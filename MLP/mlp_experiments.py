import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from MLP.mlp_config import Config, set_seed
import tensorflow as tf
from MLP.mlp_utils import load_and_preprocess_data, create_mlp_model, train_model, plot_results
import warnings


warnings.filterwarnings("ignore")
set_seed(42)
config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def run_experiments():
    print("\n" + "="*80)
    print("MLP SALARY PREDICTION - LEARNING RATE STUDY")
    print("="*80)
    
    X_train, X_test, y_train, y_test, input_dim, y_scaler = load_and_preprocess_data()
    
    all_results = {}

    total_experiments = len(config.OPTIMIZERS) * len(config.LEARNING_RATES)
    experiment_count = 0
    
    for optimizer_name in config.OPTIMIZERS:
        all_results[optimizer_name] = {}
        
        for learning_rate in config.LEARNING_RATES:
            experiment_count += 1
            print(f"\n\n Experiment {experiment_count}/{total_experiments}")

            history, model, test_metrics = train_model(
                optimizer_name, 
                learning_rate, 
                X_train, y_train,
                X_test, y_test,
                input_dim,
                y_scaler
            )
            
            all_results[optimizer_name][learning_rate] = {
                'train_loss': history['loss'],
                'val_loss': history['val_loss'],
                'train_mae': history['mae'],
                'val_mae': history['val_mae'],
                'final_train_loss': history['loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_mae': history['mae'][-1],
                'final_val_mae': history['val_mae'][-1],
                'best_val_loss': min(history['val_loss']),
                'best_val_loss_epoch': np.argmin(history['val_loss']) + 1,
                'test_mse': test_metrics['mse'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'y_test': test_metrics['y_test'],
                'y_pred': test_metrics['y_pred']
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
                'Best_Val_Loss': results['best_val_loss'],
                'Best_Epoch': results['best_val_loss_epoch'],
                'Test_MSE': results['test_mse'],
                'Test_RMSE': results['test_rmse'],
                'Test_MAE': results['test_mae'],
                'Test_R2': results['test_r2']
            })
    
    df_summary = pd.DataFrame(summary_data)

    df_summary['Final_Train_Loss'] = df_summary['Final_Train_Loss'].apply(lambda x: f'{x:,.2f}')
    df_summary['Final_Val_Loss'] = df_summary['Final_Val_Loss'].apply(lambda x: f'{x:,.2f}')
    df_summary['Best_Val_Loss'] = df_summary['Best_Val_Loss'].apply(lambda x: f'{x:,.2f}')
    df_summary['Test_MSE'] = df_summary['Test_MSE'].apply(lambda x: f'{x:,.2f}')
    df_summary['Test_RMSE'] = df_summary['Test_RMSE'].apply(lambda x: f'{x:,.2f}')
    df_summary['Test_MAE'] = df_summary['Test_MAE'].apply(lambda x: f'{x:,.2f}')
    df_summary['Test_R2'] = df_summary['Test_R2'].apply(lambda x: f'{x:.4f}')

    csv_path = os.path.join(config.RESULTS_DIR, 'summary_report.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n Summary report saved: {csv_path}")

    print("\n" + df_summary.to_string(index=False))

    df_numeric = pd.DataFrame(summary_data)
    best_idx = df_numeric['Test_R2'].idxmax()
    best_config = df_numeric.iloc[best_idx]
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION (Highest R²)")
    print(f"{'='*80}")
    print(f"Optimizer: {best_config['Optimizer']}")
    print(f"Learning Rate: {best_config['Learning_Rate']}")
    print(f"Test R² Score: {best_config['Test_R2']:.4f}")
    print(f"Test RMSE: ${best_config['Test_RMSE']:,.2f}")
    print(f"Test MAE: ${best_config['Test_MAE']:,.2f}")
    print(f"Best Validation Loss Epoch: {best_config['Best_Epoch']}")
    print(f"{'='*80}")
    
    return df_summary

if __name__ == "__main__":
    print("\n GPU Configuration:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f" - {gpu}")
    else:
        print(" No GPU found, using CPU")

    all_results = run_experiments()

    plot_results(all_results)

    df_summary = generate_summary_report(all_results)
    
    print("\n All experiments completed successfully!")
    print(f" Results saved in: {config.RESULTS_DIR}/")