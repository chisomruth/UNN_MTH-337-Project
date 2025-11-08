# Deep Learning Optimizer Comparison

Testing different optimizers (Adam, Adagrad, RMSprop, SGD) across multiple learning rates for CNN and MLP models.

## Projects

### 1. CNN - Appliance Detection
Image classification using convolutional neural networks.

### 2. MLP - Salary Prediction
Regression model for predicting salaries from employee data.

## Datasets

**CNN**: get from kaggle- https://www.kaggle.com/datasets/salmaneunus/mechanical-tools-dataset

**MLP**: get from kaggle-https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer


## Usage

```bash
# Run CNN experiments
cd CNN
python cnn_experiments.py

# Run MLP experiments
cd ..
cd MLP
python mlp_experiments.py
```

## The Experiment runs does the following:

- Tests 4 optimizers × 4 learning rates = 16 configurations
- Saves trained models and training history
- Generates comparison plots and summary reports
- Identifies best performing configuration

## Results

Results saved in `cnn_results/` and `mlp_results/`:
- Trained models (`.keras`)
- Training histories (`.pkl`)
- Visualization plots (`.png`)
- Summary report (`.csv`)

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- GPU recommended (but not required).

## Troubleshooting

**CNN**: Dataset needs at least 2 classes in separate folders

**MLP**: Make sure CSV file is in the right location (inside a Datasets folder in the root directory)

**GPU**: auto-detects cuda, falls back to CPU if unavailable
