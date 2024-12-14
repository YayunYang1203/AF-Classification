# AF Classification with Bidirectional LSTM and Attention Mechanism

This repository provides a deep learning-based approach for detecting Atrial Fibrillation (AF) using a bidirectional LSTM model with an optional Attention mechanism. The model leverages Keras and TensorFlow to train on a provided dataset, achieving accurate results in AF classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Results Visualization](#results-visualization)
8. [Cross-Validation](#cross-validation)
9. [Usage](#usage)
10. [References](#references)

## Project Overview

This project applies a bidirectional LSTM model with an optional Attention mechanism to classify heartbeats in ECG data for detecting Atrial Fibrillation. The architecture involves:

- A **Bidirectional LSTM** layer for learning long-term dependencies.
- An **Attention Mechanism** to focus on important sequence parts.
- Dense and dropout layers to enhance model robustness.
  
The model is trained with categorical cross-entropy loss and can perform stratified 10-fold cross-validation to enhance performance.

## Installation

Ensure Python 3.6+ is installed along with the required libraries:

```bash
pip install numpy pandas matplotlib tensorflow keras scikit-learn
```

## Dataset Preparation

The code assumes the input ECG data is stored in `.npz` or `.csv` format. The main datasets include `x_train`, `y_train`, `x_test`, and `y_test`, where:
- `x_train` and `x_test` contain the ECG signal data.
- `y_train` and `y_test` contain the labels indicating the AF classification.

The code first loads this data into NumPy arrays for model training and validation.

## Model Architecture

The model consists of the following layers:

1. **Bidirectional LSTM Layer**: Captures temporal dependencies in both forward and backward directions with a hidden size of 200.
2. **Attention Layer** (optional): Learns to focus on crucial time steps in the sequence, improving interpretability and performance.
3. **Global Max Pooling Layer**: Reduces dimensionality and focuses on the most significant features.
4. **Dense Layers**: Fully connected layers for final classification.
5. **Dropout Layers**: Regularization layers to reduce overfitting.
6. **Output Layer**: `Softmax` activation for multiclass classification (for AF detection with three classes).

## Training the Model

### Hyperparameters

- **Epochs**: 300
- **Batch Size**: 4096
- **Learning Rate Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy for multiclass classification

### Training

The model is trained using:

```python
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[checkpointer])
```

The training is executed with the `ModelCheckpoint` callback to save the best-performing model weights.

## Evaluation

The model evaluates performance based on validation accuracy and loss. The best accuracy achieved during training is recorded, and final accuracy/loss values are saved for further analysis.

## Results Visualization

The script provides functionality to visualize training and validation accuracy and loss over epochs. It generates two plots:
1. **Training and Validation Accuracy**: Saved as `af_lstm_training_accuracy_<timestamp>.png`.
2. **Training and Validation Loss**: Saved as `af_lstm_training_loss_<timestamp>.png`.

```python
# accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'test'], loc='upper left')

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.legend(['train', 'test'], loc='upper right')
```

## Cross-Validation

For a more robust evaluation, 10-fold cross-validation is implemented using `StratifiedKFold` from `sklearn.model_selection`. The model resets weights after each fold, and the validation accuracy/loss for each fold is recorded.

```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

for train_index, test_index in kf.split(x_data, y_data):
    # Training and evaluation logic here...
```

## Usage

To use this code:

1. **Prepare the Dataset** in `.npz` or `.csv` format and specify the path in the script.
2. **Configure Hyperparameters** as desired.
3. **Run the script** to train the model and evaluate performance.

## References

- [Bidirectional LSTM Model]

This project was inspired by research in AF detection and time-series classification using LSTM models.
