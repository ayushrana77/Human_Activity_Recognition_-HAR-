import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='keras')

np.random.seed(42)
tf.random.set_seed(42)

# Importing libraries
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.layers import Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2

"""### Load Data"""

# Data directory
DATADIR = '/content/drive/MyDrive/Colab Notebooks/UCI HAR Dataset'

# Activity labels
ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

# Raw data signals
# Signals are from Accelerometer and Gyroscope
# The signals are in x,y,z directions
# Sensor signals are filtered to have only body acceleration
# excluding the acceleration due to gravity
# Triaxial acceleration from the accelerometer is total acceleration
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

# Utility function to read the data from csv file
def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

# Utility function to load the load
def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).to_numpy()
        )

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    filename = f'{DATADIR}/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).to_numpy(), y.values

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test, y_train_raw, y_test_raw
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_train_raw = load_y('train')
    y_test, y_test_raw = load_y('test')

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw

# Enhanced CSV data processing for external data
def process_csv_data(file_path, train_mean=None, train_std=None):
    """
    Process external CSV data for inference
    Returns normalized data in the correct shape for the model
    """
    # Load CSV data with actual columns
    df = pd.read_csv(file_path)

    # Check for required columns
    required_columns = ['gFx', 'gFy', 'gFz', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Map CSV columns to required features
    raw_data = np.hstack([
        df[['gFx', 'gFy', 'gFz']].values,  # Body acceleration
        df[['wx', 'wy', 'wz']].values,     # Gyroscope
        df[['ax', 'ay', 'az']].values      # Total acceleration
    ])

    # Normalize if mean and std are provided
    if train_mean is not None and train_std is not None:
        raw_data = (raw_data - train_mean) / train_std

    # Create windows of 128 timesteps
    windows = []
    for i in range(0, len(raw_data) - 127, 32):  # Stride of 32 for less overlap
        window = raw_data[i:i+128]
        if len(window) == 128:  # Ensure complete window
            windows.append(window)

    if not windows:
        raise ValueError("Not enough data to create windows")

    return np.array(windows)

# Post-processing for predictions
def post_process_predictions(predictions, threshold=0.5, smoothing_window=5):
    """
    Apply post-processing to raw predictions:
    1. Threshold probabilities
    2. Apply smoothing to reduce noise
    3. Return predicted class indices and labels
    """
    # Get predicted class indices
    pred_indices = np.argmax(predictions, axis=1)

    # Apply smoothing with a sliding window
    if len(pred_indices) >= smoothing_window:
        smoothed_indices = np.zeros_like(pred_indices)
        for i in range(len(pred_indices)):
            start = max(0, i - smoothing_window // 2)
            end = min(len(pred_indices), i + smoothing_window // 2 + 1)
            window = pred_indices[start:end]
            # Most common class in window
            values, counts = np.unique(window, return_counts=True)
            smoothed_indices[i] = values[np.argmax(counts)]
        pred_indices = smoothed_indices

    # Map indices to activity labels (add 1 because activities are 1-indexed)
    pred_labels = [ACTIVITIES[idx + 1] for idx in pred_indices]

    return pred_indices, pred_labels

# Configure session settings
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

# Start a session
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

# Initializing parameters
epochs = 50
batch_size = 16
n_hidden = 32

# Utility function to count the number of classes
def _count_classes(y):
    return len(set([tuple(category) for category in y]))

# Loading the train and test data
X_train, X_test, Y_train, Y_test, Y_train_raw, Y_test_raw = load_data()

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)
print(f"Timesteps: {timesteps}")
print(f"Input dimensions: {input_dim}")
print(f"Number of training samples: {len(X_train)}")
print(f"Shape of training data: {X_train.shape}")

# Calculate and store normalization parameters for external data processing
train_mean = np.mean(X_train.reshape(-1, input_dim), axis=0)
train_std = np.std(X_train.reshape(-1, input_dim), axis=0)
np.save('train_normalization_params.npy', {'mean': train_mean, 'std': train_std})

"""## Defining LSTM Network model:"""

"""## Defining LSTM Network model:"""

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_lstm_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Add ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Initializing the sequential model with improved architecture
model1 = Sequential()

# First LSTM layer with increased units (64) and return_sequences=True
model1.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, input_dim)))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))  # Dropout after first LSTM layer

# Second LSTM layer
model1.add(Bidirectional(LSTM(n_hidden)))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))

# Dense layer with ReLU activation and L2 regularization
model1.add(Dense(n_hidden, activation='relu', kernel_regularizer=l2(0.01)))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))

# Output layer with softmax activation
model1.add(Dense(n_classes, activation='softmax'))
model1.summary()

# Compiling the model with Adam optimizer and learning rate of 0.001
model1.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Training the model with early stopping, checkpoints, and reduce LR
lstm_history = model1.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Testing
score = model1.evaluate(X_test, Y_test)
print("Accuracy: ", score[1])
print("Loss: ", score[0])

# Generate predictions for confusion matrix
y_pred = model1.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(Y_test, axis=1)

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Display classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes,
                          target_names=[ACTIVITIES[i+1] for i in range(n_classes)]))

"""## Defining CNN-LSTM Network model:"""

"""## Defining CNN-LSTM Network model:"""

# Reshape data for CNN-LSTM model
n_steps, n_length = 4, 32
X_train_cnn = X_train.reshape((X_train.shape[0], n_steps, n_length, input_dim))
X_test_cnn = X_test.reshape((X_test.shape[0], n_steps, n_length, input_dim))

print(f"Reshaped data for CNN-LSTM: {X_train_cnn.shape}")

# Define model with improved architecture
model2 = Sequential()

# First Conv1D layer with increased filters (128) and larger kernel size (5)
model2.add(TimeDistributed(Conv1D(filters=128, kernel_size=5, activation='relu'),
                         input_shape=(None, n_length, input_dim)))
model2.add(TimeDistributed(BatchNormalization()))
model2.add(TimeDistributed(Dropout(0.5)))

# Second Conv1D layer with 64 filters
model2.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model2.add(TimeDistributed(BatchNormalization()))
model2.add(TimeDistributed(Dropout(0.5)))

model2.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model2.add(TimeDistributed(Dropout(0.5)))  # Dropout after MaxPooling

model2.add(TimeDistributed(Flatten()))
model2.add(BatchNormalization())

# First LSTM layer with return_sequences=True
model2.add(Bidirectional(LSTM(64, return_sequences=True)))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))

# Second LSTM layer
model2.add(Bidirectional(LSTM(n_hidden)))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))

# Dense layer with ReLU activation and L2 regularization
model2.add(Dense(n_hidden, activation='relu', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))

# Output layer with softmax activation
model2.add(Dense(n_classes, activation='softmax'))
model2.summary()

# Compile model with Adam optimizer and learning rate of 0.001
model2.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Define additional callbacks for CNN-LSTM
cnn_lstm_checkpoint = ModelCheckpoint(
    'best_cnn_lstm_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Training the model with all callbacks
cnn_lstm_history = model2.fit(
    X_train_cnn,
    Y_train,
    batch_size=batch_size,
    validation_data=(X_test_cnn, Y_test),
    epochs=epochs,
    callbacks=[early_stopping, cnn_lstm_checkpoint, reduce_lr]
)

# Testing
score = model2.evaluate(X_test_cnn, Y_test)
print("Accuracy: ", score[1])
print("Loss: ", score[0])

# Generate predictions for confusion matrix
y_pred = model2.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Display classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes,
                          target_names=[ACTIVITIES[i+1] for i in range(n_classes)]))

import matplotlib.pyplot as plt

def plot_model_history(history, model_name):
    """
    Plots the training and validation accuracy and loss curves for a given model.

    Args:
        history: The training history object returned by model.fit.
        model_name: The name of the model (e.g., "LSTM", "CNN-LSTM").
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title(f'{model_name} Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title(f'{model_name} Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Plot the graphs
plot_model_history(lstm_history, "Improved Bidirectional LSTM")
plot_model_history(cnn_lstm_history, "Improved Bidirectional CNN-LSTM")

# Save models in multiple formats
model1.save('improved_lstm_model.keras')
model1.save('improved_lstm.keras')

model2.save('improved_cnn_lstm_model.keras')
model2.save('improved_cnn_lstm.keras')

# Save learning curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'], label='LSTM Train')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation')
plt.plot(cnn_lstm_history.history['accuracy'], label='CNN-LSTM Train')
plt.plot(cnn_lstm_history.history['val_accuracy'], label='CNN-LSTM Validation')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='LSTM Train')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
plt.plot(cnn_lstm_history.history['loss'], label='CNN-LSTM Train')
plt.plot(cnn_lstm_history.history['val_loss'], label='CNN-LSTM Validation')
plt.title('Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
