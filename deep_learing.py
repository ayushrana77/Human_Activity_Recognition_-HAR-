# %% [code] {"id":"p4SVE3EXnADf"}
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import itertools
from sklearn.utils import class_weight

np.random.seed(42)
tf.random.set_seed(42)

# Importing libraries
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.layers import Dense, Dropout, Bidirectional, Input, SpatialDropout1D, Concatenate, Add, Reshape, GlobalAveragePooling1D
from keras.layers import SeparableConv1D, MultiHeadAttention, LayerNormalization, Attention
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Multiply

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


# %% [code] {"id":"86x0Xu5ln2wB"}
"""### Load Data"""

# Data directory for Kaggle environment
DATADIR = '/kaggle/input/har-uic'
SAVEDIR = '/kaggle/working'  # For saving output files

# Check if we're in Kaggle environment or local
import os
if not os.path.exists(DATADIR):
    print("Not in Kaggle environment, using default paths")
    DATADIR = '/content/drive/MyDrive/Colab Notebooks/UCI HAR Dataset'
    SAVEDIR = '.'

print(f"Using data directory: {DATADIR}")
print(f"Using save directory: {SAVEDIR}")

# Activity labels - updated to be 0-indexed for Kaggle compatibility
ACTIVITIES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
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

# %% [code] {"id":"Cw5WYKqBn0hv"}
# Utility function to read the data from csv file
def _read_csv(filename):
    try:
        return pd.read_csv(filename, delim_whitespace=True, header=None)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        # Try alternate format with comma delimiter
        try:
            return pd.read_csv(filename, header=None)
        except Exception as e2:
            print(f"Failed with alternate format: {e2}")
            raise

# Utility function to load the load
def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        # Try standard directory structure first
        filename = f'{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        if not os.path.exists(filename):
            # Try alternate directory structure (Kaggle might have different structure)
            filename = f'{DATADIR}/{subset}/Inertial_Signals/{signal}_{subset}.txt'
            if not os.path.exists(filename):
                print(f"Warning: Signal file not found at {filename}")
                continue
                
        signals_data.append(
            _read_csv(filename).to_numpy()
        )

    if len(signals_data) == 0:
        raise FileNotFoundError(f"No signal files found for {subset} subset")
        
    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

# %% [code] {"id":"peKubWa2nyrD"}
def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of 
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    filename = f'{DATADIR}/{subset}/y_{subset}.txt'
    if not os.path.exists(filename):
        # Try alternate filename format
        filename = f'{DATADIR}/y_{subset}.txt'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Label file not found at {filename}")
            
    y = _read_csv(filename)[0]
    
    # Convert to 0-indexed for Kaggle if needed
    if y.min() == 1:
        print("Converting labels from 1-indexed to 0-indexed")
        y = y - 1
    
    return to_categorical(y, num_classes=6), y.values

# %% [code] {"id":"f4KO-tYenxTw"}
def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    try:
        print("Loading data from", DATADIR)
        X_train, X_test = load_signals('train'), load_signals('test')
        y_train, y_train_raw = load_y('train')
        y_test, y_test_raw = load_y('test')
        
        print(f"Data loaded successfully: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw
    except Exception as e:
        print(f"Error loading data: {e}")
        # You could add fallback data loading here if needed
        raise

# %% [code] {"id":"jaIPbvtonv36"}
# Data augmentation functions
def add_gaussian_noise(X, sigma=0.01):
    """Add random Gaussian noise to the data."""
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def apply_random_time_shift(X, shift_percent=0.05):
    """Apply random time shifts to the data (±5% of window length)."""
    X_shifted = np.zeros_like(X)

    for i in range(len(X)):
        time_steps = X.shape[1]
        features = X.shape[2]
        max_shift = int(time_steps * shift_percent)

        # Random shift between -max_shift and max_shift
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Apply shift
        for j in range(features):
            if shift > 0:
                X_shifted[i, shift:, j] = X[i, :-shift, j]
                # Pad the beginning
                X_shifted[i, :shift, j] = X[i, 0, j]
            elif shift < 0:
                X_shifted[i, :shift, j] = X[i, -shift:, j]
                # Pad the end
                X_shifted[i, shift:, j] = X[i, -1, j]
            else:
                X_shifted[i, :, j] = X[i, :, j]

    return X_shifted

# %% [code] {"id":"0GorLO3fntbV"}
def augment_training_data(X, y):
    """Apply multiple augmentation techniques to training data."""
    # Original data
    X_augmented = [X]
    y_augmented = [y]

    # Add Gaussian noise
    X_noise = add_gaussian_noise(X, sigma=0.01)
    X_augmented.append(X_noise)
    y_augmented.append(y)

    # Apply random time shifts
    X_shifted = apply_random_time_shift(X, shift_percent=0.05)
    X_augmented.append(X_shifted)
    y_augmented.append(y)

    # Concatenate all augmented data
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)

    print(f"Original data shape: {X.shape}, Augmented data shape: {X_final.shape}")

    return X_final, y_final

# %% [code] {"id":"sa-QoRrTnrhK"}
# Calculate class weights for imbalanced data
def calculate_class_weights(y_raw):
    """Calculate inverse frequency weights to address class imbalance."""
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_raw),
        y=y_raw
    )
    return dict(enumerate(class_weights))

# %% [code] {"id":"8zCFisJBnpyJ"}
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

# %% [code] {"id":"vVlwRm6qnndj"}
# Temporal smoothing for predictions
def apply_temporal_smoothing(predictions, window_size=5):
    """
    Apply temporal smoothing to model predictions to reduce noise.
    Uses a sliding window average approach.
    """
    smoothed_predictions = np.zeros_like(predictions)

    # For each prediction, take the average of surrounding predictions
    for i in range(len(predictions)):
        # Define window bounds
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)

        # Calculate average within window
        window_preds = predictions[start:end]
        smoothed_predictions[i] = np.mean(window_preds, axis=0)

    return smoothed_predictions

# %% [code] {"id":"A-c5ayx_nl6p"}
# Post-processing for predictions
def post_process_predictions(predictions, threshold=0.5, smoothing_window=5):
    """
    Apply post-processing to raw predictions:
    1. Threshold probabilities
    2. Apply smoothing to reduce noise
    3. Return predicted class indices and labels
    """
    # Apply temporal smoothing
    smoothed_preds = apply_temporal_smoothing(predictions, smoothing_window)

    # Get predicted class indices
    pred_indices = np.argmax(smoothed_preds, axis=1)

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
    pred_labels = [ACTIVITIES[idx] for idx in pred_indices]

    return pred_indices, pred_labels

# %% [code] {"id":"nweMOxaRniNP"}
# Custom layers for attention mechanisms
class TemporalAttention(tf.keras.layers.Layer):
    """Custom attention mechanism for time series data."""
    def __init__(self, units=64, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.W1 = Dense(units, activation='tanh')
        self.W2 = Dense(1)

    def call(self, inputs):
        # Reshape to 2D for dense layer application
        x = tf.reshape(inputs, (-1, inputs.shape[2]))

        # Apply attention weights calculation
        x = self.W1(x)
        x = self.W2(x)

        # Reshape attention weights back to sequence form
        attention_weights = tf.reshape(x, (-1, inputs.shape[1], 1))
        attention_weights = tf.nn.softmax(attention_weights, axis=1)

        # Apply attention weights to the input
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        config = super().get_config()
        return config

# %% [code] {"id":"pSapUSx2ngff"}
# Visualization functions
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=None, figsize=(10, 8), normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'

    if cmap is None:
        cmap = plt.cm.Blues

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    return plt

# %% [code] {"id":"KK1_Na0anex3"}
def plot_classification_metrics(y_true, y_pred, class_names, figsize=(12, 8)):
    """
    Plot precision, recall, and F1 scores from classification results
    """
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prepare data
    metrics = [precision, recall, f1]
    titles = ['Precision', 'Recall', 'F1 Score']

    # Create bar charts
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].bar(np.arange(len(class_names)), metric, color='skyblue')
        axes[i].set_title(title, fontsize=16)
        axes[i].set_xticks(np.arange(len(class_names)))
        axes[i].set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
        axes[i].set_ylim(0, 1.0)
        axes[i].set_xlabel('Activity', fontsize=14)
        axes[i].set_ylabel('Score', fontsize=14)

        # Add value labels
        for j, v in enumerate(metric):
            axes[i].text(j, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    return fig

# %% [code] {"id":"ZhJrNbkWnciU"}
def plot_combined_metrics(y_true, y_pred, class_names, figsize=(12, 8)):
    """
    Plot a combined view of precision, recall, and F1 scores
    """
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    # Create figure
    plt.figure(figsize=figsize)

    # Set up positions for grouped bars
    x = np.arange(len(class_names))
    width = 0.25

    # Create bar groups
    plt.bar(x - width, precision, width, label='Precision', color='#5DA5DA')
    plt.bar(x, recall, width, label='Recall', color='#FAA43A')
    plt.bar(x + width, f1, width, label='F1 Score', color='#60BD68')

    # Add labels and titles
    plt.title('Classification Performance Metrics by Activity', fontsize=16)
    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt

# %% [code] {"id":"FaqLPahKoEv1"}
# Configure session settings
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

# Start a session
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

# %% [code] {"id":"Jo50wvbKnawZ"}


# Initializing parameters
epochs = 50
batch_size = 16
n_hidden = 32

# %% [code] {"id":"ToTiduVEnXFW"}


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

# Augment training data - Apply Gaussian noise and random time shifts
X_train_aug, Y_train_aug = augment_training_data(X_train, Y_train)

# Calculate class weights to address imbalance
class_weights = calculate_class_weights(Y_train_raw)
print("Class weights to address imbalance:", class_weights)

# Calculate and store normalization parameters for external data processing
train_mean = np.mean(X_train.reshape(-1, input_dim), axis=0)
train_std = np.std(X_train.reshape(-1, input_dim), axis=0)
np.save(f'{SAVEDIR}/train_normalization_params.npy', {'mean': train_mean, 'std': train_std})

# %% [code] {"id":"-QtLyCKJnRPR"}
"""## Defining LSTM Network model:"""

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    f'{SAVEDIR}/best_lstm_model.keras',
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


# %% [code] {"id":"24mCScKBnPAo"}
# Function to create LSTM model (for ensemble)
def create_lstm_with_attention():
    """Create optimized LSTM model for Kaggle environment"""
    inputs = Input(shape=(timesteps, input_dim))

    # First LSTM layer with increased units (128) and return_sequences=True
    x = Bidirectional(LSTM(128, return_sequences=True, 
                           kernel_regularizer=l2(0.001)))(inputs)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(0.5)(x)  # Spatial Dropout instead of regular Dropout

    # Second LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(0.001)))(x)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(0.5)(x)

    # Attention mechanism
    attn = TemporalAttention(64)(x)

    # Dense layer with ReLU activation and L2 regularization
    x = Dense(n_hidden, activation='relu', kernel_regularizer=l2(0.01))(attn)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layer with softmax activation
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model

# Print the summary of the LSTM model (model1)
print("\n=== LSTM Model (model1) Summary ===")
model1 = create_lstm_with_attention()
model1.summary()

# %% [code] {"id":"VetGP5EanMuL"}
# Create an ensemble of LSTM models
def create_lstm_ensemble(num_models=3):
    models = []
    for i in range(num_models):
        model = create_lstm_with_attention()
        models.append(model)
    return models

# Ensemble prediction function
def ensemble_predict(models, X):
    """Combine predictions from multiple models using averaging."""
    predictions = [model.predict(X) for model in models]
    # Average the predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
# Create ensemble of LSTM models
lstm_ensemble = create_lstm_ensemble(3)

# %% [code] {"id":"bQc5eDEwnLUS"}
# Train each model in the ensemble
lstm_histories = []
for i, model in enumerate(lstm_ensemble):
    print(f"\nTraining LSTM Ensemble Model {i+1}/{len(lstm_ensemble)}")

    # Train with early stopping, checkpoints, and reduce LR
    history = model.fit(
        X_train_aug,
        Y_train_aug,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        class_weight=class_weights,  # Use class weights to address imbalance
        callbacks=[early_stopping,
                   ModelCheckpoint(f'{SAVEDIR}/best_lstm_ensemble_{i}.keras', monitor='val_accuracy', save_best_only=True),
                   reduce_lr],
        verbose=1
    )
    lstm_histories.append(history)

    # Save model
    model.save(f'{SAVEDIR}/lstm_ensemble_model_{i}.keras')

# %% [code] {"id":"IKs_Dw-LnIza"}
# Evaluate ensemble
ensemble_predictions = ensemble_predict(lstm_ensemble, X_test)
lstm_y_pred_classes = np.argmax(ensemble_predictions, axis=1)
lstm_y_true_classes = np.argmax(Y_test, axis=1)
# Evaluate final ensemble performance
lstm_ensemble_accuracy = np.mean([np.equal(lstm_y_pred_classes, lstm_y_true_classes).astype(float)])
print(f"\nLSTM Ensemble Accuracy: {lstm_ensemble_accuracy:.4f}")

# %% [code] {"id":"q8IBtx1PnHZV"}
# Get activity names for visualization
activity_names = [ACTIVITIES[i] for i in range(n_classes)]

# Generate and display enhanced confusion matrix
cm = confusion_matrix(lstm_y_true_classes, lstm_y_pred_classes)
# Regular confusion matrix
plot_confusion_matrix(cm, activity_names, title='LSTM Ensemble Confusion Matrix', figsize=(12, 10))
plt.savefig(f'{SAVEDIR}/lstm_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"qVRgCU8anFm0"}
# Normalized confusion matrix
plot_confusion_matrix(cm, activity_names, title='LSTM Ensemble Confusion Matrix', figsize=(12, 10), normalize=True)
plt.savefig(f'{SAVEDIR}/lstm_ensemble_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"-apxpU7EnD8X"}
# Display text classification report
print("\nClassification Report:")
print(classification_report(lstm_y_true_classes, lstm_y_pred_classes, target_names=activity_names))

# %% [code] {"id":"VJmJ6AgenC6F"}
# Plot graphical classification report
plot_classification_metrics(lstm_y_true_classes, lstm_y_pred_classes, activity_names)
plt.savefig(f'{SAVEDIR}/lstm_ensemble_classification_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"GHju4SDCnByv"}
# Plot combined metrics
plot_combined_metrics(lstm_y_true_classes, lstm_y_pred_classes, activity_names)
plt.savefig(f'{SAVEDIR}/lstm_ensemble_combined_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"9EIgYxXamJGS"}
"""## Defining CNN-LSTM Network model:"""

# Reshape data for CNN-LSTM model
# Define n_steps and n_length for reshaping
n_steps = 4
n_length = timesteps//n_steps
print(f"Reshaping to n_steps={n_steps}, n_length={n_length}")

# Reshape X_train and X_test for CNN-LSTM
X_train_cnn = X_train.reshape((-1, n_steps, n_length, input_dim))
X_test_cnn = X_test.reshape((-1, n_steps, n_length, input_dim))

print(f"Reshaped data for CNN-LSTM: {X_train_cnn.shape}")

# %% [code] {"id":"5V8lUqw1mzva"}
def create_cnn_lstm_hierarchical():
    """Create optimized CNN-LSTM model for Kaggle environment"""
    # Set up model input shape
    inputs = Input(shape=(n_steps, n_length, input_dim))

    # First level: Efficient convolution with focused filters
    # Using SeparableConv1D for better parameter efficiency
    x = TimeDistributed(SeparableConv1D(filters=64, kernel_size=5, 
                                       activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(SpatialDropout1D(0.25))(x)  # Spatial dropout for better regularization

    # Second level: Dilated convolution for capturing longer patterns
    x = TimeDistributed(SeparableConv1D(filters=64, kernel_size=3, dilation_rate=2,
                                     activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(SpatialDropout1D(0.25))(x)

    # Feature extraction with pooling
    # Small scale features (local patterns)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    
    # Flatten features for LSTM processing
    x = TimeDistributed(Flatten())(x)

    # Simplified attention mechanism
    attention_layer = TemporalAttention(units=32)(x)
    x = Multiply()([x, attention_layer])

    # Use single bidirectional LSTM layer for temporal relationships
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x) 

    # Global pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model

# Print the summary of the CNN-LSTM model (model2)
print("\n=== CNN-LSTM Model (model2) Summary ===")
model2 = create_cnn_lstm_hierarchical()
model2.summary()

# %% [code] {"id":"d8pwDtjemsfA"}
# Create an ensemble of CNN-LSTM models
def create_cnn_lstm_ensemble(num_models=3):
    models = []
    for i in range(num_models):
        model = create_cnn_lstm_hierarchical()
        models.append(model)
    return models
# Create ensemble of CNN-LSTM models
cnn_lstm_ensemble = create_cnn_lstm_ensemble(3)

# Modify early stopping to be more patient for accuracy but with limits for free GPU
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Balanced patience for accuracy vs compute time
    restore_best_weights=True,
    verbose=1
)

# Improved learning rate scheduler for better convergence on limited compute
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# %% [code] {"id":"bLWbHmpVmn28"}
# More efficient augmentation that targets accuracy improvements
def advanced_augment_training_data(X, y):
    """Apply targeted data augmentation for CNN-LSTM model"""
    X_aug = X.copy()
    y_aug = y.copy()

    # Gaussian noise with strategic intensity
    X_noise = add_gaussian_noise(X, sigma=0.015)
    X_aug = np.vstack((X_aug, X_noise))
    y_aug = np.vstack((y_aug, y))

    # Strategic time shifts focusing on most relevant movement variations
    X_shift = apply_random_time_shift(X, shift_percent=0.06)
    X_aug = np.vstack((X_aug, X_shift))
    y_aug = np.vstack((y_aug, y))

    # Apply more focused augmentation for underrepresented classes
    class_counts = np.sum(y, axis=0)
    max_count = np.max(class_counts)

    for class_idx in range(len(class_counts)):
        if class_counts[class_idx] < max_count * 0.6:  # Target more classes for balanced learning
            # Find samples of this class
            class_indices = np.where(y[:, class_idx] == 1)[0]
            if len(class_indices) > 0:
                # Select a subset of samples for efficiency
                selection_size = min(len(class_indices), 20)  # Limit number of samples
                selected_indices = np.random.choice(class_indices, selection_size, replace=False)
                class_samples = X[selected_indices]
                class_labels = y[selected_indices]

                # Strategic noise addition for better generalization
                augmented = add_gaussian_noise(class_samples, sigma=0.02)
                X_aug = np.vstack((X_aug, augmented))
                y_aug = np.vstack((y_aug, class_labels))

    return X_aug, y_aug

# %% [code] {"id":"TG558oxfmlxu"}
# Train each model in the ensemble with enhanced augmentation
cnn_lstm_histories = []
for i, model in enumerate(cnn_lstm_ensemble):
    print(f"\nTraining Enhanced CNN-LSTM Ensemble Model {i+1}/{len(cnn_lstm_ensemble)}")

    # Apply advanced augmentation
    X_train_cnn_aug, Y_train_cnn_aug = advanced_augment_training_data(X_train_cnn, Y_train)

    # Define checkpoint for this model with efficiency settings for Colab
    checkpoint = ModelCheckpoint(
        f'{SAVEDIR}/best_cnn_lstm_ensemble_{i}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train with early stopping, checkpoints, and reduce LR
    history = model.fit(
        X_train_cnn_aug,
        Y_train_cnn_aug,
        batch_size=48,  # Balanced batch size for T4 GPU memory
        validation_data=(X_test_cnn, Y_test),
        epochs=epochs,  # Fewer epochs for Colab free GPU limit
        class_weight=calculate_class_weights(Y_train_raw),  # Use class weights to address imbalance
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    cnn_lstm_histories.append(history)

# %% [code] {"id":"2gsg8MTemj5b"}
# Evaluate ensemble
ensemble_predictions = ensemble_predict(cnn_lstm_ensemble, X_test_cnn)

# Apply temporal smoothing to ensemble predictions
smoothed_predictions = apply_temporal_smoothing(ensemble_predictions)
y_pred_classes = np.argmax(smoothed_predictions, axis=1)

# %% [code] {"id":"4PugsRbhmiSE"}
# Evaluate final ensemble performance
ensemble_accuracy = np.mean([np.equal(y_pred_classes, np.argmax(Y_test, axis=1)).astype(float)])
print(f"\nCNN-LSTM Ensemble Accuracy: {ensemble_accuracy:.4f}")

# %% [code] {"id":"QIxA7Q1Wmda-"}
# Generate and display enhanced confusion matrix for CNN-LSTM
cm = confusion_matrix(np.argmax(Y_test, axis=1), y_pred_classes)
# Regular confusion matrix
plot_confusion_matrix(cm, [ACTIVITIES[i] for i in range(n_classes)], title='CNN-LSTM Ensemble Confusion Matrix', figsize=(12, 10))
plt.savefig(f'{SAVEDIR}/cnn_lstm_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"-cAxyW0ymcYZ"}
# Normalized confusion matrix
plot_confusion_matrix(cm, [ACTIVITIES[i] for i in range(n_classes)], title='CNN-LSTM Ensemble Confusion Matrix', figsize=(12, 10), normalize=True)
plt.savefig(f'{SAVEDIR}/cnn_lstm_ensemble_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"W-DAtqOVmbNl"}
# Display text classification report
print("\nClassification Report:")
print(classification_report(np.argmax(Y_test, axis=1), y_pred_classes, target_names=[ACTIVITIES[i] for i in range(n_classes)]))

# %% [code] {"id":"9M4lsu_3mZV_"}
# Plot graphical classification report
plot_classification_metrics(np.argmax(Y_test, axis=1), y_pred_classes, [ACTIVITIES[i] for i in range(n_classes)])
plt.savefig(f'{SAVEDIR}/cnn_lstm_ensemble_classification_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"EEoJ5Ac6mYbe"}
# Plot combined metrics
plot_combined_metrics(np.argmax(Y_test, axis=1), y_pred_classes, [ACTIVITIES[i] for i in range(n_classes)])
plt.savefig(f'{SAVEDIR}/cnn_lstm_ensemble_combined_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] {"id":"bui-oVwsmVY7"}
def plot_model_history(histories, model_name):
    """
    Plots the training and validation accuracy and loss curves for ensemble models.

    Args:
        histories: List of training history objects returned by model.fit.
        model_name: The name of the model type (e.g., "LSTM", "CNN-LSTM").
    """
    # Check if histories list is empty
    if not histories:
        print(f"Warning: No training histories found for {model_name}")
        return

    # Ensure all histories have the expected keys
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    for i, h in enumerate(histories):
        if not all(key in h.history for key in required_keys):
            print(f"Warning: Model {i+1} history is missing required keys. Skipping plot for {model_name}.")
            return

    # Check if all histories have consistent lengths
    first_history_len = len(histories[0].history['accuracy'])
    if not all(len(h.history['accuracy']) == first_history_len for h in histories):
        print(f"Warning: Inconsistent history lengths for {model_name}. Adjusting to shortest length.")
        min_len = min(len(h.history['accuracy']) for h in histories)

        # Calculate average history using the minimum length
        avg_history = {
            'accuracy': np.mean([[h.history['accuracy'][i] for h in histories] for i in range(min_len)], axis=1),
            'val_accuracy': np.mean([[h.history['val_accuracy'][i] for h in histories] for i in range(min_len)], axis=1),
            'loss': np.mean([[h.history['loss'][i] for h in histories] for i in range(min_len)], axis=1),
            'val_loss': np.mean([[h.history['val_loss'][i] for h in histories] for i in range(min_len)], axis=1),
        }
    else:
        # Calculate average history as before
        avg_history = {
            'accuracy': np.mean([[h.history['accuracy'][i] for h in histories] for i in range(first_history_len)], axis=1),
            'val_accuracy': np.mean([[h.history['val_accuracy'][i] for h in histories] for i in range(first_history_len)], axis=1),
            'loss': np.mean([[h.history['loss'][i] for h in histories] for i in range(first_history_len)], axis=1),
            'val_loss': np.mean([[h.history['val_loss'][i] for h in histories] for i in range(first_history_len)], axis=1),
        }

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot training & validation accuracy values
    axs[0].plot(avg_history['accuracy'], linewidth=2, label='Avg Train')
    axs[0].plot(avg_history['val_accuracy'], linewidth=2, label='Avg Validation')

    # Plot individual model histories
    for i, hist in enumerate(histories):
        # Ensure we only plot up to the calculated length
        hist_len = len(hist.history['accuracy'])
        actual_len = min(hist_len, len(avg_history['accuracy']))

        axs[0].plot(hist.history['accuracy'][:actual_len], linewidth=1, alpha=0.3, linestyle='--', label=f'Model {i+1} Train')
        axs[0].plot(hist.history['val_accuracy'][:actual_len], linewidth=1, alpha=0.3, linestyle='--', label=f'Model {i+1} Val')

    axs[0].set_title(f'{model_name} Ensemble Model Accuracy', fontsize=16)
    axs[0].set_ylabel('Accuracy', fontsize=14)
    axs[0].set_xlabel('Epoch', fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot training & validation loss values
    axs[1].plot(avg_history['loss'], linewidth=2, label='Avg Train')
    axs[1].plot(avg_history['val_loss'], linewidth=2, label='Avg Validation')

    # Plot individual model histories
    for i, hist in enumerate(histories):
        # Ensure we only plot up to the calculated length
        hist_len = len(hist.history['loss'])
        actual_len = min(hist_len, len(avg_history['loss']))

        axs[1].plot(hist.history['loss'][:actual_len], linewidth=1, alpha=0.3, linestyle='--', label=f'Model {i+1} Train')
        axs[1].plot(hist.history['val_loss'][:actual_len], linewidth=1, alpha=0.3, linestyle='--', label=f'Model {i+1} Val')

    axs[1].set_title(f'{model_name} Ensemble Model Loss', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=14)
    axs[1].set_xlabel('Epoch', fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig.savefig(f'{SAVEDIR}/{model_name.lower().replace(" ", "_")}_ensemble_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [code] {"id":"CJ8MVKZ7mPNF"}
# Plot the training history graphs
plot_model_history(lstm_histories, "Improved LSTM")
plot_model_history(cnn_lstm_histories, "Improved CNN-LSTM")

# Model performance comparison
lstm_acc = lstm_ensemble_accuracy
cnn_lstm_acc = ensemble_accuracy

# Compare model performance with a bar chart
plt.figure(figsize=(10, 6))
models = ['LSTM Ensemble', 'CNN-LSTM Ensemble']
accuracies = [lstm_acc, cnn_lstm_acc]
# Create a gradient-colored bar chart
plt.bar(models, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8, width=0.6)
plt.title('Ensemble Model Accuracy Comparison', fontsize=16)
plt.ylabel('Test Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value labels
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'{SAVEDIR}/ensemble_model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()