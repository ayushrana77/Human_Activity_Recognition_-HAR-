import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure session settings for reproducibility
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.compat.v1.set_random_seed(42)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Load the previously saved data
try:
    # Try to load the preprocessed data if it exists
    X_train_cnn = np.load('X_train_cnn.npy')
    X_test_cnn = np.load('X_test_cnn.npy')
    Y_train_aug = np.load('Y_train_aug.npy')
    Y_test = np.load('Y_test.npy')
    class_weights = np.load('class_weights.npy', allow_pickle=True).item()
    print("Loaded saved data successfully")
except:
    print("Unable to load saved data. Please run the main script first to prepare the data.")
    exit(1)

# Safe plot_model_history function with error handling
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
    fig.savefig(f'{model_name.lower().replace(" ", "_")}_ensemble_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Data augmentation function for enhancing CNN-LSTM training
def add_gaussian_noise(X, sigma=0.01):
    """Add random Gaussian noise to the data."""
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def apply_random_time_shift(X, shift_percent=0.05):
    """Apply random time shifts to the data."""
    n_samples, n_timesteps, n_features = X.shape if len(X.shape) == 3 else (X.shape[0], X.shape[1], 1)
    shifted_X = np.zeros_like(X)
    
    for i in range(n_samples):
        # Calculate random shift amount (positive or negative)
        max_shift = int(n_timesteps * shift_percent)
        if max_shift == 0:
            shift = 0
        else:
            shift = np.random.randint(-max_shift, max_shift + 1)
        
        sample = X[i].copy()
        
        if shift > 0:
            # Shift right (forward in time)
            shifted_X[i, shift:, :] = sample[:-shift, :]
            # Repeat first values
            shifted_X[i, :shift, :] = np.tile(sample[0:1, :], (shift, 1))
        elif shift < 0:
            # Shift left (backward in time)
            shift = abs(shift)
            shifted_X[i, :-shift, :] = sample[shift:, :]
            # Repeat last values
            shifted_X[i, -shift:, :] = np.tile(sample[-1:, :], (shift, 1))
        else:
            # No shift
            shifted_X[i] = sample
            
    return shifted_X

def advanced_augment_training_data(X, y):
    """Apply enhanced data augmentation for CNN-LSTM model"""
    X_aug = X.copy()
    y_aug = y.copy()
    
    # Add Gaussian noise with varying intensities
    for sigma in [0.01, 0.02, 0.03]:
        X_noise = add_gaussian_noise(X, sigma=sigma)
        X_aug = np.vstack((X_aug, X_noise))
        y_aug = np.vstack((y_aug, y))
    
    # Apply time shifts with varying intensities
    for shift in [0.05, 0.08, 0.1]:
        X_shift = apply_random_time_shift(X, shift_percent=shift)
        X_aug = np.vstack((X_aug, X_shift))
        y_aug = np.vstack((y_aug, y))
    
    # Apply magnitude scaling (simulate different sensor sensitivities)
    for scale in [0.9, 1.1]:
        X_scale = X * scale
        X_aug = np.vstack((X_aug, X_scale))
        y_aug = np.vstack((y_aug, y))
    
    # Apply more augmentation for underrepresented classes
    class_counts = np.sum(y, axis=0)
    max_count = np.max(class_counts)
    
    for class_idx in range(len(class_counts)):
        if class_counts[class_idx] < max_count * 0.5:  # Heavily underrepresented
            # Find samples of this class
            class_indices = np.where(y[:, class_idx] == 1)[0]
            if len(class_indices) > 0:
                # Additional augmentation for underrepresented classes
                class_samples = X[class_indices]
                class_labels = y[class_indices]
                
                # Create more variations
                for _ in range(3):  # Create more samples for underrepresented classes
                    # Combine multiple augmentations
                    augmented = add_gaussian_noise(apply_random_time_shift(class_samples), sigma=0.02)
                    X_aug = np.vstack((X_aug, augmented))
                    y_aug = np.vstack((y_aug, class_labels))
    
    return X_aug, y_aug

# Ensemble prediction function
def ensemble_predict(models, X):
    """Combine predictions from multiple models using averaging."""
    predictions = np.array([model.predict(X) for model in models])
    return np.mean(predictions, axis=0)

# Modify early stopping to be more patient
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

# Learning rate scheduler with warm restarts
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Load or create ensemble of CNN-LSTM models
try:
    # Try to load existing models
    cnn_lstm_ensemble = [load_model(f'enhanced_cnn_lstm_ensemble_model_{i}.keras') for i in range(3)]
    print("Loaded existing CNN-LSTM ensemble models")
except:
    # If models don't exist, we would need to create them
    # This would require importing all the model definition code
    print("Could not load existing models. Please run with updated.py first.")
    exit(1)

# Train each model in the ensemble with enhanced augmentation
print("Training enhanced CNN-LSTM models to improve accuracy beyond 91%...")
cnn_lstm_histories = []

for i, model in enumerate(cnn_lstm_ensemble):
    print(f"\nTraining Enhanced CNN-LSTM Ensemble Model {i+1}/{len(cnn_lstm_ensemble)}")
    
    # Apply advanced augmentation
    X_train_cnn_aug, Y_train_cnn_aug = advanced_augment_training_data(X_train_cnn, Y_train_aug)
    print(f"Augmented training data shape: {X_train_cnn_aug.shape}")
    
    # Define checkpoint for this model
    checkpoint = ModelCheckpoint(
        f'best_cnn_lstm_ensemble_{i}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train with early stopping, checkpoints, and reduce LR
    history = model.fit(
        X_train_cnn_aug,
        Y_train_cnn_aug,
        batch_size=32,  # Smaller batch size for better generalization
        validation_data=(X_test_cnn, Y_test),
        epochs=50,  # Allow more epochs with early stopping
        class_weight=class_weights,  # Use class weights to address imbalance
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    cnn_lstm_histories.append(history)
    
    # Save model
    model.save(f'enhanced_cnn_lstm_ensemble_model_{i}.keras')

# Evaluate ensemble
ensemble_predictions = ensemble_predict(cnn_lstm_ensemble, X_test_cnn)
y_pred_classes = np.argmax(ensemble_predictions, axis=1)
y_true_classes = np.argmax(Y_test, axis=1)

# Calculate and display ensemble accuracy
ensemble_accuracy = np.mean(np.equal(y_pred_classes, y_true_classes).astype(float))
print(f"\nImproved CNN-LSTM Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Plot the training history
if cnn_lstm_histories:
    plot_model_history(cnn_lstm_histories, "Improved CNN-LSTM")

print("Completed training and evaluation of enhanced CNN-LSTM model")
