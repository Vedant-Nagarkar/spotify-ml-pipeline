import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (f1_score, roc_auc_score,
                            accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

from logger import logger
from preprocessing import load_processed_data

# ── Settings ──────────────────────────────────
plt.style.use('seaborn-v0_8')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)



def build_model(input_dim):
    """
    Build a deep neural network for binary classification.

    Architecture explanation:
    - Dense(128): first hidden layer, 128 neurons
    - BatchNormalization: normalizes layer inputs
                        speeds up training
                        reduces sensitivity to init
    - Activation('relu'): non-linearity
                        avoids vanishing gradient
    - Dropout(0.3): randomly drops 30% of neurons
                    prevents overfitting

    Why this order? Dense -> BN -> Activation -> Dropout
    This is the recommended order for best performance.

    Args:
        input_dim: number of input features (15)

    Returns:
        model: compiled Keras model
    """
    model = keras.Sequential([

        # ── Hidden Layer 1 ────────────────────
        layers.Dense(128, input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # ── Hidden Layer 2 ────────────────────
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # ── Hidden Layer 3 ────────────────────
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        # ── Output Layer ──────────────────────
        # sigmoid squashes output to 0-1
        # represents probability of being popular
        layers.Dense(1, activation='sigmoid')

    ], name='SpotifyPopularityNN')

    return model



def get_callbacks():
    """
    Define training callbacks for smart training.

    Callbacks are functions called at each epoch.
    They allow us to:
    - Stop training early if not improving
    - Reduce learning rate when stuck
    - Save best model automatically

    EarlyStopping:
    - Monitors validation AUC
    - Stops if no improvement for 8 epochs
    - Restores best weights automatically
    - Prevents overfitting!

    ReduceLROnPlateau:
    - Monitors validation AUC
    - If no improvement for 4 epochs
    - Reduces learning rate by half
    - Helps escape local minima

    Returns:
        callbacks: list of callback objects
    """
    callbacks = [
        # Stop training if val_auc doesn't improve
        keras.callbacks.EarlyStopping(
            monitor              = 'val_auc',
            patience             = 8,
            restore_best_weights = True,
            mode                 = 'max',
            verbose              = 1
        ),

        # Reduce LR if val_auc plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_auc',
            factor   = 0.5,      # multiply LR by 0.5
            patience = 4,
            mode     = 'max',
            verbose  = 1,
            min_lr   = 1e-6      # never go below this LR
        )
    ]
    return callbacks


# ═══════════════════════════════════════════════
# FUNCTION 3: COMPUTE CLASS WEIGHTS
# ═══════════════════════════════════════════════
def get_class_weights(y_train):
    """
    Compute class weights to handle imbalance.

    Our dataset: 74% not popular, 26% popular
    Without weights: model ignores minority class

    compute_class_weight('balanced'):
    weight = total_samples / (n_classes * samples_in_class)

    For popular class (26%):
    weight = 113999 / (2 * 29367) = ~1.94
    (model pays 1.94x more attention to popular songs)

    For not popular class (74%):
    weight = 113999 / (2 * 84632) = ~0.67

    Args:
        y_train: training labels

    Returns:
        class_weight_dict: dict {0: weight, 1: weight}
    """
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes      = np.unique(y_train),
        y            = y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    return class_weight_dict



def train_model(X_train, y_train):
    """
    Build, compile and train the neural network.

    Adam optimizer:
    - Adaptive learning rate optimizer
    - Combines momentum + RMSprop
    - Best default choice for most problems

    binary_crossentropy loss:
    - Standard loss for binary classification
    - Works with sigmoid output

    AUC metric:
    - Monitor ROC-AUC during training
    - Better metric than accuracy for imbalanced data

    Args:
        X_train: scaled training features
        y_train: training labels

    Returns:
        model: trained Keras model
        history: training history object
    """
    logger.info("Building neural network...")
    logger.info(f"TensorFlow version: {tf.__version__}")

    input_dim = X_train.shape[1]
    model     = build_model(input_dim)

    # Print model architecture
    model.summary()

    # Compile model
    model.compile(
        optimizer = keras.optimizers.Adam(
                        learning_rate=0.001),
        loss      = 'binary_crossentropy',
        metrics   = [
            'accuracy',
            keras.metrics.AUC(name='auc')
        ]
    )

    # Get callbacks and class weights
    callbacks         = get_callbacks()
    class_weight_dict = get_class_weights(y_train)

    logger.info("Training neural network...")
    logger.info("(EarlyStopping will stop if no improvement)")

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs           = 100,
        batch_size       = 256,
        validation_split = 0.1,   # 10% of train for validation
        class_weight     = class_weight_dict,
        callbacks        = callbacks,
        verbose          = 1
    )

    logger.info(f"Training stopped at epoch: {len(history.history['loss'])}")

    return model, history



def evaluate_model(model, X_test, y_test):
    """
    Evaluate neural network on test data.

    Args:
        model: trained Keras model
        X_test: scaled test features
        y_test: test labels

    Returns:
        nn_results: dict with metrics
    """
    logger.info("Evaluating neural network...")

    # Get predictions
    # model.predict() returns probabilities (0.0 to 1.0)
    # .flatten() converts [[0.8],[0.3]] to [0.8, 0.3]
    y_pred_prob = model.predict(X_test).flatten()

    # Convert probabilities to labels
    # threshold = 0.5: above = popular, below = not popular
    y_pred = (y_pred_prob >= 0.5).astype(int)

    nn_results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1'      : f1_score(y_test, y_pred),
        'roc_auc' : roc_auc_score(y_test, y_pred_prob)
    }

    logger.info("NEURAL NETWORK RESULTS:")
    logger.info(f"  Accuracy : {nn_results['accuracy']:.4f}")
    logger.info(f"  F1 Score : {nn_results['f1']:.4f}")
    logger.info(f"  ROC-AUC  : {nn_results['roc_auc']:.4f}")

    return nn_results



def plot_training_history(history):
    """
    Plot training and validation curves.

    Two plots:
    1. Accuracy over epochs (train vs val)
    2. Loss over epochs (train vs val)

    Good training signs:
    - Both curves go down/up together
    - No big gap between train and val
    - Smooth curves

    Overfitting signs:
    - Train keeps improving
    - Val starts getting worse

    Args:
        history: Keras History object
    """
    logger.info("Plotting training history...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: AUC
    axes[0].plot(history.history['auc'],
                label='Train', color='blue', linewidth=2)
    axes[0].plot(history.history['val_auc'],
                label='Validation',
                color='orange', linewidth=2)
    axes[0].set_title('Neural Network AUC')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AUC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Loss
    axes[1].plot(history.history['loss'],
                label='Train', color='blue', linewidth=2)
    axes[1].plot(history.history['val_loss'],
                label='Validation',
                color='orange', linewidth=2)
    axes[1].set_title('Neural Network Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Neural Network Training History',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/12_nn_training.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/12_nn_training.png")



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING NEURAL NETWORK PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test, \
        scaler, feature_cols = load_processed_data()

    # Step 2: Train model
    model, history = train_model(X_train, y_train)

    # Step 3: Evaluate
    nn_results = evaluate_model(model, X_test, y_test)

    # Step 4: Plot training history
    plot_training_history(history)

    # Step 5: Compare with best classical model
    logger.info("=" * 50)
    logger.info("COMPARISON WITH BEST CLASSICAL MODEL")
    logger.info("=" * 50)

    # Load classical results
    all_results = joblib.load('models/all_results.pkl')
    best_name   = joblib.load('models/best_model_name.pkl')

    logger.info(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>10}")
    logger.info("-" * 45)
    for name, r in all_results.items():
        logger.info(
            f"{name:<25} {r['f1']:>8.4f} {r['roc_auc']:>10.4f}"
        )
    logger.info(
        f"{'Neural Network':<25} "
        f"{nn_results['f1']:>8.4f} "
        f"{nn_results['roc_auc']:>10.4f}"
    )

    # Step 6: Save neural network
    model.save('models/neural_network_model')
    joblib.dump(nn_results, 'models/nn_results.pkl')
    logger.info("Saved: models/neural_network_model/")
    logger.info("Saved: models/nn_results.pkl")

    logger.info("=" * 50)
    logger.info("NEURAL NETWORK PIPELINE COMPLETE!")
    logger.info("=" * 50)