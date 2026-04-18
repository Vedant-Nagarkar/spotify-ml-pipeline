
import numpy as np
import pandas as pd
import os
import joblib           # for saving/loading sklearn objects
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.logger import logger


# ── Output folder for processed data ──────────
# We'll save processed data here
PROCESSED_DIR = 'processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)



def load_and_clean(filepath='spotify.csv'):
    """
    Load and clean the spotify dataset.
    Same cleaning as 01_eda.py but returns
    a copy ready for ML preprocessing.

    Args:
        filepath: path to CSV file

    Returns:
        df: cleaned DataFrame
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'track_id',
                    'track_name', 'album_name', 'artists']
    cols_exist   = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_exist)
    logger.info(f"Dropped columns: {cols_exist}")

    # Drop missing values
    df = df.dropna()
    logger.info(f"Shape after cleaning: {df.shape}")

    return df



def engineer_features(df):
    """
    Create new features and encode categoricals.

    Why encode?
    → ML models only understand numbers
    → 'pop', 'rock' must become 45, 78

    Why create binary target?
    → popularity (0-100) → popular (0 or 1)
    → Converts regression → classification

    Args:
        df: cleaned DataFrame

    Returns:
        df: DataFrame with new features
        le: fitted LabelEncoder (saved for later use)
    """
    logger.info("Engineering features...")

    # ── Encode explicit (bool → int) ──────────
    # True  → 1 (has explicit lyrics)
    # False → 0 (no explicit lyrics)
    df['explicit'] = df['explicit'].astype(int)
    logger.info("Encoded: explicit (bool to int)")

    # ── Encode track_genre ────────────────────
    # 'pop' → 45, 'rock' → 78, etc.
    # LabelEncoder assigns alphabetical numbers
    le = LabelEncoder()
    df['track_genre_encoded'] = le.fit_transform(df['track_genre'])
    logger.info(f"Encoded: track_genre ({len(le.classes_)} genres to numbers)")

    # ── Create binary target ──────────────────
    # popularity >= 50 → popular (1)
    # popularity <  50 → not popular (0)
    df['popular'] = (df['popularity'] >= 50).astype(int)
    pop_pct = df['popular'].mean() * 100
    logger.info(f"Created target: popular (1={pop_pct:.1f}%, 0={100-pop_pct:.1f}%)")

    return df, le



def split_and_scale(df):
    """
    Split data into train/test and scale features.

    IMPORTANT ORDER:
    1. Split FIRST
    2. Fit scaler on TRAIN only
    3. Transform both train and test

    Why this order?
    → Fitting scaler on test data = data leakage!
    → Test data must be completely unseen

    Args:
        df: engineered DataFrame

    Returns:
        X_train, X_test: feature arrays
        y_train, y_test: label arrays
        scaler: fitted StandardScaler
    """
    logger.info("Splitting and scaling data...")

    # ── Define features ───────────────────────
    # These are the columns we'll use for ML
    # We exclude: popularity (used to make target)
    #             track_genre (we use encoded version)
    feature_cols = [
        'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness',
        'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms',
        'explicit', 'key', 'mode',
        'time_signature', 'track_genre_encoded'
    ]

    X = df[feature_cols].values  # .values → numpy array
    y = df['popular'].values

    logger.info(f"Features : {X.shape}")
    logger.info(f"Target   : {y.shape}")
    logger.info(f"Feature columns: {feature_cols}")

    # ── Train/Test Split ──────────────────────
    # test_size=0.2  → 20% for testing, 80% for training
    # random_state=42 → reproducible split
    # stratify=y     → maintain class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y    # important for imbalanced data!
    )

    logger.info(f"Train size: {X_train.shape[0]:,}")
    logger.info(f"Test size : {X_test.shape[0]:,}")

    # ── Scale Features ────────────────────────
    # StandardScaler: (x - mean) / std
    # After scaling: mean=0, std=1 for each feature
    # Why? Prevents features with large values
    #      (like duration_ms = 200,000)
    #      from dominating features with small values
    #      (like danceability = 0.7)
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)  # fit + transform
    X_test_sc   = scaler.transform(X_test)        # transform only!

    logger.info("Features scaled with StandardScaler")
    logger.info(f"Feature means (should be ~0): {X_train_sc.mean(axis=0).round(3)}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols



def save_processed_data(X_train, X_test, y_train, y_test,
                        scaler, le, feature_cols):
    """
    Save all processed data and fitted objects to disk.

    Why save?
    → Files 03-06 can load without reprocessing
    → Consistent data across all experiments
    → Can reload and retrain anytime

    .npy files → numpy arrays (fast to load)
    .pkl files → sklearn objects (scaler, encoder)

    Args:
        X_train, X_test: scaled feature arrays
        y_train, y_test: label arrays
        scaler: fitted StandardScaler
        le: fitted LabelEncoder
        feature_cols: list of feature names
    """
    logger.info(f"Saving processed data to {PROCESSED_DIR}/...")

    # Save numpy arrays
    # These are the actual data matrices
    np.save(f'{PROCESSED_DIR}/X_train.npy', X_train)
    np.save(f'{PROCESSED_DIR}/X_test.npy',  X_test)
    np.save(f'{PROCESSED_DIR}/y_train.npy', y_train)
    np.save(f'{PROCESSED_DIR}/y_test.npy',  y_test)
    logger.info("Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

    # Save sklearn objects using joblib
    # joblib is better than pickle for numpy arrays
    joblib.dump(scaler,       f'{PROCESSED_DIR}/scaler.pkl')
    joblib.dump(le,           f'{PROCESSED_DIR}/label_encoder.pkl')
    joblib.dump(feature_cols, f'{PROCESSED_DIR}/feature_cols.pkl')
    logger.info("Saved: scaler.pkl, label_encoder.pkl, feature_cols.pkl")



def load_processed_data():
    """
    Load saved processed data from disk.
    Used by files 03, 04, 05, 06 to load data
    without reprocessing.

    Returns:
        X_train, X_test, y_train, y_test: arrays
        scaler: fitted StandardScaler
        feature_cols: list of feature names
    """
    logger.info(f"Loading processed data from {PROCESSED_DIR}/...")

    X_train      = np.load(f'{PROCESSED_DIR}/X_train.npy')
    X_test       = np.load(f'{PROCESSED_DIR}/X_test.npy')
    y_train      = np.load(f'{PROCESSED_DIR}/y_train.npy')
    y_test       = np.load(f'{PROCESSED_DIR}/y_test.npy')
    scaler       = joblib.load(f'{PROCESSED_DIR}/scaler.pkl')
    feature_cols = joblib.load(f'{PROCESSED_DIR}/feature_cols.pkl')

    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test : {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load and clean
    df = load_and_clean('spotify.csv')

    # Step 2: Feature engineering
    df, le = engineer_features(df)

    # Step 3: Split and scale
    X_train, X_test, y_train, y_test, scaler, feature_cols = \
        split_and_scale(df)

    # Step 4: Save everything
    save_processed_data(
        X_train, X_test, y_train, y_test,
        scaler, le, feature_cols
    )

    logger.info("=" * 50)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info(f"Saved to: {PROCESSED_DIR}/")
    logger.info("=" * 50)
