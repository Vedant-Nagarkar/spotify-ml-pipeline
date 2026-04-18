import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI pop-ups)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.cluster import KMeans

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "spotify.csv"
OUTPUT_DIR     = "outputs"
POPULARITY_THRESHOLD = 50   # same threshold used in notebook
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_CLUSTERS     = 5          # K-Means

ALL_STEPS = ["eda", "preprocessing", "clustering", "classification", "neural_network", "evaluation"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: EDA ────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    log("── EDA ──────────────────────────────────────")

    print(f"  Shape      : {df.shape}")
    print(f"  Missing    : {df.isnull().sum().sum()}")
    print(f"  Duplicates : {df.duplicated().sum()}")

    # Drop missing rows (3 rows in original dataset)
    df = df.dropna()
    log(f"  Shape after dropna: {df.shape}")

    # Create popularity label
    df["popular"] = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)
    class_dist = df["popular"].value_counts(normalize=True) * 100
    log(f"  Class balance → Popular: {class_dist[1]:.1f}%  Not Popular: {class_dist[0]:.1f}%")

    # Top genres by avg popularity
    top_genres = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\n  Top 5 Genres by Avg Popularity:")
    for genre, score in top_genres.items():
        print(f"    {genre:<15} {score:.2f}")

    # Save correlation heatmap
    audio_features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[audio_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=120)
    plt.close()
    log(f"  Saved → {OUTPUT_DIR}/eda_correlation.png")

    return df


# ── Step 2: Preprocessing ──────────────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame):
    log("── Preprocessing ────────────────────────────")

    # Drop irrelevant columns
    drop_cols = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "popularity"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    le = LabelEncoder()
    df["track_genre"] = le.fit_transform(df["track_genre"])
    df["explicit"]    = df["explicit"].astype(int)

    log(f"  Features after encoding: {df.shape[1]}")

    # Features / target
    X = df.drop(columns=["popular"])
    y = df["popular"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    log(f"  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, X_scaled, y, X.columns.tolist()


# ── Step 3: Clustering ─────────────────────────────────────────────────────────
def run_clustering(X_scaled: pd.DataFrame):
    log("── Clustering (K-Means) ─────────────────────")

    # Elbow method
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Fit final model
    km_final = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels   = km_final.fit_predict(X_scaled)
    log(f"  K-Means with k={N_CLUSTERS} → cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Save elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", color="steelblue")
    plt.title("K-Means Elbow Curve")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_elbow.png", dpi=120)
    plt.close()
    log(f"  Saved → {OUTPUT_DIR}/clustering_elbow.png")

    return labels


# ── Step 4: Classification ─────────────────────────────────────────────────────
def run_classification(X_train, X_test, y_train, y_test, feature_names):
    log("── Classification ───────────────────────────")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost":             xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                                  eval_metric="logloss", verbosity=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model":   model,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_prob),
        }
        log(f"  {name:<25} ROC-AUC={results[name]['roc_auc']:.3f}  F1={results[name]['f1']:.3f}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    log(f"  Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.3f})")

    # Feature importance from Random Forest
    rf_model   = results["Random Forest"]["model"]
    feat_imp   = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values("importance")

    plt.figure(figsize=(8, 6))
    plt.barh(feat_imp["feature"].tail(10), feat_imp["importance"].tail(10), color="steelblue")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=120)
    plt.close()
    log(f"  Saved → {OUTPUT_DIR}/feature_importance.png")

    return results, feat_imp


# ── Step 5: Neural Network ─────────────────────────────────────────────────────
def run_neural_network(X_train, X_test, y_train, y_test):
    log("── Neural Network (TensorFlow) ──────────────")

    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    nn_results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_prob),
    }
    log(f"  Neural Network → ROC-AUC={nn_results['roc_auc']:.3f}  F1={nn_results['f1']:.3f}")

    # Training curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"],     label="Train Loss", color="steelblue")
    plt.plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
    plt.title("Neural Network Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nn_training_curve.png", dpi=120)
    plt.close()
    log(f"  Saved → {OUTPUT_DIR}/nn_training_curve.png")

    return nn_results


# ── Step 6: Final Evaluation ───────────────────────────────────────────────────
def run_evaluation(results: dict, nn_results: dict, feat_imp: pd.DataFrame):
    log("── Final Evaluation ─────────────────────────")

    all_models = list(results.keys()) + ["Neural Network"]
    all_aucs   = [results[m]["roc_auc"] for m in results] + [nn_results["roc_auc"]]

    print("\n  ┌─────────────────────────┬──────────┬────────┬──────────┐")
    print("  │ Model                   │ Accuracy │   F1   │ ROC-AUC  │")
    print("  ├─────────────────────────┼──────────┼────────┼──────────┤")
    for name in results:
        r = results[name]
        print(f"  │ {name:<23}  │  {r['accuracy']:.3f}   │ {r['f1']:.3f}  │  {r['roc_auc']:.3f}   │")
    nn = nn_results
    print(f"  │ {'Neural Network':<23}  │  {nn['accuracy']:.3f}   │ {nn['f1']:.3f}  │  {nn['roc_auc']:.3f}   │")
    print("  └─────────────────────────┴──────────┴────────┴──────────┘")

    best_name = all_models[all_aucs.index(max(all_aucs))]
    log(f"  Best overall: {best_name} (ROC-AUC={max(all_aucs):.3f})")
    log(f"  Top feature : {feat_imp.iloc[-1]['feature']} (importance={feat_imp.iloc[-1]['importance']:.4f})")

    # Final bar chart
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(all_models, all_aucs, color=colors, alpha=0.8)
    for bar, val in zip(bars, all_aucs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9)
    plt.title("ROC-AUC — All Models", fontweight="bold")
    plt.ylabel("ROC-AUC")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_roc_auc_comparison.png", dpi=120)
    plt.close()
    log(f"  Saved → {OUTPUT_DIR}/final_roc_auc_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Spotify ML Pipeline")
    parser.add_argument(
        "--steps", nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Pipeline steps to run (default: all)"
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH,
        help=f"Path to CSV file (default: {DATA_PATH})"
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    steps = args.steps

    ensure_output_dir()
    log(f"Starting pipeline | Steps: {steps}")
    log(f"Data: {args.data}")
    start = time.time()

    # ── Load data (always required) ────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found → {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    log(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", "{:.3f}".format)

    # ── Run steps ──────────────────────────────────────────────────────────────
    X_train = X_test = y_train = y_test = X_scaled = y = feature_names = None
    results = nn_results = feat_imp = cluster_labels = None

    if "eda" in steps:
        df = run_eda(df)

    if "preprocessing" in steps:
        X_train, X_test, y_train, y_test, X_scaled, y, feature_names = run_preprocessing(df)

    if "clustering" in steps:
        if X_scaled is None:
            log("  Skipping clustering — preprocessing must run first")
        else:
            cluster_labels = run_clustering(X_scaled)

    if "classification" in steps:
        if X_train is None:
            log("  Skipping classification — preprocessing must run first")
        else:
            results, feat_imp = run_classification(X_train, X_test, y_train, y_test, feature_names)

    if "neural_network" in steps:
        if X_train is None:
            log("  Skipping neural_network — preprocessing must run first")
        else:
            nn_results = run_neural_network(X_train, X_test, y_train, y_test)

    if "evaluation" in steps:
        if results is None or nn_results is None:
            log("  Skipping evaluation — classification and neural_network must run first")
        else:
            run_evaluation(results, nn_results, feat_imp)

    elapsed = time.time() - start
    log(f"Pipeline complete in {elapsed:.1f}s → outputs saved to /{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()