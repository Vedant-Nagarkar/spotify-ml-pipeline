
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.logger import logger

# import load_processed_data from our preprocessing file
# This is why we made it a function — easy to reuse!
from src.preprocessing import load_processed_data

# ── Settings ──────────────────────────────────
plt.style.use('seaborn-v0_8')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)   # folder to save models

BEST_K = 5   # we determined this from elbow method



def elbow_method(X_train, k_range=range(2, 11)):
    """
    Find the best K using the elbow method.
    Trains KMeans for each K and plots inertia.

    Args:
        X_train: scaled training features
        k_range: range of K values to try

    Returns:
        inertias: list of inertia values
    """
    logger.info("Running Elbow Method...")
    inertias = []

    for k in k_range:
        # n_init=10 means run KMeans 10 times
        # with different random initializations
        # and keep the best result
        # This avoids bad random initialization
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        inertias.append(km.inertia_)
        logger.info(f"  K={k}: inertia={km.inertia_:.0f}")

    # ── Plot elbow curve ──────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias,
            'bo-', linewidth=2, markersize=8)
    plt.axvline(x=BEST_K, color='red',
                linestyle='--', label=f'K={BEST_K}')
    plt.title('Elbow Method - Finding Best K')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Inertia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/06_elbow_curve.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/06_elbow_curve.png")

    return inertias



def train_kmeans(X_train, k=BEST_K):
    """
    Train KMeans with the best K.
    Also computes silhouette score to measure
    cluster quality.

    Silhouette score:
    +1 = perfectly separated clusters
    0 = overlapping clusters
    -1 = wrong cluster assignments

    Args:
        X_train: scaled training features
        k: number of clusters

    Returns:
        km: fitted KMeans model
    """
    logger.info(f"Training KMeans with K={k}...")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train)

    logger.info(f"Inertia: {km.inertia_:.2f}")

    # Silhouette score on subset (full dataset too slow)
    # sample_size=10000 uses random 10k samples for speed
    sil = silhouette_score(
        X_train, km.labels_,
        sample_size  = 10000,
        random_state = 42
    )
    logger.info(f"Silhouette Score: {sil:.4f}")
    logger.info("(closer to 1.0 = better clusters)")

    return km



def visualize_clusters(X_train, labels):
    """
    Visualize clusters in 2D using PCA.

    Why PCA?
    Our data has 15 features (15 dimensions)
    We can only plot in 2D or 3D
    PCA compresses 15D → 2D keeping most variance

    Args:
        X_train: scaled training features
        labels: cluster labels from KMeans
    """
    logger.info("Visualizing clusters with PCA...")

    # Reduce 15 dimensions → 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    logger.info(f"PCA variance: PC1={var1:.1f}%, PC2={var2:.1f}%")
    logger.info(f"Total variance captured: {var1+var2:.1f}%")

    # ── Plot clusters ─────────────────────────
    colors = ['steelblue', 'darkorange',
            'green', 'red', 'purple']

    plt.figure(figsize=(10, 7))
    for i in range(BEST_K):
        # mask = boolean array, True where label == i
        mask = labels == i
        plt.scatter(
            X_pca[mask, 0],   # PC1 values for cluster i
            X_pca[mask, 1],   # PC2 values for cluster i
            c      = colors[i],
            label  = f'Cluster {i}',
            alpha  = 0.3,     # transparency
            s      = 5        # dot size
        )

    plt.title('K-Means Clusters (PCA Visualization)')
    plt.xlabel(f'PC1 ({var1:.1f}% variance)')
    plt.ylabel(f'PC2 ({var2:.1f}% variance)')
    plt.legend(markerscale=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/07_kmeans_clusters.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/07_kmeans_clusters.png")



def cluster_profiles(X_train, labels, feature_cols):
    """
    Analyze what each cluster represents.
    Computes mean of each feature per cluster.

    This helps us understand:
    - Cluster 0 = high danceability, low acousticness
                → "Dance Pop"
    - Cluster 1 = low energy, high acousticness
                → "Acoustic/Chill"
    etc.

    Args:
        X_train: scaled training features
        labels: cluster labels
        feature_cols: list of feature names

    Returns:
        profiles_df: DataFrame with cluster profiles
    """
    logger.info("Computing cluster profiles...")

    # Create DataFrame with features + cluster labels
    df_cluster = pd.DataFrame(X_train, columns=feature_cols)
    df_cluster['cluster'] = labels

    # Compute mean of each feature per cluster
    profiles = df_cluster.groupby('cluster').mean().round(3)

    logger.info(f"Cluster Profiles:\n{profiles}")

    # ── Heatmap ───────────────────────────────
    # Normalize profiles to 0-1 for better visualization
    profiles_norm = (profiles - profiles.min()) / \
                    (profiles.max() - profiles.min())

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        profiles_norm,
        annot      = True,
        fmt        = '.2f',
        cmap       = 'YlOrRd',  # yellow to red
        linewidths = 0.5
    )
    plt.title('Cluster Profiles (Normalized)',
            fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/08_cluster_profiles.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/08_cluster_profiles.png")

    return profiles



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING CLUSTERING PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load preprocessed data
    # Notice: we import and call load_processed_data()
    # from 02_preprocessing.py — code reuse!
    X_train, X_test, y_train, y_test, \
        scaler, feature_cols = load_processed_data()

    # Step 2: Elbow method
    elbow_method(X_train)

    # Step 3: Train KMeans
    km = train_kmeans(X_train, k=BEST_K)

    # Step 4: Visualize clusters
    visualize_clusters(X_train, km.labels_)

    # Step 5: Cluster profiles
    profiles = cluster_profiles(
        X_train, km.labels_, feature_cols
    )

    # Step 6: Save model
    joblib.dump(km, 'models/kmeans_model.pkl')
    logger.info("Saved: models/kmeans_model.pkl")

    logger.info("=" * 50)
    logger.info("CLUSTERING COMPLETE!")
    logger.info("=" * 50)