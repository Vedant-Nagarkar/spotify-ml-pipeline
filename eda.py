# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# os for file/folder operations
import os

# warnings to suppress unnecessary messages
import warnings
warnings.filterwarnings('ignore')

# our custom logger (reused from logger.py)
from logger import logger


# ── Plot Settings ─────────────────────────────
# These settings apply to ALL plots in this file
plt.style.use('seaborn-v0_8')   # clean background style
sns.set_palette('husl')          # color palette
os.makedirs('outputs', exist_ok=True)  # create outputs/ if not exists


def load_data(filepath='spotify.csv'):
    """
    Load spotify dataset from CSV file.
    
    Args:
        filepath: path to the CSV file
    
    Returns:
        df: cleaned pandas DataFrame
    """
    logger.info(f"Loading dataset from: {filepath}")
    
    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Drop unnecessary columns if they exist
    # 'Unnamed: 0' is just an index column — not useful
    # 'track_id' is a string ID — not useful for ML
    cols_to_drop = ['Unnamed: 0', 'track_id']
    cols_exist   = [c for c in cols_to_drop if c in df.columns]
    if cols_exist:
        df = df.drop(columns=cols_exist)
        logger.info(f"Dropped columns: {cols_exist}")
    
    # Drop rows with missing values
    # .dropna() removes any row that has at least one NaN
    before = len(df)
    df     = df.dropna()
    after  = len(df)
    logger.info(f"Dropped {before - after} rows with missing values")
    
    return df


def basic_info(df):
    """
    Print basic dataset information.
    
    Args:
        df: pandas DataFrame
    """
    logger.info("=" * 50)
    logger.info("DATASET OVERVIEW")
    logger.info("=" * 50)
    
    # Shape = (rows, columns)
    logger.info(f"Shape    : {df.shape}")
    
    # dtypes tells us what type each column is
    # object = string, int64 = integer, float64 = decimal
    logger.info(f"Dtypes   :\n{df.dtypes}")
    
    # isnull().sum() counts missing values per column
    logger.info(f"Missing  :\n{df.isnull().sum()}")
    
    # describe() gives count, mean, std, min, max etc
    logger.info(f"Statistics:\n{df.describe().round(3)}")



def analyze_popularity(df):
    """
    Analyze and visualize the popularity distribution.
    Creates binary 'popular' column (1 if popularity >= 50)
    
    Args:
        df: pandas DataFrame
    
    Returns:
        df: DataFrame with new 'popular' column added
    """
    logger.info("Analyzing popularity distribution...")
    
    # Basic stats about popularity
    logger.info(f"Mean popularity  : {df['popularity'].mean():.2f}")
    logger.info(f"Median popularity: {df['popularity'].median():.2f}")
    logger.info(f"Std popularity   : {df['popularity'].std():.2f}")
    
    # Create binary target variable
    # Songs with popularity >= 50 are "popular" (1)
    # Songs with popularity < 50 are "not popular" (0)
    # This converts a regression problem → classification problem
    df['popular'] = (df['popularity'] >= 50).astype(int)
    
    popular_pct     = df['popular'].mean() * 100
    not_popular_pct = 100 - popular_pct
    logger.info(f"Popular songs    : {popular_pct:.1f}%")
    logger.info(f"Not popular songs: {not_popular_pct:.1f}%")
    
    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of raw popularity scores
    axes[0].hist(df['popularity'], bins=50,
                color='steelblue', edgecolor='white')
    axes[0].set_title('Popularity Score Distribution')
    axes[0].set_xlabel('Popularity (0-100)')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Pie chart of popular vs not popular
    # value_counts() counts how many 0s and 1s
    df['popular'].value_counts().plot.pie(
        labels   = ['Not Popular', 'Popular'],
        autopct  = '%1.1f%%',        # show percentage
        colors   = ['steelblue', 'darkorange'],
        startangle = 90,
        ax       = axes[1]
    )
    axes[1].set_title('Popular vs Not Popular')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    # Save to outputs/ — dpi=150 gives good quality
    plt.savefig('outputs/01_popularity.png', dpi=150)
    plt.close()   # close instead of show (no popup in script mode!)
    logger.info("Saved: outputs/01_popularity.png")
    
    return df



def analyze_audio_features(df):
    """
    Visualize distribution of all audio features.
    Also compares popular vs not popular songs.
    
    Args:
        df: pandas DataFrame with 'popular' column
    """
    logger.info("Analyzing audio features...")
    
    # List of numerical audio features to analyze
    audio_features = [
        'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness',
        'instrumentalness', 'liveness',
        'valence', 'tempo'
    ]
    
    # ── Plot 1: Feature Distributions ─────────
    # 3x3 grid of histograms, one per feature
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flat   # flatten 2D array → 1D for easy looping
    
    for i, feature in enumerate(audio_features):
        axes[i].hist(df[feature], bins=50,
                    color='steelblue',
                    edgecolor='white', alpha=0.7)
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Audio Features Distribution',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/02_audio_features.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/02_audio_features.png")
    
    # ── Plot 2: Popular vs Not Popular ────────
    # Overlapping histograms to compare distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flat
    
    for i, feature in enumerate(audio_features):
        # Split by popular/not popular
        popular     = df[df['popular'] == 1][feature]
        not_popular = df[df['popular'] == 0][feature]
        
        # density=True normalizes so we can compare
        # even though there are more not-popular songs
        axes[i].hist(not_popular, bins=30, alpha=0.6,
                    color='steelblue',
                    label='Not Popular', density=True)
        axes[i].hist(popular, bins=30, alpha=0.6,
                    color='darkorange',
                    label='Popular', density=True)
        axes[i].set_title(f'{feature}')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Popular vs Not Popular — Feature Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/03_popular_vs_notpopular.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/03_popular_vs_notpopular.png")



def analyze_correlations(df):
    """
    Create correlation heatmap for numerical features.
    
    Args:
        df: pandas DataFrame
    """
    logger.info("Analyzing feature correlations...")
    
    # Select only numerical columns for correlation
    numeric_cols = [
        'popularity', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms'
    ]
    
    # .corr() computes pairwise correlation between columns
    # Values range from -1 (negative) to +1 (positive)
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot    = True,        # show numbers in cells
        fmt      = '.2f',       # 2 decimal places
        cmap     = 'coolwarm',  # red=positive, blue=negative
        center   = 0,           # white = 0 correlation
        square   = True,        # square cells
        linewidths = 0.5
    )
    plt.title('Feature Correlation Heatmap',
            fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/04_correlation_heatmap.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/04_correlation_heatmap.png")



def analyze_genres(df):
    """
    Analyze and visualize top genres by popularity.
    
    Args:
        df: pandas DataFrame
    """
    logger.info("Analyzing genres...")
    
    # Group by genre, compute mean popularity, sort descending
    # .head(20) takes only top 20 genres
    genre_popularity = df.groupby('track_genre')['popularity'] \
                        .mean() \
                        .sort_values(ascending=False) \
                        .head(20)
    
    logger.info(f"Top 5 genres:\n{genre_popularity.head()}")
    
    plt.figure(figsize=(12, 8))
    genre_popularity.plot(kind='barh',   # horizontal bar chart
                        color='steelblue',
                        alpha=0.8)
    plt.title('Top 20 Genres by Average Popularity',
            fontsize=14, fontweight='bold')
    plt.xlabel('Average Popularity Score')
    plt.ylabel('Genre')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/05_top_genres.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/05_top_genres.png")



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING EDA PIPELINE")
    logger.info("=" * 50)
    
    # Step 1: Load data
    df = load_data('spotify.csv')
    
    # Step 2: Print basic info
    basic_info(df)
    
    # Step 3: Analyze popularity + create binary target
    # df is returned with new 'popular' column added
    df = analyze_popularity(df)
    
    # Step 4: Analyze audio features
    analyze_audio_features(df)
    
    # Step 5: Correlation heatmap
    analyze_correlations(df)
    
    # Step 6: Genre analysis
    analyze_genres(df)
    
    logger.info("=" * 50)
    logger.info("EDA COMPLETE! Check outputs/ folder for plots")
    logger.info("=" * 50)




