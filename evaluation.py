
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.metrics import (confusion_matrix,
                            ConfusionMatrixDisplay,
                            precision_recall_curve,
                            roc_curve, roc_auc_score,
                            classification_report)

from logger import logger
from preprocessing import load_processed_data

# ── Settings ──────────────────────────────────
plt.style.use('seaborn-v0_8')
os.makedirs('outputs', exist_ok=True)



def load_all_results():
    """
    Load all saved models and results from disk.

    This is why we saved everything in previous files!
    No need to retrain — just load and evaluate.

    Returns:
        all_results : dict with classical model results
        nn_results  : dict with neural network results
        best_model  : best classical model object
        best_name   : name of best classical model
    """
    logger.info("Loading all saved models and results...")

    # Load classical model results
    all_results = joblib.load('models/all_results.pkl')
    best_model  = joblib.load('models/best_model.pkl')
    best_name   = joblib.load('models/best_model_name.pkl')
    nn_results  = joblib.load('models/nn_results.pkl')

    logger.info(f"Loaded {len(all_results)} classical models")
    logger.info(f"Best classical model: {best_name}")

    # Load neural network
    nn_model = tf.keras.models.load_model(
        'models/neural_network_model'
    )
    logger.info("Loaded neural network model")

    return all_results, nn_results, best_model, \
        best_name, nn_model



def plot_final_dashboard(all_results, nn_results,
                        best_model, best_name,
                        X_test, y_test):
    """
    Create a comprehensive final dashboard with:
    1. ROC-AUC comparison bar chart
    2. Confusion matrix for best model
    3. ROC curves for all models
    4. Precision-Recall curve for best model

    Args:
        all_results : classical model results
        nn_results  : neural network results
        best_model  : best classical model
        best_name   : name of best model
        X_test      : test features
        y_test      : test labels
    """
    logger.info("Creating final dashboard...")

    # Create 2x2 grid of plots
    fig = plt.figure(figsize=(18, 14))

    # gridspec gives more control than subplots
    gs = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # top left
    ax2 = fig.add_subplot(gs[0, 1])  # top right
    ax3 = fig.add_subplot(gs[1, 0])  # bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # bottom right

    # ── Plot 1: ROC-AUC Comparison ────────────
    all_names = list(all_results.keys()) + ['Neural Network']
    all_aucs  = [all_results[m]['roc_auc']
                for m in all_results.keys()] + \
                [nn_results['roc_auc']]
    colors    = ['steelblue', 'darkorange',
                'green', 'red', 'purple']

    bars = ax1.bar(all_names, all_aucs,
                color=colors, alpha=0.8)
    ax1.set_title('ROC-AUC — All Models',
                fontweight='bold')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_ylim(0.5, 1.0)
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='gray',
                linestyle='--', alpha=0.5,
                label='Random baseline')

    for bar, val in zip(bars, all_aucs):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center',
            fontsize=9, fontweight='bold'
        )

    # ── Plot 2: Confusion Matrix ──────────────
    # Confusion matrix shows:
    # TP = predicted popular, actually popular
    # TN = predicted not popular, actually not popular
    # FP = predicted popular, actually not popular (false alarm)
    # FN = predicted not popular, actually popular (missed!)
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = ['Not Popular', 'Popular']
    )
    disp.plot(ax=ax2, colorbar=False, cmap='Blues')
    ax2.set_title(f'Confusion Matrix — {best_name}',
                fontweight='bold')

    # Add metrics as text below
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred_best)
    f1  = f1_score(y_test, y_pred_best)
    ax2.text(0.02, -0.15,
            f'Accuracy={acc:.3f}  F1={f1:.3f}',
            transform=ax2.transAxes,
            fontsize=10, color='navy')

    # ── Plot 3: ROC Curves ────────────────────
    roc_colors = ['steelblue', 'darkorange',
                'green', 'red']

    for (name, result), color in zip(
        all_results.items(), roc_colors
    ):
        fpr, tpr, _ = roc_curve(
            y_test, result['y_pred_prob']
        )
        ax3.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} ({result['roc_auc']:.3f})")

    ax3.plot([0,1], [0,1], 'k--',
            linewidth=1, label='Random')
    ax3.set_title('ROC Curves — All Models',
                fontweight='bold')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Precision-Recall Curve ────────
    # Precision-Recall is better than ROC
    # for imbalanced datasets!
    # Shows tradeoff between precision and recall
    # at different thresholds
    best_prob = all_results[best_name]['y_pred_prob']
    prec, rec, _ = precision_recall_curve(
        y_test, best_prob
    )

    ax4.plot(rec, prec, color='darkorange',
            linewidth=2, label=f'{best_name}')
    ax4.fill_between(rec, prec,
                    alpha=0.1, color='darkorange')

    # Baseline = random classifier for imbalanced data
    baseline = y_test.mean()
    ax4.axhline(y=baseline, color='gray',
                linestyle='--', alpha=0.7,
                label=f'Baseline ({baseline:.2f})')

    ax4.set_title(f'Precision-Recall — {best_name}',
                fontweight='bold')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Spotify ML Pipeline — Final Dashboard',
                fontsize=18, fontweight='bold', y=1.01)

    plt.savefig('outputs/13_final_dashboard.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/13_final_dashboard.png")



def print_classification_report(best_model,
                                best_name,
                                X_test, y_test):
    """
    Print detailed classification report.

    Classification report shows per-class metrics:
    - Precision for each class
    - Recall for each class
    - F1 for each class
    - Support (number of samples per class)

    Args:
        best_model: best classical model
        best_name: name of best model
        X_test: test features
        y_test: test labels
    """
    logger.info("=" * 55)
    logger.info(f"CLASSIFICATION REPORT — {best_name}")
    logger.info("=" * 55)

    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names = ['Not Popular', 'Popular']
    )
    logger.info(f"\n{report}")



def print_business_insights(all_results, nn_results):
    """
    Print final business insights and recommendations.

    This is the most important part!
    Translates ML results into actionable insights.

    Args:
        all_results: classical model results
        nn_results: neural network results
    """
    logger.info("=" * 55)
    logger.info("FINAL BUSINESS INSIGHTS")
    logger.info("=" * 55)

    # Find best model
    best_name = max(all_results,
                    key=lambda x: all_results[x]['roc_auc'])
    best_auc  = all_results[best_name]['roc_auc']

    insights = [
        "",
        "WHAT MAKES A SONG POPULAR ON SPOTIFY?",
        "-" * 40,
        "1. GENRE matters most (importance=0.118)",
        "   Best genres: pop-film, k-pop, chill",
        "   Avoid: purely niche/underground genres",
        "",
        "2. SONG LENGTH matters (importance=0.091)",
        "   Sweet spot: 3-4 minutes (180,000-240,000ms)",
        "   Too long or too short = less popular",
        "",
        "3. MOOD matters (valence importance=0.087)",
        "   Happy songs slightly more popular",
        "   But sad songs also do well (chill, sad genres)",
        "",
        "4. PRODUCTION STYLE matters",
        "   Electronic > Acoustic for popularity",
        "   High energy + loud = more streams",
        "   Songs WITH vocals > instrumental",
        "",
        "5. DANCEABILITY matters (importance=0.083)",
        "   More danceable = more popular",
        "   Sweet spot: 0.6-0.8 danceability",
        "",
        "ML FINDINGS:",
        "-" * 40,
        f"Best Model    : {best_name}",
        f"Best ROC-AUC  : {best_auc:.4f}",
        f"Neural Network: {nn_results['roc_auc']:.4f}",
        "Tree models beat Neural Networks on tabular data!",
        "This is consistent with recent research (2023-2024)",
        "",
        "LIMITATIONS:",
        "-" * 40,
        "- Popularity changes over time (recency bias)",
        "- Artist fame not captured in audio features",
        "- Marketing spend not included",
        "- Social media virality not modeled",
    ]

    for insight in insights:
        logger.info(insight)



if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("STARTING FINAL EVALUATION")
    logger.info("=" * 55)

    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test, \
        scaler, feature_cols = load_processed_data()

    # Step 2: Load all saved models
    all_results, nn_results, best_model, \
        best_name, nn_model = load_all_results()

    # Step 3: Final dashboard
    plot_final_dashboard(
        all_results, nn_results,
        best_model, best_name,
        X_test, y_test
    )

    # Step 4: Classification report
    print_classification_report(
        best_model, best_name, X_test, y_test
    )

    # Step 5: Business insights
    print_business_insights(all_results, nn_results)

    logger.info("=" * 55)
    logger.info("EVALUATION COMPLETE!")
    logger.info("All outputs saved to outputs/ folder")
    logger.info("=" * 55)