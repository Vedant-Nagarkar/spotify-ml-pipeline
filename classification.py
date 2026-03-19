
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                            GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, f1_score,
                            precision_score, recall_score,
                            roc_auc_score, roc_curve,
                            confusion_matrix,
                            ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from logger import logger
from preprocessing import load_processed_data

# ── Settings ──────────────────────────────────
plt.style.use('seaborn-v0_8')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)



def get_models():
    """
    Define all models with their hyperparameters.

    Why class_weight='balanced'?
    Our dataset is imbalanced (74/26)
    balanced automatically adjusts weights:
    minority class gets higher weight
    so model pays more attention to it

    Why scale_pos_weight in XGBoost?
    XGBoost's way of handling imbalance
    scale_pos_weight = negative/positive = 74/26 = ~3

    Returns:
        models: dict of model name -> model object
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter     = 1000,
            class_weight = 'balanced',
            random_state = 42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators = 100,
            class_weight = 'balanced',
            random_state = 42,
            n_jobs       = -1    # use all CPU cores!
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators = 100,
            random_state = 42
        ),
        'XGBoost': XGBClassifier(
            n_estimators      = 100,
            random_state      = 42,
            scale_pos_weight  = 3,
            eval_metric       = 'logloss',
            use_label_encoder = False
        )
    }
    return models



def train_and_evaluate(models, X_train, X_test,
                        y_train, y_test):
    """
    Train each model and compute all metrics.

    Metrics computed:
    - Accuracy  : overall correct predictions
    - Precision : quality of positive predictions
    - Recall    : coverage of actual positives
    - F1        : balance of precision and recall
    - ROC-AUC   : overall discriminative ability
    - Time      : training time in seconds

    Args:
        models: dict of models
        X_train, X_test: feature arrays
        y_train, y_test: label arrays

    Returns:
        results: dict with metrics for each model
    """
    results = {}

    for name, model in models.items():
        logger.info(f"Training: {name}...")
        start = time.time()

        # Train model
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # Predictions
        # predict()       → hard labels (0 or 1)
        # predict_proba() → probabilities (0.0 to 1.0)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute all metrics
        results[name] = {
            'model'    : model,
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall'   : recall_score(y_test, y_pred),
            'f1'       : f1_score(y_test, y_pred),
            'roc_auc'  : roc_auc_score(y_test, y_pred_prob),
            'time'     : elapsed,
            'y_pred'   : y_pred,
            'y_pred_prob': y_pred_prob
        }

        logger.info(f"Done in {elapsed:.1f}s")
        logger.info(f"  Accuracy : {results[name]['accuracy']:.4f}")
        logger.info(f"  F1 Score : {results[name]['f1']:.4f}")
        logger.info(f"  ROC-AUC  : {results[name]['roc_auc']:.4f}")

    return results



def plot_comparison(results):
    """
    Create visual comparison of all models.
    3 plots: Accuracy, F1 Score, ROC-AUC

    Args:
        results: dict with metrics for each model
    """
    logger.info("Plotting model comparison...")

    metrics      = ['accuracy', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'F1 Score', 'ROC-AUC']
    model_names  = list(results.keys())
    colors       = ['steelblue', 'darkorange', 'green', 'red']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[m][metric] for m in model_names]
        bars   = axes[i].bar(model_names, values,
                            color=colors, alpha=0.8)
        axes[i].set_title(f'{name} Comparison')
        axes[i].set_ylabel(name)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=15)
        axes[i].grid(axis='y', alpha=0.3)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', fontsize=9,
                fontweight='bold'
            )

    plt.suptitle('Model Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/09_model_comparison.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/09_model_comparison.png")



def plot_roc_curves(results, y_test):
    """
    Plot ROC curves for all models on same chart.

    ROC curve plots:
    X axis: False Positive Rate (FPR)
    Y axis: True Positive Rate (TPR = Recall)

    A perfect model hugs the top-left corner
    A random model follows the diagonal line

    Args:
        results: dict with metrics for each model
        y_test: true labels
    """
    logger.info("Plotting ROC curves...")

    colors = ['steelblue', 'darkorange', 'green', 'red']

    plt.figure(figsize=(8, 6))

    for (name, result), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test,
                                result['y_pred_prob'])
        auc = result['roc_auc']
        plt.plot(fpr, tpr, color=color,
                linewidth=2, label=f'{name} (AUC={auc:.3f})')

    # Random classifier baseline
    plt.plot([0,1], [0,1], 'k--',
            linewidth=1, label='Random')
    plt.fill_between([0,1], [0,1],
                    alpha=0.1, color='gray')

    plt.title('ROC Curves - All Models',
            fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/10_roc_curves.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/10_roc_curves.png")



def plot_feature_importance(results, feature_cols):
    """
    Plot feature importance from Random Forest.

    Feature importance tells us:
    Which features are most useful for prediction?

    Random Forest computes this as:
    Mean decrease in impurity across all trees

    Args:
        results: dict with trained models
        feature_cols: list of feature names
    """
    logger.info("Plotting feature importance...")

    rf_model    = results['Random Forest']['model']
    importances = rf_model.feature_importances_

    # Create sorted DataFrame
    feat_df = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)

    logger.info(f"Top 5 features:\n{feat_df.tail(5)}")

    plt.figure(figsize=(10, 7))
    plt.barh(feat_df['feature'],
            feat_df['importance'],
            color='steelblue', alpha=0.8)
    plt.title('Feature Importance - Random Forest',
            fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.grid(axis='x', alpha=0.3)

    for i, (val, name) in enumerate(
        zip(feat_df['importance'], feat_df['feature'])
    ):
        plt.text(val + 0.001, i,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/11_feature_importance.png', dpi=150)
    plt.close()
    logger.info("Saved: outputs/11_feature_importance.png")



def save_best_model(results):
    """
    Find and save the best model by ROC-AUC.

    Why ROC-AUC?
    Our dataset is imbalanced
    ROC-AUC is robust to class imbalance
    Better metric than accuracy here

    Args:
        results: dict with metrics for each model

    Returns:
        best_name: name of best model
    """
    # Find model with highest ROC-AUC
    best_name = max(results,
                    key=lambda x: results[x]['roc_auc'])
    best_model = results[best_name]['model']

    logger.info(f"Best model: {best_name}")
    logger.info(f"Best ROC-AUC: {results[best_name]['roc_auc']:.4f}")

    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(best_name,  'models/best_model_name.pkl')
    logger.info("Saved: models/best_model.pkl")

    return best_name



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING CLASSIFICATION PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test, \
        scaler, feature_cols = load_processed_data()

    # Step 2: Define models
    models = get_models()

    # Step 3: Train and evaluate all models
    results = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # Step 4: Print summary table
    logger.info("=" * 55)
    logger.info("FINAL RESULTS")
    logger.info("=" * 55)
    logger.info(f"{'Model':<25} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Time':>6}")
    logger.info("-" * 55)
    for name, r in results.items():
        logger.info(
            f"{name:<25} {r['accuracy']:>6.3f} "
            f"{r['f1']:>6.3f} {r['roc_auc']:>6.3f} "
            f"{r['time']:>5.1f}s"
        )

    # Step 5: Plot comparison
    plot_comparison(results)

    # Step 6: Plot ROC curves
    plot_roc_curves(results, y_test)

    # Step 7: Feature importance
    plot_feature_importance(results, feature_cols)

    # Step 8: Save best model
    best_name = save_best_model(results)

    # Step 9: Save all results
    joblib.dump(results, 'models/all_results.pkl')
    logger.info("Saved: models/all_results.pkl")

    logger.info("=" * 50)
    logger.info(f"CLASSIFICATION COMPLETE!")
    logger.info(f"Best Model: {best_name}")
    logger.info("=" * 50)