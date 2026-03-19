# 🎵 Spotify ML Pipeline

End-to-end Machine Learning analysis to predict whether 
a Spotify track will be popular based on its audio features.

## 📊 Dataset
- 113,999 Spotify tracks
- 19 features (audio + metadata)
- Source: Kaggle Spotify Tracks Dataset

## 🗺️ Pipeline
| Step | File | Description |
|------|------|-------------|
| 1 | EDA | Exploratory Data Analysis |
| 2 | Preprocessing | Encoding, scaling, splitting |
| 3 | Clustering | K-Means to find song clusters |
| 4 | Classification | 4 ML models compared |
| 5 | Neural Network | TensorFlow deep learning |
| 6 | Evaluation | Final metrics and insights |

## 🏆 Results
| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| Logistic Regression | 0.615 | 0.422 |
| Gradient Boosting | 0.768 | 0.190 |
| Neural Network | 0.712 | 0.440 |
| XGBoost | 0.855 | 0.638 |
| **Random Forest** | **0.879** | **0.588** |

## 🔑 Key Findings
1. **Genre** is the most important predictor (importance=0.118)
2. **Random Forest** is the best model (ROC-AUC=0.879)
3. **Tree models beat Neural Networks** on tabular data
4. **Popular songs**: high danceability, energy, loudness
5. **Top genres**: pop-film, k-pop, chill
6. **Imbalanced dataset** (74/26) — use F1 & ROC-AUC not accuracy

## 💡 Business Insights
- Target pop-film or k-pop genres for maximum reach
- High energy + danceable + loud = better popularity
- Song duration matters more than musical key
- Avoid purely acoustic tracks for mainstream success

## 🛠️ Tech Stack
- Python 3.11
- Pandas, NumPy, Scikit-learn
- XGBoost, TensorFlow/Keras
- Matplotlib, Seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook spotify_analysis.ipynb
```

## 👤 Author
Vedant Nagarkar — AI Learning Journey Week 0 Capstone