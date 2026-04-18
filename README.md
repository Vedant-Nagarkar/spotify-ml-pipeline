# 🎵 Spotify ML Pipeline

End-to-end Machine Learning pipeline to predict whether a Spotify track will be popular based on its audio features.

## 📊 Dataset
- 113,999 Spotify tracks
- 19 features (audio + metadata)
- Source: Kaggle Spotify Tracks Dataset

## 📁 Project Structure
```
spotify-ml-pipeline/
├── src/
│   ├── eda.py              # Exploratory Data Analysis
│   ├── preprocessing.py    # Encoding, scaling, splitting
│   ├── clustering.py       # K-Means clustering
│   ├── classification.py   # ML model training & evaluation
│   ├── neural_network.py   # TensorFlow deep learning
│   ├── evaluation.py       # Final metrics & visualizations
│   └── logger.py           # Logging setup
├── outputs/                # Saved plots and charts
├── models/                 # Saved model files
├── logs/                   # Pipeline logs
├── main.py                 # Pipeline entry point
├── spotify.csv             # Dataset
└── requirements.txt
```

## 🗺️ Pipeline
| Step | File | Description |
|------|------|-------------|
| 1 | `src/eda.py` | Exploratory Data Analysis |
| 2 | `src/preprocessing.py` | Encoding, scaling, splitting |
| 3 | `src/clustering.py` | K-Means to find song clusters |
| 4 | `src/classification.py` | 4 ML models compared |
| 5 | `src/neural_network.py` | TensorFlow deep learning |
| 6 | `src/evaluation.py` | Final metrics and insights |

## 🚀 How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python main.py
```

**Run specific steps only:**
```bash
python main.py --steps eda preprocessing classification
```

**Use a custom data file:**
```bash
python main.py --data path/to/your/file.csv
```

## 🏆 Results
| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| Logistic Regression | 0.615 | 0.422 |
| Gradient Boosting | 0.768 | 0.190 |
| Neural Network | 0.712 | 0.440 |
| XGBoost | 0.855 | 0.638 |
| **Random Forest** | **0.880** | **0.588** |

## 🔑 Key Findings
1. **Genre** is the most important predictor (importance=0.116)
2. **Random Forest** is the best model (ROC-AUC=0.880)
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

## 👤 Author
Vedant Nagarkar — [GitHub](https://github.com/Vedant-Nagarkar)