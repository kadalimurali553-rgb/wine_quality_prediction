
# VinoSVM — Wine Quality Prediction Website

A polished Flask website for the **Wine Quality Prediction Using Machine Learning** project. The web app trains an RBF-kernel Support Vector Classifier (SVC) on the Wine Quality CSV dataset and exposes a form for predicting the quality score of a new wine sample.

The project presentation describes the use of physicochemical properties, EDA/correlation analysis, train-test splitting, SVM model training, and classification metrics such as accuracy, precision, recall, and F1-score.

## 1. Dataset

Place the Wine Quality dataset at:

`data/winequality-red.csv`

The CSV should use the standard semicolon separator and contain these columns:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality

## 2. Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start:

```bash
python app.py
```

Open the local address shown by Flask, normally `http://127.0.0.1:5000`.

## 3. Model

The app uses:

- StandardScaler for feature standardization
- SVC with an RBF kernel
- C=2.0
- gamma="scale"
- 80/20 train-test split
- random_state=42
- weighted precision/recall/F1

The model predicts the original integer `quality` classes present in the dataset.

## 4. Project basis

The UI and terminology are based on the uploaded project presentation: Wine Quality Prediction Using Machine Learning. The presentation identifies physicochemical properties such as acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates and alcohol; proposes SVM among the ML models; and lists classification metrics including accuracy, precision, recall and F1-score.
