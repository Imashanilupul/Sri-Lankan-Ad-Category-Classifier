# Sri-Lankan Ad Category Classifier

A small project for classifying Sri Lankan classified ads into categories using machine learning (CatBoost was used in experiments). The repository contains the cleaned datasets and a notebook showing data exploration, training, and inference.

## Contents

- `Notebook.ipynb` — interactive notebook with data loading, preprocessing, training, evaluation and inference examples.
- `datasets/cleaned_sri_lankan_classified_ads_matching_dataset.csv` — cleaned dataset used for experiments.
- `datasets/sri_lankan_classified_ads_matching_dataset_v1.csv` — original/raw dataset (v1).

## Quick overview

This project aims to classify short classified ad text from Sri Lanka into predefined categories. The focus is on a compact, reproducible workflow: data loading, basic cleaning, feature preparation, model training (CatBoost in original experiments), evaluation and example inference.

## Repository structure

```
.
├─ Notebook.ipynb
├─ README.md
└─ datasets/
	├─ cleaned_sri_lankan_classified_ads_matching_dataset.csv
	└─ sri_lankan_classified_ads_matching_dataset_v1.csv
```

## Getting started

Requirements

- Python 3.8+ recommended
- Typical packages: pandas, scikit-learn, catboost (or transformers/torch if you try deep learning), matplotlib/seaborn for plots

Optional: create a virtual environment and install packages from `requirements.txt` (this repo does not include one yet — see TODOs).

Example (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install core packages manually:

```cmd
pip install pandas scikit-learn catboost matplotlib seaborn
```

## Usage

Open `Notebook.ipynb` in Jupyter or VS Code. The notebook contains these steps:

1. Load dataset from `datasets/cleaned_sri_lankan_classified_ads_matching_dataset.csv` using pandas.
2. Inspect and preprocess text (lowercase, remove punctuation, optional tokenization).
3. Split into train/validation/test sets.
4. Extract features (TF-IDF, simple text stats, or embeddings).
5. Train a classifier (CatBoost was used in the experiments).
6. Evaluate with accuracy, precision/recall/F1 and a confusion matrix.
7. Save model and run inference on new text examples.

Minimal code snippets you will find in the notebook:

```python
import pandas as pd
df = pd.read_csv('datasets/cleaned_sri_lankan_classified_ads_matching_dataset.csv')
df.head()
```

And a minimal training sketch (conceptual):

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier

X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
tf = TfidfVectorizer(max_features=20000)
X_train_tf = tf.fit_transform(X_train)
X_val_tf = tf.transform(X_val)

model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
model.fit(X_train_tf, y_train, eval_set=(X_val_tf, y_val))
```

## Evaluation and expected results

Look in the notebook for the experiment logs. Typical evaluation steps include classification report (precision/recall/F1) and a confusion matrix. Results will vary depending on preprocessing, features and hyperparameters.

## Inference example

```python
text = "Selling a second-hand motorbike in Colombo, good condition"
vec = tf.transform([text])
pred = model.predict(vec)
print('predicted category:', pred[0])
```

## Next steps / TODO

- Add `requirements.txt` listing the project's dependencies.
- Add a small script `train.py` or a compact notebook with a step-by-step runnable training pipeline.
- Add tests or a minimal CI workflow to ensure notebooks/scripts run.

If you'd like, I can add a `requirements.txt` and a short example script/notebook — tell me which you'd prefer.



## License

Add a license of your choice (e.g., MIT). If you want, I can add an `LICENSE` file.


