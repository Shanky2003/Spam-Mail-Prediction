# Spam Mail Prediction using Machine Learning

Classifies a text message as **spam** or **ham** (not spam) using TF-IDF features and four machine learning models.

**Author:** Shashank Deshmukh
**Internship:** AI & ML, Naviotech Solutions Pvt Ltd (Guide: Mr. Md. Ameen)
**University:** Bharati Vidyapeeth (Deemed to be University)

## Results

Dataset: SMS Spam Collection, 5,169 unique messages (4,516 ham, 653 spam), 80/20 stratified split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Naive Bayes (best)** | 98.74% | 98.36% | 91.60% | 94.86% |
| Random Forest | 98.55% | 100.00% | 88.55% | 93.93% |
| Logistic Regression | 98.36% | 95.24% | 91.60% | 93.39% |
| Linear SVM | 98.36% | 96.72% | 90.08% | 93.28% |

5-fold cross-validation accuracy (Naive Bayes): 98.59%.

## Project structure

```
Spam_Mail_Project/
├── spam_detection.py                    # full pipeline as a script
├── Spam_Mail_Prediction.ipynb           # same pipeline, step by step with outputs
├── sms.tsv                              # dataset (downloaded automatically if missing)
├── spam_model.joblib                    # saved trained model
├── requirements.txt
├── Spam_Mail_Prediction_Report.docx/.pdf
├── Spam_Mail_Prediction_Presentation.pptx
├── screenshots/                         # dataset, training, accuracy, prediction
└── figures/                             # charts used in the report
```

## Setup

Requires Python 3.9 or higher.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install the libraries
pip install -r requirements.txt
```

## Run

**Script**

```bash
python spam_detection.py
```

This trains all four models, prints the comparison table, saves the charts (`fig_*.png`), `model_comparison.csv`, `metrics.json` and `spam_model.joblib`, and shows predictions on four sample messages.

**Notebook:** open `Spam_Mail_Prediction.ipynb` in Jupyter, or upload it to [Google Colab](https://colab.research.google.com) (no installation needed) and choose Runtime > Run all.

## Predict a new message

```python
import joblib, re, string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d{5,}", " longnumber ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

model = joblib.load("spam_model.joblib")
msg = "Free entry! Text WIN to 80086 to get a 500 pound prize"
print("SPAM" if model.predict([clean_text(msg)])[0] == 1 else "HAM")
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` (or pandas, seaborn, joblib...) | Run `pip install -r requirements.txt` |
| `pip` not found | Use `python -m pip install -r requirements.txt` |
| `FileNotFoundError: sms.tsv` | Run the script from inside the project folder; it also downloads the file automatically if internet is available |
| Plot windows do not open | Normal: the script saves charts as PNG files instead of showing them |
| Old scikit-learn error | `pip install --upgrade scikit-learn` |

## Dataset

SMS Spam Collection: Almeida, T. A., Gomez Hidalgo, J. M., and Yamakami, A. (2011), available from the UCI Machine Learning Repository.

## License

Released under the MIT License. See the `LICENSE` file.
