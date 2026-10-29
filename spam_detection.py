"""
Spam Mail Prediction using Machine Learning
Author : Shashank Deshmukh
Internship : Naviotech Solutions Pvt Ltd (AI & ML)

Pipeline: load data -> EDA -> clean text -> TF-IDF -> train 4 models
          -> evaluate -> save best model -> predict on new messages
"""

import json
import re
import string
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ------------------------------------------------------------------ 1. Load
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
print("Dataset shape:", df.shape)
print(df.head(), "\n")
print(df["label"].value_counts(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape, "\n")

df["target"] = df["label"].map({"ham": 0, "spam": 1})
df["length"] = df["message"].str.len()
df["words"] = df["message"].str.split().str.len()

# ------------------------------------------------------------------ 2. EDA plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
counts = df["label"].value_counts()
ax[0].bar(counts.index, counts.values, color=["#2e86ab", "#e4572e"])
ax[0].set_title("Class Distribution")
ax[0].set_ylabel("Number of messages")
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 30, str(v), ha="center")
ax[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
          colors=["#2e86ab", "#e4572e"], startangle=90)
ax[1].set_title("Ham vs Spam Share")
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="length", hue="label", bins=50,
             palette=["#2e86ab", "#e4572e"], element="step")
plt.title("Message Length Distribution")
plt.xlabel("Characters per message")
plt.tight_layout()
plt.savefig("fig_length_distribution.png", dpi=150)
plt.close()


# ------------------------------------------------------------------ 3. Clean
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d{5,}", " longnumber ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


df["clean"] = df["message"].apply(clean_text)
print("Cleaning example:")
print(" before:", df["message"][2])
print(" after :", df["clean"][2], "\n")

# top words per class
def top_words(series, n=12):
    return pd.Series(" ".join(series).split()).value_counts().head(n)

tw_ham = top_words(df[df.target == 0]["clean"])
tw_spam = top_words(df[df.target == 1]["clean"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
tw_ham[::-1].plot.barh(ax=ax[0], color="#2e86ab")
ax[0].set_title("Top words in HAM")
tw_spam[::-1].plot.barh(ax=ax[1], color="#e4572e")
ax[1].set_title("Top words in SPAM")
plt.tight_layout()
plt.savefig("fig_top_words.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2,
    stratify=df["target"], random_state=RANDOM_STATE)
print("Train size:", len(X_train), "| Test size:", len(X_test), "\n")

# ------------------------------------------------------------------ 5. Models
def make_pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=10,
                                              class_weight="balanced"),
    "Linear SVM": LinearSVC(C=1, class_weight="balanced",
                            random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=RANDOM_STATE),
}

results, fitted, preds, scores = [], {}, {}, {}
for name, clf in models.items():
    pipe = make_pipe(clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        s = pipe.predict_proba(X_test)[:, 1]
    else:
        s = pipe.decision_function(X_test)
    fitted[name], preds[name], scores[name] = pipe, y_pred, s
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, s),
    }
    results.append(row)
    print(f"{name:20s} acc={row['Accuracy']:.4f} prec={row['Precision']:.4f} "
          f"rec={row['Recall']:.4f} f1={row['F1']:.4f} auc={row['ROC_AUC']:.4f}")

res_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n", res_df.round(4).to_string(index=False), "\n")
res_df.to_csv("model_comparison.csv", index=False)

# ------------------------------------------------------------------ 6. Plots
plt.figure(figsize=(8, 4.5))
x = np.arange(len(res_df))
w = 0.2
for i, m in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    plt.bar(x + i * w, res_df[m], w, label=m)
plt.xticks(x + 1.5 * w, res_df["Model"], rotation=12)
plt.ylim(0.7, 1.02)
plt.title("Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig_model_comparison.png", dpi=150)
plt.close()

best_name = res_df.loc[0, "Model"]
best = fitted[best_name]
print("Best model:", best_name, "\n")

cm = confusion_matrix(y_test, preds[best_name])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("fig_confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(5.5, 4.5))
for name in fitted:
    fpr, tpr, _ = roc_curve(y_test, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, scores[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_roc_curves.png", dpi=150)
plt.close()

report = classification_report(y_test, preds[best_name],
                               target_names=["Ham", "Spam"], digits=4)
print(report)

# ------------------------------------------------------------------ 7. Save
joblib.dump(best, "spam_model.joblib")
tn, fp, fn, tp = cm.ravel()
summary = {
    "total_messages": int(len(df)),
    "ham": int((df.target == 0).sum()),
    "spam": int((df.target == 1).sum()),
    "train": int(len(X_train)),
    "test": int(len(X_test)),
    "best_model": best_name,
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "report": report,
    "results": res_df.round(4).to_dict(orient="records"),
}
with open("metrics.json", "w") as f:
    json.dump(summary, f, indent=2)


# ------------------------------------------------------------------ 8. Predict
def predict_message(msg: str):
    model = joblib.load("spam_model.joblib")
    label = model.predict([clean_text(msg)])[0]
    return "SPAM" if label == 1 else "HAM (not spam)"


samples = [
    "Congratulations! You have won a free iPhone. Click http://win-prize.com now to claim",
    "Hey, are we still meeting for lunch at 1 pm tomorrow?",
    "URGENT! Your account is suspended. Call 09061701461 to claim your 1000 pound cash prize",
    "Please send me the internship report draft by evening",
]
print("FINAL PREDICTIONS")
print("-" * 60)
for s in samples:
    print(f"[{predict_message(s):15s}] {s}")
