# 💬 Duplicate Question Detection

> **An NLP project that determines whether two questions are semantically similar using multiple Natural Language Processing techniques and machine learning models. The repository compares traditional feature engineering with deep learning approaches for duplicate question identification.**

---

## 🎯 Problem Statement

Online Q&A platforms often receive multiple users asking the same question in different ways. Detecting duplicate questions helps improve search quality, reduce redundant content, and enhance user experience.

This project explores several NLP techniques to determine whether two questions express the same intent.

---

## 🔍 What's Inside?

### 📊 Dataset

The project uses the **Quora Question Pairs** dataset, containing thousands of question pairs labeled as duplicate or non-duplicate.

---

### 🧠 Feature Engineering

Instead of relying on a single approach, this project experiments with multiple text representations.

◆ Bag of Words (BoW)

◆ TF-IDF

◆ Word2Vec

◆ Advanced handcrafted NLP features

---

### 🤖 Models

| Model | Purpose |
|--------|----------|
| Bag of Words | Traditional text representation |
| TF-IDF | Statistical text representation |
| Word2Vec | Semantic word embeddings |
| LSTM | Deep Learning sequence model |

---

## 📁 Repository Overview

```
📂 Data
    └── Quora Question Pairs Dataset

📂 Files
    ├── Data Exploration
    ├── Feature Engineering
    └── Model Development

📂 Models
    ├── Bag of Words
    ├── TF-IDF
    ├── Word2Vec
    └── LSTM

📂 UI
    └── Prediction Interface
```

---

## 🚀 NLP Pipeline

```text
Question Pair
      │
      ▼
Text Cleaning
      │
      ▼
Feature Extraction
      │
      │
      │              
      ▼               
Traditional NLP   
      │            
      ▼            
Prediction Models
      │
      ▼
Duplicate / Non-Duplicate
```

---

## ⚡ Highlights

◆ Multiple feature extraction techniques

◆ Comparison of classical and deep learning methods

◆ Word embedding-based semantic understanding

◆ LSTM implementation for sequence learning

◆ Ready-to-use trained models

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| Programming | Python |
| NLP | NLTK, Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 📌 Repository Includes

✔ Data preprocessing notebooks

✔ Advanced feature engineering

✔ Multiple trained models

✔ Saved vectorizers and tokenizers

✔ User interface for predictions

---

## 👨‍💻 Author

**Harshil Kothiya**

AI/ML Engineer

- GitHub: https://github.com/Harshilkothiya
- LinkedIn: https://www.linkedin.com/in/harshil-kothiya/
