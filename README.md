# AWS Cloud Machine Learning: Diabetes Prediction Pipeline

## 🚀 Project Overview
This project demonstrates an end-to-end Machine Learning workflow on AWS. I developed a Random Forest classifier to predict diabetes based on the Pima Indians dataset, achieving a **84.42% test accuracy**.

## 🏗️ Architecture
The project leverages a "Storage-First" cloud architecture:
* **Storage:** Data and trained models are versioned in **Amazon S3**.
* **Compute:** Training was performed on an **Amazon EC2** (Amazon Linux 2023) instance.
* **Security:** **IAM Roles** allowed the EC2 instance to communicate with S3 securely without hardcoded credentials.

## 📂 Project Structure
- `train.py`: Main Python script for data preprocessing and model training.
- `requirements.txt`: List of dependencies needed to run the project.
- `data/`: (Stored on S3) Contains the Pima Indians Diabetes dataset.
- `models/`: (Stored on S3) Contains the serialized `.pkl` model file.

## 📊 Model Performance
* **Test Accuracy:** 84.42%
* **Key Predictors:** 1. Plasma glucose concentration (26.09%)
  2. Age (15.72%)
  3. Body Mass Index (14.13%)

### Classification Report
| Metric | No Diabetes (0) | Diabetes (1) |
| :--- | :--- | :--- |
| Precision | 0.84 | 0.85 |
| Recall | 0.94 | 0.65 |
| F1-Score | 0.89 | 0.74 |

## 🛠️ Tech Stack
* **Cloud:** AWS (EC2, S3, IAM, VPC)
* **Language:** Python 3.9
* **Libraries:** Scikit-learn, Pandas, Pickle
* **CLI:** AWS CLI for data synchronization

## 🏁 Key Takeaways
* Successfully implemented the **Cloud-Native Lifecycle**: Stage -> Compute -> Store -> Terminate.
* Managed AWS networking via custom **VPC** and security groups.
* Optimized model deployment by exporting as a `.pkl` file for future inference.

## 🎬 Sprin5 15 - NLP Project: Sentiment Analysis of IMDB Movie Reviews

### **Objective**
To develop a high-performance NLP model capable of classifying movie reviews as positive or negative, progressing from basic statistical methods to state-of-the-art Deep Learning (BERT).

---

### **Model Comparison & Performance**

I implemented and evaluated multiple architectures to find the best balance between speed and contextual understanding.

| Model | Transformation | Classifier | Test Accuracy | Test F1 | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 0** | Baseline | Dummy (Most Frequent) | 0.50 | 0.00 | 0.50 |
| **Model 1** | NLTK (TF-IDF) | Logistic Regression | 0.88 | 0.88 | 0.95 |
| **Model 3** | spaCy Lemmatization | Logistic Regression | 0.88 | 0.88 | 0.95 |
| **Model 4** | spaCy Lemmatization | LGBM (Gradient Boosting) | 0.86 | 0.86 | 0.94 |
| **Model 9** | **BERT Embeddings** | **Logistic Regression** | **0.80*** | **0.79*** | **0.89*** |

*\*Note: BERT was trained on a subset of 400 reviews to optimize for compute time on Google Colab T4 GPU.*

---

### **Key Insights: Why BERT Wins**

While traditional TF-IDF models performed well on simple vocabulary, **Model 9 (BERT)** demonstrated a superior grasp of nuanced language. In custom testing, BERT successfully identified positive sentiment in reviews containing mixed feedback (e.g., "upsides and downsides") where traditional models failed.



#### **Technical Highlights:**
* **Cloud Hardware:** Utilized **Google Colab T4 GPU** to accelerate transformer embedding extraction.
* **Advanced Preprocessing:** Compared **NLTK** vs. **spaCy** lemmatization to reduce feature sparsity.
* **Contextual Embeddings:** Leveraged `bert-base-uncased` to capture bidirectional context in text data.

---

### **Live Project Links**
* **[Open Notebook in Google Colab](https://colab.research.google.com/drive/1BF4kgm-ZoCn6Coh49QjkZuvARSGBtoO5)**
* **[View Source Code on GitHub](./Sprint_15_NLP_Revised.ipynb)**
