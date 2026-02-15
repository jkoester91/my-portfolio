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
