# ❤️ Heart Disease Prediction using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete machine learning project that predicts whether a person has **heart disease** using **Logistic Regression**, with full EDA, visualizations, metrics, and a predictive system.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Algorithm** | Logistic Regression |
| **Dataset** | Heart Disease UCI (303 samples, 13 features) |
| **Target** | `target` – 1 = Disease, 0 = Healthy |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Typical Accuracy** | Train ~85%, Test ~82% |

### Features
| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia (0–3) |

**Target:** `1` → Defective Heart | `0` → Healthy Heart

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

jupyter notebook Project_10_Heart_Disease_Prediction.ipynb

# Full script (all figures + metrics)
python heart_disease_prediction.py
```

---

## 📈 What This Project Includes (A–Z)

### Data & Preprocessing
- Load `heart_disease_data.csv`
- Head / tail / shape / info / missing values
- Statistical summary
- Target class distribution

### Visualizations
- Target distribution (Healthy vs Disease)
- Correlation heatmap
- Age distribution by target
- Chest pain type vs target
- Confusion matrix heatmap

### Model & Metrics
- Logistic Regression (`max_iter=1000` for convergence)
- Accuracy (train & test)
- Classification report (precision, recall, F1)
- Confusion matrix

### Predictive System
- Sample input → Disease / No Disease prediction

---

## 📁 Project Structure

```
heart-disease-prediction/
├── Project_10_Heart_Disease_Prediction.ipynb
├── heart_disease_prediction.py
├── heart_disease_data.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Technologies

- Python 3.8+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, metrics)

---

## 📝 License

MIT License – free to use and modify.

---

**Made with ❤️ for learning Machine Learning**
