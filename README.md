
# 🎓 Student Performance Predictor

This project predicts student academic performance based on multiple features using two models:
- A **Linear Regression** model with polynomial feature engineering
- A **Neural Network** model built with TensorFlow/Keras

It compares both models using Mean Squared Error (MSE) and visualizes the results.

---

## 📁 Project Structure

```
student-performance-predictor/
├── student_performance.ipynb       ← Main notebook with all code
├── requirements.txt                ← Required Python libraries
├── data/
│   └── Student_Performance.csv     ← Dataset used
├── plots/
│   └── mse_comparison.png          ← Bar chart comparing MSE
├── models/
│   └── student_predictor_model.h5  ← Trained neural network model
└── README.md                       ← Project documentation
```

---

## 🔍 Problem Overview

Educational institutions often want to predict student performance for academic planning, intervention, or personalized learning. This project trains machine learning models to predict a **Performance Index** (numerical score) from student features.

---

## ⚙️ Features & Workflow

- ✅ Loads and preprocesses student data (encoding, scaling)
- ✅ Splits into training, validation (CV), and test sets
- ✅ Applies polynomial feature engineering to enhance linear model
- ✅ Builds and trains a neural network with regularization
- ✅ Compares both models using MSE
- ✅ Saves trained model and visualizations

---

## 📊 Models Used

### 🔹 Linear Regression
- Uses `PolynomialFeatures` (degree chosen by cross-validation)
- Trained with `scikit-learn`

### 🔹 Neural Network
- Built using `Sequential` API in TensorFlow
- Regularized with L2 to reduce overfitting
- Architecture:
  - Input Layer
  - Dense(10) → ReLU
  - Dense(64) → ReLU
  - Dense(32) → ReLU
  - Dense(16) → ReLU
  - Dense(1)  → Linear (for regression)

---

## 📉 Results

Both models were evaluated using **Mean Squared Error (MSE)** on a held-out test set.

Sample Output:

| Model             | MSE (Test Set) |
|------------------|----------------|
| Neural Network    | `X.XX`         |
| Linear Regression | `Y.YY`         |

![MSE Comparison](plots/mse_comparison.png)

---

## 📦 Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

---

## 🚀 How to Run

You can run this project in **Google Colab**:

1. Upload the dataset `Student_Performance.csv`
2. Run all cells in `student_performance.ipynb`
3. Trained model and plots will be saved and downloadable

---

## 💡 What You’ll Learn

- Practical ML model comparison
- Preprocessing with `PolynomialFeatures` and `StandardScaler`
- Regularization in neural networks
- Evaluation using MSE
- TensorFlow model saving and visualization

---

## 📁 License

This project is for educational purposes. You are free to fork, modify, and share it.

---

## ✍️ Author

**Ahmed Frahat**

Connect on [LinkedIn](https://www.linkedin.com) | Explore more on [GitHub](https://github.com)
