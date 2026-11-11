# 🏥 InsureAI – Medical Insurance Cost Prediction

InsureAI is a Machine Learning project that predicts medical insurance charges based on personal information such as age, BMI, smoking habits, gender, number of children and region.

The project uses a K-Nearest Neighbors (KNN) Regression algorithm and a Streamlit frontend to make predictions.

## 🚀 Live Demo

[Open InsureAI](https://pojogszkhajvmdw2agjlcz.streamlit.app/)

## 📌 Project Overview

The main objective of this project is to understand how Machine Learning can be used to predict medical insurance costs using historical data.

Users can enter their information through a simple web interface and receive an estimated insurance cost.

## 🧠 Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Selection
6. Feature Scaling
7. Model Training using KNN Regression
8. Model Evaluation
9. Model Deployment using Streamlit

## 📊 Dataset

The project uses a medical insurance dataset containing information such as:

- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Region
- Insurance Charges


## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 🤖 Machine Learning Model

**Algorithm:** K-Nearest Neighbors (KNN) Regression

The model takes selected personal information as input and predicts estimated medical insurance charges.

Saved model files:

- `knn_model.pkl` – Trained KNN model
- `scaler.pkl` – Feature scaling object
- `columns.pkl` – Model input column order

## 📂 Project Structure

```text
InsureAI/
│
├── app.py
├── knn_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   └── insurance.csv
│
└── notebook/
    └── model_training.ipynb
```

This project is developed for educational purposes and demonstrates the application of Machine Learning.

The model uses a US medical insurance dataset. Any INR amount displayed is a currency conversion for presentation purposes, not an actual Indian insurance quotation.

## 👨‍💻 Author

Developed as a learning project to explore Machine Learning, Python and application deployment.
Developed as a learning project to explore Machine Learning, Python and application deployment.
