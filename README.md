# Customer Churn Prediction

A beginner-friendly Data Science and Machine Learning project that predicts whether a customer is likely to churn.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── customer_churn.csv
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── churn_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate dataset

```powershell
python src/data_preprocessing.py
```

### 4. Train model

```powershell
python src/train_model.py
```

### 5. Run Streamlit app

```powershell
streamlit run app.py
```

## Machine Learning

The project uses a Random Forest classifier with preprocessing for numerical and categorical features.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Future Improvements

- Compare more machine learning algorithms
- Use a real-world churn dataset
- Add model explainability with SHAP
- Add customer retention recommendations
- Deploy the Streamlit application
