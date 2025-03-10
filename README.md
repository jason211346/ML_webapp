# Machine Learning Platform

This is a web-based machine learning platform built using Streamlit. It allows users to upload datasets, train machine learning models, and make predictions in an interactive and user-friendly environment.

## Features
- **User Management**: Create and select users to manage different datasets and models.
- **Data Upload & Display**: Upload and manage Excel or CSV files.
- **Model Training**: Train regression or classification models using various algorithms such as Linear Regression, Random Forest, XGBoost, and more.
- **Model Storage & Download**: Save trained models and download them for future use.
- **Prediction Module**: Use trained models to make predictions on new data.
- **Secure Access**: Password-protected access for enhanced security.

## Installation
To run the application locally, install the required dependencies and start the Streamlit app:

```sh
pip install streamlit pandas plotly scikit-learn xgboost lightgbm
streamlit run app.py
```

## Usage
1. Open the web application.
2. Log in using the password.
3. Select or create a new user.
4. Upload a dataset (Excel or CSV file).
5. Choose features and targets for model training.
6. Train a model and evaluate performance.
7. Save or download the trained model.
8. Use the prediction module to make predictions on new data.

## Web Application
Access the online version of the platform here: [Machine Learning Platform](https://mlwebapp-gispdntd.streamlit.app/)
