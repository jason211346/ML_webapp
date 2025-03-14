import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    classification_report,
    confusion_matrix
)
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def train_and_plot(df, x_columns, y_columns, task_type, model_type):
    """
    Train model and plot performance metrics with standardization
    task_type: 'regression' or 'classification'
    """
    # Handle missing values
    if task_type == 'regression':
        df = df.fillna(df.mean())
    
    X = df[x_columns]
    y = df[y_columns]

    # Initialize scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler() if task_type == 'regression' else None
    
    # Scale features (X)
    X_scaled = x_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=x_columns)
    
    # Handle classification and regression differently for y
    if task_type == 'classification':
        if len(y_columns) == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.write("Classes:", dict(enumerate(le.classes_)))
            class_encoder = le
        else:
            y_encoded = pd.DataFrame()
            class_encoders = {}
            for col in y_columns:
                le_col = LabelEncoder()
                y_encoded[col] = le_col.fit_transform(y[col])
                class_encoders[col] = le_col
            y = y_encoded
            class_encoder = class_encoders
    else:  # regression
        # Scale target variable(s) for regression
        y = y_scaler.fit_transform(y)

    # Split the scaled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model selection based on task type and model type
    models = {
        'regression': {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42)
        },
        'classification': {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42),
            "LightGBM": LGBMClassifier(random_state=42)
        }
    }
    
    model = models[task_type][model_type]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate model based on task type
    if task_type == 'regression':
        # Inverse transform predictions and actual values for proper evaluation
        y_train_orig = y_scaler.inverse_transform(y_train)
        y_test_orig = y_scaler.inverse_transform(y_test)
        y_train_pred_orig = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
        y_test_pred_orig = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
        
        train_score = r2_score(y_train_orig, y_train_pred_orig)
        test_score = r2_score(y_test_orig, y_test_pred_orig)
        train_error = mean_squared_error(y_train_orig, y_train_pred_orig)
        test_error = mean_squared_error(y_test_orig, y_test_pred_orig)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Train R²', 'Test R²', 'Train MSE', 'Test MSE'],
            'Value': [train_score, test_score, train_error, test_error]
        })
        
        st.write(f"Train R²: {train_score:.2f}")
        st.write(f"Test R²: {test_score:.2f}")
        st.write(f"Train MSE: {train_error:.2f}")
        st.write(f"Test MSE: {test_error:.2f}")

    else:  # classification (no need to inverse transform)
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        st.write("Classification Report:")
        cm = confusion_matrix(y_test, y_test_pred)
        fig_cm = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual"),
                          title="Confusion Matrix",
                          color_continuous_scale="Viridis")
        st.plotly_chart(fig_cm)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Train Accuracy', 'Test Accuracy'],
            'Value': [train_score, test_score]
        })
    # Plot performance metrics
    fig = px.bar(metrics_df, x='Metric', y='Value', 
                 title=f'Model Performance Metrics - {model_type}')
    st.plotly_chart(fig)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)[0]
    else:
        return model


    importance_df = pd.DataFrame({
        'Feature': x_columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 title='Feature Importance', orientation='h')
    st.plotly_chart(fig)
    model_data = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler if task_type == 'regression' else None,
        'class_encoder': class_encoder if task_type == 'classification' else None,
        'x_columns': x_columns,
        'y_columns': y_columns,
        'task_type': task_type
    }
    
    return model_data