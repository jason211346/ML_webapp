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

def train_and_plot(df, x_columns, y_columns, task_type, model_type):
    """
    Train model and plot performance metrics
    task_type: 'regression' or 'classification'
    """
    # Handle missing values
    # breakpoint()
    if task_type == 'regression':
        df = df.fillna(df.mean())
    
    X = df[x_columns]
    y = df[y_columns] # Convert to 1D array for classification
    
    # Add label encoding for classification tasks
    if task_type == 'classification':
        le = LabelEncoder()
        # Handle single or multiple target columns
        if len(y_columns) == 1:
            y = le.fit_transform(y)
            st.write("Classes:", dict(enumerate(le.classes_)))
        else:
            # Create a new DataFrame for encoded targets
            y_encoded = pd.DataFrame()
            class_mappings = {}
            
            # Encode each target column separately
            for col in y_columns:
                le_col = LabelEncoder()
                y_encoded[col] = le_col.fit_transform(y[col])
                class_mappings[col] = dict(enumerate(le_col.classes_))
            
            y = y_encoded
            st.write("Class mappings for each target:", class_mappings)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        train_error = mean_squared_error(y_train, y_train_pred)
        test_error = mean_squared_error(y_test, y_test_pred)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Train R²', 'Test R²', 'Train MSE', 'Test MSE'],
            'Value': [train_score, test_score, train_error, test_error]
        })
        
        st.write(f"Train R²: {train_score:.2f}")
        st.write(f"Test R²: {test_score:.2f}")
        st.write(f"Train MSE: {train_error:.2f}")
        st.write(f"Test MSE: {test_error:.2f}")

    else:  # classification
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        st.write("Classification Report:")
        # st.text(classification_report(y_test, y_test_pred))
        
        # Plot confusion matrix
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

    return model