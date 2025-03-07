import streamlit as st
import pandas as pd
import os

import plotly.express as px
import pickle
from Train_2 import train_and_plot

# Function to upload file
def upload_file(directory):
    """Upload Excel or CSV file"""
    uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "csv"])
    if uploaded_file:
        upload_dir = directory
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        return file_path
    return None

def load_files(directory):
    """Load Excel and CSV files from directory"""
    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.csv'))]
    return files

def display_dataframe(file_path):
    """Display dataframe from Excel or CSV file"""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:  # CSV file
        df = pd.read_csv(file_path)
    return df



def save_model_to_directory(model, directory, filename):
    """Save model to specified directory with custom filename"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    return file_path

def load_models(directory):
    """Load models from directory"""
    models = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    return models

def load_model(model_path):
    """Load a model from file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def make_predictions(model, data):
    """Make predictions using the model"""
    return model.predict(data)

def load_names():
    """Load existing names from directories"""
    base_dir = "D:/Jason/webapp/ML/data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Add at the beginning of your file after imports
def create_new_user():
    with st.sidebar:
        dialog = st.empty()
        with dialog.container():
            st.write("Create New User")
            new_name = st.text_input("Enter username", key="new_user_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("OK"):
                    if new_name:
                        new_dir = f"D:/Jason/webapp/ML/data/{new_name}"
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                            st.session_state['show_dialog'] = False
                            st.session_state['users'] = load_names()
                            st.session_state['selected_user'] = new_name
                            dialog.empty()
                            return True
                        else:
                            st.error("This username already exists")
                    else:
                        st.error("Please enter a username")
            with col2:
                if st.button("Cancel"):
                    st.session_state['show_dialog'] = False
                    dialog.empty()
                    return False
    return False



# Streamlit app
st.title("Machine Learning Platform")

# Sidebar for selecting name
st.sidebar.header("User Selection")

# Initialize session state
if 'show_dialog' not in st.session_state:
    st.session_state['show_dialog'] = False
if 'users' not in st.session_state:
    st.session_state['users'] = load_names()
if 'selected_user' not in st.session_state:
    st.session_state['selected_user'] = None

# Add New User button
if st.sidebar.button("Add New User"):
    st.session_state['show_dialog'] = True

# Show dialog if button was clicked
if st.session_state['show_dialog']:
    if create_new_user():
        st.sidebar.success("New user created successfully!")

# User selection dropdown
name = st.sidebar.selectbox(
    "Select User",
    options=st.session_state['users'],
    index=st.session_state['users'].index(st.session_state['selected_user']) if st.session_state['selected_user'] in st.session_state['users'] else 0
) if st.session_state['users'] else None

if not st.session_state['users']:
    st.sidebar.warning("No users found. Please create a new user.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Management", "Model Training", "Prediction"])

with tab1:
    st.header("Data Upload and Display")
    # Set directory based on selected name
    directory = f"D:/Jason/webapp/ML/data/{name}"
    files = load_files(directory)
    selected_file = st.selectbox("Select an Excel file from directory", files)

    # Upload file
    uploaded_file_path = upload_file(directory)

    # Read file button
    if st.button("Read File"):
        if uploaded_file_path:
            df = display_dataframe(uploaded_file_path)
            st.session_state['df'] = df
        elif selected_file:
            df = display_dataframe(os.path.join(directory, selected_file))
            st.session_state['df'] = df

    # Display the dataframe if it exists
    if 'df' in st.session_state:
        st.write("Current Dataset:")
        st.write(st.session_state['df'])

with tab2:
    st.header("Model Training and Evaluation")
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Select task type
        task_type = st.selectbox(
            "Select Task Type",
            ["regression", "classification"],
            key='task_type'
        )
        
        # Select columns for X and Y
        columns = df.columns.tolist()
        x_columns = st.multiselect("Select Features (X)", columns, key='x_columns')
        y_columns = st.multiselect("Select Target (Y)", columns, key='y_columns')
        
        # Model selection based on task type
        if task_type == 'regression':
            model_options = [
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
                "LightGBM"
            ]
        else:
            model_options = [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
                "LightGBM"
            ]
            
        model_type = st.selectbox("Select Model Type", model_options, key='model_type')
        
        if st.button("Train Model"):
            model = train_and_plot(df, x_columns, y_columns, task_type, model_type)
            st.session_state['trained_model'] = model
        
        # Add download button if model exists in session state
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            col1, col2 = st.columns(2)
            
            with col1:
                if st.download_button(
                    label="Download Trained Model",
                    data=pickle.dumps(model),
                    file_name=f"{model_type.lower().replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                ):
                    st.success("Model downloaded successfully!")
            
            with col2:
                # Create a container for the save functionality
                with st.container():
                    custom_filename = st.text_input(
                        "Enter model filename (optional)", 
                        value=f"{model_type.lower().replace(' ', '_')}_model",
                        key="model_filename"
                    )
                    if st.button("Save Model to Directory"):
                        model_directory = f"D:/Jason/webapp/ML/model/{name}"
                        # Add .pkl extension if not present
                        if not custom_filename.endswith('.pkl'):
                            custom_filename += '.pkl'
                        # Create full file path
                        file_path = os.path.join(model_directory, custom_filename)
                        # Create directory if it doesn't exist
                        os.makedirs(model_directory, exist_ok=True)
                        # Save the model
                        with open(file_path, 'wb') as f:
                            pickle.dump(model, f)
                        st.success(f"Model saved successfully to: {file_path}")
    else:
        st.warning("Please upload or select a dataset in the Data Management tab first.")


# Add the new tab3 content:
with tab3:
    st.header("Model Prediction")
    
    # Create two columns for uploading
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload data for prediction
        pred_file = st.file_uploader("Upload Excel or CSV file for prediction", 
                                    type=["xlsx", "csv"], 
                                    key="pred_file")
        if pred_file:
            if pred_file.name.endswith('.xlsx'):
                pred_df = pd.read_excel(pred_file)
            else:  # CSV file
                pred_df = pd.read_csv(pred_file)
            st.session_state['pred_df'] = pred_df
            st.write("Prediction Data Preview:")
            st.write(pred_df)
    
    with col2:
        # Model selection/upload
        model_source = st.radio("Select model source", ["Upload Model", "Select Saved Model"])
        
        if model_source == "Upload Model":
            uploaded_model = st.file_uploader("Upload model file", type=["pkl"], key="model_upload")
            if uploaded_model:
                model = pickle.loads(uploaded_model.read())
                st.session_state['pred_model'] = model
                st.success("Model loaded successfully!")
        
        else:
            model_directory = f"D:/Jason/webapp/ML/model/{name}"
            if os.path.exists(model_directory):
                saved_models = load_models(model_directory)
                if saved_models:
                    selected_model = st.selectbox("Select a saved model", saved_models)
                    if selected_model:
                        model_path = os.path.join(model_directory, selected_model)
                        model = load_model(model_path)
                        st.session_state['pred_model'] = model
                        st.success("Model loaded successfully!")
                else:
                    st.warning("No saved models found in directory")
            else:
                st.warning("Model directory does not exist")

    # Prediction section
    if 'pred_df' in st.session_state and 'pred_model' in st.session_state:
        st.subheader("Make Predictions")
        
        # Select features for prediction
        available_features = st.session_state['pred_df'].columns.tolist()
        pred_features = st.multiselect(
            "Select features for prediction (make sure to match training features)",
            available_features,
            key='pred_features'
        )
        
        if pred_features and st.button("Make Predictions"):
            try:
                if task_type == 'regression':
                    pred_df = pred_df.fillna(pred_df.mean())
                X_pred = pred_df[pred_features]
                
                # Get predictions
                predictions = make_predictions(st.session_state['pred_model'], X_pred)
                
                # Handle multiple target predictions
                result_df = st.session_state['pred_df'].copy()
                
                if predictions.ndim > 1:  # If multiple target predictions
                    for i in range(predictions.shape[1]):
                        result_df[f'Prediction_{i+1}'] = predictions[:, i]
                else:  # If single target prediction
                    result_df['Prediction'] = predictions
                
                # Display results
                st.write("Prediction Results:")
                st.write(result_df)
                
                # Download results button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")