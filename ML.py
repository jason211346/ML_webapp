import streamlit as st
import pandas as pd
import os

import hmac

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

def delete_file(file_path):
    """Delete a file"""
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False

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
    # base_dir = "D:/Jason/webapp/ML/data"
    base_dir = os.path.join(os.getcwd(), username)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Add at the beginning of your file after imports
def create_new_Project():
    with st.sidebar:
        dialog = st.empty()
        with dialog.container():
            st.write("Create New Project")
            new_name = st.text_input("Enter Projectname", key="new_Project_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("OK"):
                    if new_name:
                        # new_dir = f"D:/Jason/webapp/ML/data/{new_name}"
                        new_dir = os.path.join(os.getcwd(), username, new_name, "data")
                        
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                            st.session_state['show_dialog'] = False
                            st.session_state['Project'] = load_names()
                            st.session_state['selected_project'] = new_name
                            dialog.empty()
                            return True
                        else:
                            st.error("This Projectname already exists")
                    else:
                        st.error("Please enter a Projectname")
            with col2:
                if st.button("Cancel"):
                    st.session_state['show_dialog'] = False
                    dialog.empty()
                    return False
    return False

# def check_password():
#     """Returns `True` if the Project had the correct password."""
 
#     def password_entered():
#         """Checks whether a password entered by the Project is correct."""
#         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#         else:
#             st.session_state["password_correct"] = False
 
#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True
 
#     # Show input for password.
#     st.text_input(
#         "Password", type="password", on_change=password_entered, key="password"
#     )
#     if "password_correct" in st.session_state:
#         st.error("😕 Password incorrect")
#     return False
 
def check_password():
    """Returns `True` if the User entered a valid username and password."""
    
    def password_entered():
        """Checks whether the entered username and password are correct."""
        username = st.session_state.get("username")
        password = st.session_state.get("password")
        
        # 檢查所輸入的使用者名稱和密碼是否與 secrets.toml 中的資料匹配
        if username in st.secrets and hmac.compare_digest(password, st.secrets[username]["password"]):
            st.session_state["authentication_status"] = True
            st.session_state["authenticated_user"] = username
        else:
            st.session_state["authentication_status"] = False
        # 清除密碼欄位，不在 session_state 中保存密碼
        del st.session_state["password"]

    # 如果已驗證成功，返回 True
    if st.session_state.get("authentication_status", False):
        return True

    # 顯示使用者名稱和密碼輸入欄位
    username = st.text_input("username", key="username")
    password = st.text_input("Password", type="password", on_change=password_entered, key="password")

    # 提示使用者輸入正確的使用者名稱和密碼
    if "authentication_status" in st.session_state:
        if st.session_state["authentication_status"] == False:
            st.error("😕 使用者名稱或密碼錯誤")
        else:
            st.warning("請輸入使用者名稱和密碼")
    
    return False

def logout():
    """Clear all session state variables and log out"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    return True
def show_logout_dialog():
    """Show a confirmation dialog for logout"""
    if 'show_logout_dialog' not in st.session_state:
        st.session_state['show_logout_dialog'] = False

    if st.session_state.get('show_logout_dialog', False):
        with st.sidebar:
            dialog = st.empty()
            with dialog.container():
                st.write("Confirm Logout")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirm"):
                        if logout():
                            st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state['show_logout_dialog'] = False
                        dialog.empty()

def show_delete_project_dialog(project_name):
    """Show a confirmation dialog for project deletion"""
    if 'show_delete_dialog' not in st.session_state:
        st.session_state['show_delete_dialog'] = False

    if st.session_state.get('show_delete_dialog', False):
        with st.sidebar:
            dialog = st.empty()
            with dialog.container():
                st.write(f"Delete Project: {project_name}?")
                st.write("⚠️ This action cannot be undone!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Delete", type="primary"):
                        project_path = os.path.join(os.getcwd(), username, project_name)
                        try:
                            import shutil
                            shutil.rmtree(project_path)
                            st.session_state['Project'] = load_names()
                            st.session_state['selected_project'] = None
                            st.session_state['show_delete_dialog'] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting project: {str(e)}")
                with col2:
                    if st.button("Cancel"):
                        st.session_state['show_delete_dialog'] = False
                        dialog.empty()

if __name__ == "__main__":

    check = check_password()
    if not check:
        st.stop()

    username = st.session_state["authenticated_user"]
    st.set_page_config(
        page_title="GIS機器學習平台",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
                'About': "GIS-PD-NTD 開發的機器學習平台"
            }
        )

    # Streamlit app
    st.title("Machine Learning Platform")

    # Sidebar for selecting name
    st.sidebar.header("Project Selection")

    col1, col2 = st.sidebar.columns([5,1])
    with col1:
        st.header(f"👤 {username}")
    with col2:
        if st.button("🚪", help="Logout"):
            st.session_state['show_logout_dialog'] = True

    if st.session_state.get('show_logout_dialog', False):
        show_logout_dialog()

    st.sidebar.markdown("---")  # Add separator line

    # Project section header
    st.sidebar.subheader("Project Selection")


    # Initialize session state
    if 'show_dialog' not in st.session_state:
        st.session_state['show_dialog'] = False
    if 'Project' not in st.session_state:
        st.session_state['Project'] = load_names()
    if 'selected_project' not in st.session_state:
        st.session_state['selected_project'] = None

    # Add New Project button
    if st.sidebar.button("Add New Project"):
        st.session_state['show_dialog'] = True

    # Show dialog if button was clicked
    if st.session_state['show_dialog']:
        if create_new_Project():
            st.sidebar.success("New Project created successfully!")

        # Project selection section
    col1, col2 = st.sidebar.columns([6,1])
    with col1:
        name = st.selectbox(
            "Select Project",
            options=st.session_state['Project'],
            index=st.session_state['Project'].index(st.session_state['selected_project']) 
            if st.session_state['selected_project'] in st.session_state['Project'] else 0
        ) if st.session_state['Project'] else None
    with col2:
        if name and st.button("🗑️", help="Delete Project"):
            st.session_state['show_delete_dialog'] = True

    if st.session_state.get('show_delete_dialog', False):
        show_delete_project_dialog(name)

    if not st.session_state['Project']:
        st.sidebar.warning("No Project found. Please create a new Project.")


    else:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Data Management", "Model Training", "Prediction"])

        with tab1:
            st.header("Data Upload and Display")
            # Set directory based on selected name
            # directory = f"D:/Jason/webapp/ML/data/{name}"
            directory = os.path.join(os.getcwd(), username, name, "data")
            files = load_files(directory)
            selected_file = st.selectbox("Select an Excel file from directory", files)

            # Add delete button for selected file
            col1, col2 = st.columns([3, 1])
            with col1:
                # Existing upload file functionality
                uploaded_file_path = upload_file(directory)
            with col2:
                if selected_file and st.button("Delete Selected File"):
                    file_path = os.path.join(directory, selected_file)
                    if delete_file(file_path):
                        st.success(f"File {selected_file} deleted successfully!")
                        st.rerun()
            # # Upload file
            # uploaded_file_path = upload_file(directory)

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
                            model_directory = os.path.join(os.getcwd() ,username , name, "model")
                            if st.button("Save Model to Directory"):
                                # model_directory = f"D:/Jason/webapp/ML/model/{name}"
                                
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
                            # Add delete button for saved models
                            if os.path.exists(model_directory) and load_models(model_directory):
                                st.markdown("---")
                                saved_models = load_models(model_directory)
                                model_to_delete = st.selectbox(
                                    "Select model to delete",
                                    saved_models,
                                    key="model_to_delete"
                                )
                                if st.button("Delete Selected Model"):
                                    model_path = os.path.join(model_directory, model_to_delete)
                                    if delete_file(model_path):
                                        st.success(f"Model {model_to_delete} deleted successfully!")
                                        st.rerun()
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
                    # model_directory = f"D:/Jason/webapp/ML/model/{name}"
                    model_directory = os.path.join(os.getcwd() ,username , name, "model")
                    if os.path.exists(model_directory):
                        saved_models = load_models(model_directory)
                        if saved_models:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                selected_model = st.selectbox("Select a saved model", saved_models)
                                if selected_model:
                                    model_path = os.path.join(model_directory, selected_model)
                                    model = load_model(model_path)
                                    st.session_state['pred_model'] = model
                                    st.success("Model loaded successfully!")
                            with col2:
                                if st.button("Delete Model"):
                                    if delete_file(model_path):
                                        st.success(f"Model {selected_model} deleted successfully!")
                                        st.rerun()
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

