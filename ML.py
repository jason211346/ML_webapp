import streamlit as st
import pandas as pd
import os
import hmac
import pickle
from Train import train_and_plot  # Assuming this function exists
from GPyOpt.methods import BayesianOptimization
import multiprocessing as mp
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

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



def save_model_to_directory(model_data, directory, filename):  # Changed to accept model_data
    """Save model to specified directory with custom filename"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)  # Save model_data
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

# def make_predictions(model, data, x_columns): # Added x_columns argument
#     """Make predictions using the model, handling missing columns."""
#     # Check for missing columns
#     missing_cols = set(x_columns) - set(data.columns)
#     if missing_cols:
#         st.error(f"Missing columns in prediction data: {missing_cols}")
#         return None  # Or raise an exception

#     try:
#         # Select only required columns and ensure correct order
#         data_for_prediction = data[x_columns]
#         return model.predict(data_for_prediction)
#     except Exception as e:
#             st.error(f"Prediction error: {e}")
#             return None
# def make_predictions(model_data, data, x_columns):
#     """Make predictions using the model with proper scaling."""
#     model = model_data['model']
#     x_scaler = model_data['x_scaler']
#     y_scaler = model_data['y_scaler']
    
#     try:
#         # 選擇需要的列並確保順序正確
#         data_for_prediction = data[x_columns]
        
#         # 標準化輸入數據
#         X_scaled = x_scaler.transform(data_for_prediction)
        
#         # 進行預測
#         scaled_predictions = model.predict(X_scaled)
        
#         # 反標準化預測結果
#         predictions = y_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
        
#         return predictions
        
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

def make_predictions(model_data, data, x_columns):
    """Make predictions using the model with proper scaling for both classification and regression."""
    model = model_data['model']
    x_scaler = model_data['x_scaler']
    y_scaler = model_data['y_scaler']
    class_encoder = model_data['class_encoder']
    task_type = model_data['task_type']
    
    try:
        # Select required columns and ensure correct order
        data_for_prediction = data[x_columns]
        
        
        
        # Handle predictions based on task type
        if task_type == 'classification':
            # Scale input data
            #             
            # Make predictions
            predictions = model.predict(data_for_prediction)
            if isinstance(class_encoder, dict):  # Multiple target columns
                final_predictions = pd.DataFrame()
                for col, encoder in class_encoder.items():
                    final_predictions[col] = encoder.inverse_transform(predictions[col])
                return final_predictions
            else:  # Single target column
                return class_encoder.inverse_transform(predictions)
        else:  # regression
            # Inverse transform predictions for regression
            # Scale input data
            X_scaled = x_scaler.transform(data_for_prediction)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            return y_scaler.inverse_transform(predictions.reshape(-1, 1))
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

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

# -------------------- INVERSE PREDICTION FUNCTIONALITY --------------------


# def inverse_predict(model_data, target_y, x_columns):
#     """Uses Bayesian Optimization to find X values with proper scaling."""
#     st.write("Starting inverse prediction process...")
    
#     try:
#         model = model_data['model']
#         x_scaler = model_data['x_scaler']
#         y_scaler = model_data['y_scaler']
#         st.write("✓ Model components loaded successfully")
        
#         # Create a container for the optimization progress
#         optimization_container = st.empty()
        
#         def objective_function(x):
#             """Objective function for Bayesian Optimization"""
#             try:
#                 x_reshaped = x.reshape(1, -1)
#                 x_df = pd.DataFrame(x_reshaped, columns=x_columns)
#                 x_scaled = x_scaler.transform(x_df)
                
#                 predicted_y_scaled = model.predict(x_scaled)
#                 predicted_y = y_scaler.inverse_transform(predicted_y_scaled.reshape(-1, 1))
                
#                 total_error = 0
#                 for idx, (key, target_value) in enumerate(target_y.items()):
#                     target_scaled = y_scaler.transform([[target_value]])[0]
#                     predicted_scaled = predicted_y_scaled[0]
#                     error = (predicted_scaled - target_scaled) ** 2
#                     total_error += error
                
#                 return float(total_error)
                
#             except Exception as e:
#                 st.error(f"Error in objective function: {e}")
#                 return 1e10
        
#         st.write("✓ Objective function defined")
#         bounds = []
#         for col in x_columns:
#             bounds.append({'name': col, 'type': 'continuous', 'domain': (0, 10)})
#         st.write("✓ Optimization bounds set")
        
#         # Initialize optimizer
#         st.write("Initializing Bayesian Optimization...")
#         optimizer = BayesianOptimization(
#             f=objective_function,
#             domain=bounds,
#             model_type='GP',
#             acquisition_type='EI',
#             normalize_Y=True,
#             initial_design_numdata=20,
#             exact_feval=True
#         )
#         st.write("✓ Optimizer initialized")
        
#         # Run optimization with progress tracking
#         st.write("Starting optimization process...")
#         max_iter = 100
        
#         with optimization_container.container():
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for i in range(max_iter):
#                 optimizer.run_optimization(max_iter=1)
#                 progress = (i + 1) / max_iter
#                 progress_bar.progress(progress)
#                 current_min = optimizer.fx_opt
#                 status_text.text(f'Iteration {i+1}/{max_iter}: Best error = {current_min:.6f}')
                
#                 if current_min < 1e-6:
#                     st.write("✓ Convergence achieved early")
#                     break
        
#         st.write("✓ Optimization completed")
        
#         # Process results
#         optimized_x = optimizer.x_opt
#         result_df = pd.DataFrame([optimized_x], columns=x_columns)
#         predicted_y = make_predictions(model_data, result_df, x_columns)
        
#         # Add predictions and targets
#         for i, col in enumerate(model_data['y_columns']):
#             result_df[f'predicted_{col}'] = predicted_y[0][i]
#             result_df[f'target_{col}'] = target_y[col]
        
#         # Add optimization metrics
#         result_df['final_error'] = optimizer.fx_opt
#         result_df['iterations'] = optimizer.num_acquisitions
        
#         st.write("✓ Results processed successfully")
#         return result_df
        
#     except Exception as e:
#         st.error(f"Optimization failed: {str(e)}")
#         st.write("Stack trace:", format_exc())
#         return None

# def inverse_predict_batch(model_data, target_y_batch, x_columns):
#     """Parallel batch processing for inverse prediction"""
#     status_placeholder = st.empty()
#     progress_placeholder = st.empty()
    
#     # Create a progress bar
#     with progress_placeholder:
#         progress_bar = st.progress(0)
    
#     results = []
#     total = len(target_y_batch)
    
#     # Process targets sequentially with progress updates
#     for idx, target_y in enumerate(target_y_batch):
#         status_placeholder.text(f"Processing target {idx + 1}/{total}")
        
#         result = inverse_predict_single_target(
#             target_y=target_y,
#             model_data=model_data,
#             x_columns=x_columns
#         )
        
#         if result is not None:
#             results.append(result)
            
#         # Update progress
#         progress = (idx + 1) / total
#         progress_bar.progress(progress)
        
#     # Clear temporary UI elements
#     status_placeholder.empty()
#     progress_placeholder.empty()
    
#     successful = len(results)
#     st.write(f"✓ Batch completed: {successful}/{total} successful predictions")
    
#     return results

def inverse_predict_single_target(target_y, model_data, x_columns):
    """Optimize for a single target Y value"""
    try:
        model = model_data['model']
        x_scaler = model_data['x_scaler']
        y_scaler = model_data['y_scaler']
        
        def objective_function(x):
            try:
                x_reshaped = x.reshape(1, -1)
                x_df = pd.DataFrame(x_reshaped, columns=x_columns)
                x_scaled = x_scaler.transform(x_df)
                predicted_y_scaled = model.predict(x_scaled)
                total_error = 0
                for idx, (key, target_value) in enumerate(target_y.items()):
                    target_scaled = y_scaler.transform([[target_value]])[0]
                    predicted_scaled = predicted_y_scaled[0]
                    error = (predicted_scaled - target_scaled) ** 2
                    total_error += error
                return float(total_error)
            except Exception as e:
                return 1e10

        bounds = []
        for col in x_columns:
            bounds.append({'name': col, 'type': 'continuous', 'domain': (0, 10)})

        optimizer = BayesianOptimization(
            f=objective_function,
            domain=bounds,
            model_type='GP',
            acquisition_type='EI',
            normalize_Y=True,
            initial_design_numdata=20,
            exact_feval=True
        )
        
        max_iter = 50
        optimizer.run_optimization(max_iter=max_iter)
        
        optimized_x = optimizer.x_opt
        final_error = optimizer.fx_opt
        
        result_df = pd.DataFrame([optimized_x], columns=x_columns)
        predicted_y = make_predictions(model_data, result_df, x_columns)
        
        for i, col in enumerate(model_data['y_columns']):
            result_df[f'predicted_{col}'] = predicted_y[0][i]
            result_df[f'target_{col}'] = target_y[col]
        
        result_df['final_error'] = optimizer.fx_opt
        return result_df
        
    except Exception as e:
        return None

def inverse_predict_parallel(model_data, target_y_batch, x_columns):
    """Process a single target Y using multiprocessing"""
    try:
        result = inverse_predict_single_target(target_y_batch, model_data, x_columns)
        return result
    except Exception as e:
        return None

def inverse_predict_batch(model_data, target_y_batch, x_columns):
    """Parallel batch processing using multiprocessing"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    total = len(target_y_batch)
    
    with progress_placeholder:
        progress_bar = st.progress(0)
    
    # 使用CPU核心數量的進程
    num_processes = mp.cpu_count()
    st.write(f"Using {num_processes} CPU cores for parallel processing")
    
    results = []
    with Pool(processes=(num_processes//2)) as pool:
        # 創建異步結果
        async_results = []
        for target_y in target_y_batch:
            async_results.append(
                pool.apply_async(inverse_predict_parallel, 
                               args=(model_data, target_y, x_columns))
            )
        
        # 監控進度
        completed = 0
        for async_result in async_results:
            result = async_result.get()  # 等待結果
            if result is not None:
                results.append(result)
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_placeholder.text(f"Processed {completed}/{total} targets")
    
    # 清理UI元素
    status_placeholder.empty()
    progress_placeholder.empty()
    
    successful = len(results)
    st.write(f"✓ Batch completed: {successful}/{total} successful predictions")
    
    return results


if __name__ == "__main__":

    # --- Authentication ---
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

    # --- App Layout ---
    st.title("Machine Learning Platform")
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
    st.sidebar.subheader("Project Selection")

    # --- Session State Initialization ---
    if 'show_dialog' not in st.session_state:
        st.session_state['show_dialog'] = False
    if 'Project' not in st.session_state:
        st.session_state['Project'] = load_names()
    if 'selected_project' not in st.session_state:
        st.session_state['selected_project'] = None
    # Removed x_columns and y_columns from session_state

    # --- Add New Project ---
    if st.sidebar.button("Add New Project"):
        st.session_state['show_dialog'] = True

    if st.session_state['show_dialog']:
        if create_new_Project():
            st.sidebar.success("New Project created successfully!")

    # --- Project Selection ---
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
        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["Data Management", "Model Training", "Prediction", "Inverse Prediction"])

        with tab1:
            st.header("Data Upload and Display")
            directory = os.path.join(os.getcwd(), username, name, "data")
            files = load_files(directory)
            selected_file = st.selectbox("Select an Excel file from directory", files)

            col1, col2 = st.columns([3, 1])
            with col1:
                uploaded_file_path = upload_file(directory)
            with col2:
                if selected_file and st.button("Delete Selected File"):
                    file_path = os.path.join(directory, selected_file)
                    if delete_file(file_path):
                        st.success(f"File {selected_file} deleted successfully!")
                        st.rerun()

            if st.button("Read File"):
                if uploaded_file_path:
                    df = display_dataframe(uploaded_file_path)
                    st.session_state['df'] = df
                elif selected_file:
                    df = display_dataframe(os.path.join(directory, selected_file))
                    st.session_state['df'] = df

            if 'df' in st.session_state:
                st.write("Current Dataset:")
                st.write(st.session_state['df'])

        with tab2:
            st.header("Model Training and Evaluation")
            if 'df' in st.session_state:
                df = st.session_state['df']

                task_type = st.selectbox(
                    "Select Task Type",
                    ["regression", "classification"],
                    key='task_type'
                )

                columns = df.columns.tolist()
                x_columns = st.multiselect("Select Features (X)", columns, key='x_columns_multiselect')
                y_columns = st.multiselect("Select Target (Y)", columns, key='y_columns_multiselect')

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
                    model_data = train_and_plot(df, x_columns, y_columns, task_type, model_type)
                    # Store model, x_columns, and y_columns as a dictionary
                    # model_data = {
                    #     'model': model,
                    #     'x_columns': x_columns,
                    #     'y_columns': y_columns,
                    #     'task_type': task_type  # Store the task type
                    # }
                    st.session_state['trained_model'] = model_data # Store model_data
                    # No longer storing x_columns and y_columns directly in session_state

                if 'trained_model' in st.session_state:
                    model_data = st.session_state['trained_model'] # Retrieve model_data
                    model = model_data['model'] # Extract model
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.download_button(
                            label="Download Trained Model",
                            data=pickle.dumps(model_data),  # Download model_data
                            file_name=f"{model_type.lower().replace(' ', '_')}_model.pkl",
                            mime="application/octet-stream"
                        ):
                            st.success("Model downloaded successfully!")

                    with col2:
                        with st.container():
                            custom_filename = st.text_input(
                                "Enter model filename (optional)",
                                value=f"{model_type.lower().replace(' ', '_')}_model",
                                key="model_filename"
                            )
                            model_directory = os.path.join(os.getcwd() ,username , name, "model")
                            if st.button("Save Model to Directory"):
                                file_path = os.path.join(model_directory, custom_filename + ".pkl")
                                os.makedirs(model_directory, exist_ok=True)
                                with open(file_path, 'wb') as f:
                                    pickle.dump(model_data, f)  # Save model_data
                                st.success(f"Model saved successfully to: {file_path}")
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
        with tab3:  # Prediction Tab
           st.header("Model Prediction")

           # Model Selection (Keep this part similar to before, but load model_data)
           model_source = st.radio("Select model source", ["Upload Model", "Select Saved Model"])
           if model_source == "Upload Model":
               uploaded_model = st.file_uploader("Upload model file", type=["pkl"], key="model_upload")
               if uploaded_model:
                   model_data = pickle.loads(uploaded_model.read())
                   st.session_state['pred_model'] = model_data
                   st.success("Model loaded successfully!")
           else:
               model_directory = os.path.join(os.getcwd(), username, name, "model")
               if os.path.exists(model_directory):
                   saved_models = load_models(model_directory)
                   if saved_models:
                       selected_model = st.selectbox("Select a saved model", saved_models)
                       if selected_model:
                           model_path = os.path.join(model_directory, selected_model)
                           model_data = load_model(model_path)
                           st.session_state['pred_model'] = model_data
                           st.success("Model loaded successfully!")
                   else:
                       st.warning("No saved models found in directory")
               else:
                   st.warning("Model directory does not exist")


           if 'pred_model' in st.session_state:
                model_data = st.session_state['pred_model']
                model = model_data['model']
                x_columns = model_data['x_columns']
                task_type = model_data['task_type']

                # File Upload for Prediction
                pred_file = st.file_uploader("Upload Excel or CSV file for prediction", 
                                            type=["xlsx", "csv"], key="pred_file")
                if pred_file:
                    if pred_file.name.endswith('.xlsx'):
                        pred_df = pd.read_excel(pred_file)
                    else:
                        pred_df = pd.read_csv(pred_file)

                    if set(x_columns).issubset(pred_df.columns):
                        st.write("Prediction Data Preview:")
                        st.write(pred_df)

                        if st.button("Make Predictions"):
                            predictions = make_predictions(model_data, pred_df, x_columns)
                            if predictions is not None:
                                if task_type == 'classification':
                                    # Handle classification predictions
                                    if isinstance(predictions, pd.DataFrame):  # Multiple target columns
                                        for col in predictions.columns:
                                            pred_df[f'{col}_pred'] = predictions[col]
                                    else:  # Single target column
                                        pred_df[f'{model_data["y_columns"][0]}_pred'] = predictions
                                else:  # regression
                                    # Handle regression predictions
                                    for i, col in enumerate(model_data['y_columns']):
                                        pred_df[f'{col}_pred'] = predictions[:, i]
                                
                                st.write("Prediction Results:")
                                st.write(pred_df)

                                # Add download button
                                csv = pred_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Predictions as CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                )
                    else:
                        st.error(f"Uploaded file is missing required X columns: {set(x_columns) - set(pred_df.columns)}")

        # -------------------- INVERSE PREDICTION TAB --------------------
        with tab4:
            st.header("Inverse Prediction")

            # Load the model if available (similar to prediction tab)
            if 'pred_model' in st.session_state:  # Use 'pred_model' - consistent naming
                model_data = st.session_state['pred_model']
                model = model_data['model']
                x_columns = model_data['x_columns']
                y_columns = model_data['y_columns']
                task_type = model_data['task_type'] # Get the task type


                st.subheader("Upload Y Values for Inverse Prediction")
                inverse_file = st.file_uploader("Upload Excel or CSV with target Y values", type=["xlsx", "csv"])

                if inverse_file:
                    try:
                        if inverse_file.name.endswith('.xlsx'):
                            inverse_df = pd.read_excel(inverse_file)
                        else:
                            inverse_df = pd.read_csv(inverse_file)

                        # --- KEY CHANGE: Check against loaded y_columns ---
                        if set(y_columns).issubset(inverse_df.columns):
                            st.write("Uploaded Target Y Values:")
                            st.write(inverse_df)

                            # 在tab4中修改按鈕處理部分:
                        # In tab4, update the button handler:
                        # In tab4, the button handler remains the same
                        if st.button("Predict X from Y"):
                            overall_progress = st.progress(0)
                            status_text = st.empty()
                            
                            batch_size = 500
                            total_batches = (len(inverse_df) + batch_size - 1) // batch_size
                            all_inverse_predictions = []
                            
                            st.write(f"Starting inverse prediction for {len(inverse_df)} rows")
                            
                            for batch_idx in range(total_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, len(inverse_df))
                                batch_df = inverse_df.iloc[start_idx:end_idx]
                                
                                status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}")
                                
                                target_y_batch = [
                                    {y_col: row[y_col] for y_col in y_columns}
                                    for _, row in batch_df.iterrows()
                                ]
                                
                                batch_results = inverse_predict_batch(model_data, target_y_batch, x_columns)
                                all_inverse_predictions.extend(batch_results)
                                
                                overall_progress.progress((batch_idx + 1) / total_batches)
                            
                            status_text.empty()
                            
                            if all_inverse_predictions:
                                combined_df = pd.concat(all_inverse_predictions, ignore_index=True)
                                st.success("Inverse prediction completed!")
                                st.write(f"Total successful predictions: {len(combined_df)}/{len(inverse_df)}")
                                st.write(combined_df)
                                
                                csv = combined_df.to_csv(index=False)
                                st.download_button(
                                    label="Download All Inverse Predictions as CSV",
                                    data=csv,
                                    file_name="inverse_predictions.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.error(f"Uploaded file must contain all target Y columns: {y_columns}")
                    except Exception as e:
                        st.error(f"Error processing uploaded file: {e}")

            else: # Model not selected/loaded
                st.warning("Please load a trained model first in Model Training tab.")