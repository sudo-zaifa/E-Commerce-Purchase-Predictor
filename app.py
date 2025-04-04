import flask
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
import sklearn
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import warnings
import platform
import io
import base64
import os
from datetime import datetime, timedelta
import time
import random
from matplotlib.figure import Figure
import threading
import uuid
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store session data
global_data = {
    'progress': 0,
    'status': 'Initializing...',
    'data': None,
    'processed_data': None,
    'model': None,
    'results': None,
    'best_model_name': None,
    'importance_df': None,
    'X_test': None,
    'y_test': None,
    'scaler': None,
    'numeric_features': None,
    'label_encoders': None,
    'predict_function': None,
    'current_process': None,
    'task_id': None,
    'task_status': {},
    'user_analytics': None
}

# Generate a timestamp for the current session
session_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
user_login = "admin"  # Using the user login provided

# Functions for data processing, model training, etc.
def update_status(message, color='white'):
    """Update status message"""
    global_data['status'] = message
    print(f"[{color.upper()}] {message}")

def update_progress(value, max_value=100):
    """Update progress value"""
    global_data['progress'] = int(value / max_value * 100)

def create_sample_ecommerce_dataset(num_records=10000, output_file='ecommerce_sample_data.csv'):
    """Generate a sample e-commerce dataset and save it to a file"""
    update_status("Generating detailed sample e-commerce dataset...", "cyan")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create user IDs
    user_ids = list(range(1000, 2000))
    
    # Create event types with weights
    event_types = ['view', 'cart', 'purchase']
    event_weights = [0.7, 0.2, 0.1]
    
    # Create product categories
    categories = {
        1: "Electronics",
        2: "Clothing",
        3: "Home & Garden",
        4: "Sports",
        5: "Books",
        6: "Toys",
        7: "Health & Beauty",
        8: "Automotive",
        9: "Jewelry",
        10: "Food & Grocery"
    }
    
    # Define possible device types
    devices = ['mobile', 'desktop', 'tablet']
    device_weights = [0.6, 0.3, 0.1]
    
    # Define possible browsers
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
    browser_weights = [0.6, 0.2, 0.15, 0.04, 0.01]
    
    # Define possible locations
    locations = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan', 'India', 'Brazil', 'Mexico']
    location_weights = [0.3, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]
    
    # Generate base timestamp (30 days from now)
    base_timestamp = datetime.now() - timedelta(days=30)
    
    # Function to generate realistic timestamps
    def generate_timestamp():
        # Random day in past 30 days
        days = random.randint(0, 29)
        # Higher probability during business hours
        hour = np.random.choice(
            range(24), 
            p=[0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.09, 
               0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.07, 0.08, 0.09, 0.07, 
               0.05, 0.04, 0.03, 0.02]
        )
        # Random minute and second
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_timestamp + timedelta(days=days, hours=hour, minutes=minute, seconds=second)
    
    # Generate session IDs for each user
    user_sessions = {}
    for user_id in user_ids:
        # Generate 1-5 sessions per user
        num_sessions = random.randint(1, 5)
        user_sessions[user_id] = [random.randint(10000, 50000) for _ in range(num_sessions)]
    
    # Initialize data structure
    data = {
        'user_id': [],
        'event_time': [],
        'event_type': [],
        'product_id': [],
        'category_id': [],
        'category_name': [],
        'user_session': [],
        'price': [],
        'device': [],
        'browser': [],
        'location': [],
        'time_on_page': []
    }
    
    update_status("Generating user sessions and events...", "cyan")
    
    # Generate data for each user
    for user_id in user_ids:
        user_loc = np.random.choice(locations, p=location_weights)
        user_device = np.random.choice(devices, p=device_weights)
        user_browser = np.random.choice(browsers, p=browser_weights)
        
        for session_id in user_sessions[user_id]:
            # Generate 1-20 events per session
            num_events = random.randint(1, 20)
            
            # Start with 'view' events
            session_events = ['view'] * num_events
            
            # Add cart and purchase events based on probability
            if random.random() < 0.4:  # 40% chance of cart
                cart_indices = random.sample(range(num_events), min(random.randint(1, 3), num_events))
                for idx in cart_indices:
                    session_events[idx] = 'cart'
                
                # Add purchase after cart with 30% probability
                if random.random() < 0.3:
                    # Find last cart event
                    if 'cart' in session_events:
                        last_cart_idx = max([i for i, e in enumerate(session_events) if e == 'cart'])
                        if last_cart_idx < num_events - 1:
                            session_events[random.randint(last_cart_idx + 1, num_events - 1)] = 'purchase'
            
            # Create timestamps for this session with increasing order
            base_time = generate_timestamp()
            timestamps = []
            for i in range(num_events):
                # Add 1-5 minutes between events
                if i == 0:
                    timestamps.append(base_time)
                else:
                    timestamps.append(timestamps[i-1] + timedelta(minutes=random.randint(1, 5)))
            
            # Generate product IDs - keep same product for consecutive events with increasing probability
            prev_product = None
            products = []
            for i in range(num_events):
                if prev_product and random.random() < 0.7:  # 70% chance to keep looking at the same product
                    products.append(prev_product)
                else:
                    new_product = random.randint(1, 100)
                    products.append(new_product)
                    prev_product = new_product
            
            # Add data for each event
            for i in range(num_events):
                product_id = products[i]
                category_id = random.randint(1, 10)
                
                data['user_id'].append(user_id)
                data['event_time'].append(timestamps[i].strftime('%Y-%m-%d %H:%M:%S'))
                data['event_type'].append(session_events[i])
                data['product_id'].append(product_id)
                data['category_id'].append(category_id)
                data['category_name'].append(categories[category_id])
                data['user_session'].append(session_id)
                
                # Price - consistent for the same product
                price = round(random.uniform(10, 1000), 2)
                data['price'].append(price)
                
                # Device & browser - consistent for user in a session
                data['device'].append(user_device)
                data['browser'].append(user_browser)
                data['location'].append(user_loc)
                
                # Time on page - depends on event type
                if session_events[i] == 'view':
                    time_on_page = random.randint(5, 120)  # 5-120 seconds for view
                elif session_events[i] == 'cart':
                    time_on_page = random.randint(10, 180)  # 10-180 seconds for cart
                else:  # purchase
                    time_on_page = random.randint(60, 300)  # 60-300 seconds for purchase
                
                data['time_on_page'].append(time_on_page)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by user_id and event_time
    df = df.sort_values(['user_id', 'event_time'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    update_status(f"Sample dataset created with {len(df)} events and saved to {output_file}", "green")
    
    return df

def load_kaggle_dataset(sample_size=10000):
    """Load dataset from Kaggle with better error handling"""
    update_status("Loading Kaggle e-commerce dataset...", "cyan")
    update_progress(5, 100)
    
    try:
        # Try loading from local CSV first if available
        local_path = "kaggle_ecommerce_sample.csv"
        if os.path.exists(local_path):
            update_status("Loading from local cache...", "cyan")
            df = pd.read_csv(local_path)
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
            update_status(f"Data loaded with shape: {df.shape}", "green")
            update_progress(30, 100)
            return df
            
        # Check if we have a generated sample
        sample_path = "ecommerce_sample_data.csv"
        if os.path.exists(sample_path):
            update_status("Loading from generated sample file...", "cyan")
            df = pd.read_csv(sample_path)
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
            update_status(f"Data loaded with shape: {df.shape}", "green")
            update_progress(30, 100)
            return df
            
        # If local file not available, try to download via Kaggle API
        try:
            update_status("Downloading from Kaggle...", "cyan")
            
            # Try using the Kaggle API if available
            try:
                import kaggle
                
                # Make sure the directory exists
                os.makedirs('./kaggle_data', exist_ok=True)
                
                # Download the dataset
                kaggle.api.dataset_download_files(
                    'mkechinov/ecommerce-behavior-data-from-multi-category-store',
                    path='./kaggle_data',
                    unzip=True
                )
                
                # Find the smallest CSV file in the directory (to avoid huge files)
                csv_files = [f for f in os.listdir('./kaggle_data') if f.endswith('.csv')]
                
                if csv_files:
                    file_sizes = [(f, os.path.getsize(os.path.join('./kaggle_data', f))) for f in csv_files]
                    smallest_file = min(file_sizes, key=lambda x: x[1])[0]
                    
                    # Load the smallest CSV file
                    df = pd.read_csv(os.path.join('./kaggle_data', smallest_file))
                    
                    # Sample to reduce size
                    if len(df) > sample_size:
                        df = df.sample(sample_size, random_state=42)
                    
                    # Save to local cache
                    df.to_csv(local_path, index=False)
                    update_status(f"Data loaded and cached with shape: {df.shape}", "green")
                    update_progress(30, 100)
                    return df
                else:
                    raise Exception("No CSV files found in downloaded dataset")
                    
            except ImportError:
                update_status("Kaggle API not installed. Falling back to generated sample.", "yellow")
                raise Exception("Kaggle API not installed")
            
        except Exception as kaggle_error:
            update_status(f"Kaggle download error: {str(kaggle_error)}", "red")
            raise kaggle_error
            
    except Exception as e:
        update_status(f"Error loading Kaggle dataset: {str(e)}", "yellow")
        # Fall back to generating sample data
        return create_sample_ecommerce_dataset(sample_size)

def load_or_generate_data(sample_size=10000):
    """Load Kaggle dataset or generate a sample one"""
    try:
        # First try to load from Kaggle
        return load_kaggle_dataset(sample_size)
    except Exception as e:
        # If that fails, fall back to generating data
        update_status(f"Falling back to generated data: {str(e)}", "yellow")
        return create_sample_ecommerce_dataset(sample_size)

def preprocess_data(data):
    """Preprocess the dataset and engineer features"""
    update_status("Preprocessing data and engineering features...", "cyan")
    update_progress(50, 100)
    
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Column mapping for Kaggle dataset and consistency
    if 'event_time' in data.columns and 'timestamp' not in data.columns:
        data.rename(columns={'event_time': 'timestamp'}, inplace=True)
    
    if 'user_session' in data.columns and 'session_id' not in data.columns:
        data.rename(columns={'user_session': 'session_id'}, inplace=True)
    
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create target variable
    if 'event_type' in data.columns:
        event_types = data['event_type'].unique()
        
        if 'purchase' in event_types:
            data['purchase'] = data['event_type'].apply(lambda x: 1 if x == 'purchase' else 0)
        elif 'transaction' in event_types:
            data['purchase'] = data['event_type'].apply(lambda x: 1 if x == 'transaction' else 0)
        else:
            purchase_terms = ['purchase', 'transaction', 'buy', 'checkout', 'order']
            data['purchase'] = data['event_type'].apply(lambda x: 1 if any(term in str(x).lower() for term in purchase_terms) else 0)
    
    # Extract time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['month'] = data['timestamp'].dt.month
    data['day'] = data['timestamp'].dt.day
    
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    
    # Handle optional columns
    if 'browser' not in data.columns:
        data['browser'] = 'Unknown'
    
    if 'device' not in data.columns:
        data['device'] = 'Unknown'
    
    if 'location' not in data.columns:
        data['location'] = 'Unknown'
        
    if 'price' not in data.columns:
        data['price'] = 0
    
    update_progress(60, 100)
    
    if 'time_on_page' not in data.columns:
        # Estimate time_on_page
        data_sorted = data.sort_values(by=['user_id', 'session_id', 'timestamp'])
        
        data_sorted['next_timestamp'] = data_sorted.groupby(['user_id', 'session_id'])['timestamp'].shift(-1)
        data_sorted['time_on_page'] = (data_sorted['next_timestamp'] - data_sorted['timestamp']).dt.total_seconds()
        
        median_time = data_sorted['time_on_page'].median()
        data_sorted['time_on_page'].fillna(median_time, inplace=True)
        
        data_sorted.loc[data_sorted['time_on_page'] < 0, 'time_on_page'] = median_time
        data_sorted.loc[data_sorted['time_on_page'] > 3600, 'time_on_page'] = median_time
        
        data = data_sorted
    
    update_progress(65, 100)
    
    # Create session-level features
    session_features = {
        'user_id': 'first',
        'purchase': ['sum', 'count'],
        'time_on_page': ['mean', 'sum', 'max']
    }
    
    # Add price features if available
    if 'price' in data.columns:
        session_features['price'] = ['mean', 'sum']
    
    session_stats = data.groupby('session_id').agg(session_features).reset_index()
    
    # Flatten multi-level column names
    new_columns = ['session_id', 'user_id', 'purchases_in_session', 'session_events', 
                  'avg_time_on_page', 'total_time_on_site', 'max_time_on_page']
    
    if 'price' in data.columns:
        new_columns.extend(['avg_price', 'total_price'])
    
    session_stats.columns = new_columns[:len(session_stats.columns)]
    
    # Add cart event features
    cart_types = []
    if 'event_type' in data.columns:
        event_types = data['event_type'].unique()
        for et in event_types:
            if 'cart' in str(et).lower():
                cart_types.append(et)
    
    if cart_types:
        cart_events = data[data['event_type'].isin(cart_types)].groupby('session_id').size().reset_index(name='cart_adds')
    else:
        cart_events = pd.DataFrame({'session_id': session_stats['session_id'], 'cart_adds': 0})
    
    session_stats = pd.merge(session_stats, cart_events, on='session_id', how='left')
    session_stats['cart_adds'].fillna(0, inplace=True)
    
    # Add ratio features
    session_stats['cart_to_event_ratio'] = session_stats['cart_adds'] / session_stats['session_events']
    session_stats['cart_to_event_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    session_stats['cart_to_event_ratio'].fillna(0, inplace=True)
    
    # Add purchase rate
    if 'price' in data.columns:
        session_stats['price_per_event'] = session_stats['total_price'] / session_stats['session_events']
        session_stats['price_per_event'].replace([np.inf, -np.inf], 0, inplace=True)
        session_stats['price_per_event'].fillna(0, inplace=True)
    
    update_progress(75, 100)
    
    # Get user history
    user_aggs = {
        'purchases_in_session': ['mean', 'sum'],
        'session_events': ['mean', 'sum'],
        'cart_adds': ['mean', 'sum'],
        'cart_to_event_ratio': 'mean'
    }
    
    if 'price' in data.columns:
        user_aggs['total_price'] = ['mean', 'sum']
    
    user_history = session_stats.groupby('user_id').agg(user_aggs).reset_index()
    
    # Flatten multi-level columns
    user_columns = ['user_id', 'avg_purchase_rate', 'total_purchases', 
                   'avg_events_per_session', 'total_events',
                   'avg_cart_adds', 'total_cart_adds', 'avg_cart_ratio']
    
    if 'price' in data.columns:
        user_columns.extend(['avg_session_spend', 'total_spend'])
    
    user_history.columns = user_columns[:len(user_history.columns)]
    
    enriched_sessions = pd.merge(session_stats, user_history, on='user_id', how='left')
    
    # Mark if user made a purchase in the session
    enriched_sessions['made_purchase'] = (enriched_sessions['purchases_in_session'] > 0).astype(int)
    
    # Add modal device, browser, location
    modal_columns = ['device', 'browser', 'location']
    modal_features = pd.DataFrame({'session_id': session_stats['session_id']})
    
    for col in modal_columns:
        if col in data.columns:
            temp = data.groupby('session_id')[col].apply(
                lambda x: x.mode()[0] if not x.mode().empty else "unknown"
            ).reset_index()
            modal_features = pd.merge(modal_features, temp, on='session_id', how='left')
        else:
            modal_features[col] = "unknown"
    
    enriched_sessions = pd.merge(enriched_sessions, modal_features, on='session_id', how='left')
    
    # Add time features
    time_features = data.groupby('session_id').agg({
        'hour': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'month': 'first'
    }).reset_index()
    
    enriched_sessions = pd.merge(enriched_sessions, time_features, on='session_id', how='left')
    
    # Calculate conversion rates
    enriched_sessions['conversion_rate'] = enriched_sessions['purchases_in_session'] / enriched_sessions['cart_adds']
    enriched_sessions['conversion_rate'].replace([np.inf, -np.inf], 0, inplace=True)
    enriched_sessions['conversion_rate'].fillna(0, inplace=True)
    
    # Fill missing values
    enriched_sessions.fillna(0, inplace=True)
    
    update_status(f"Preprocessing complete! Final dataset shape: {enriched_sessions.shape}", "green")
    update_progress(80, 100)
    
    return enriched_sessions

def prepare_model_data(data):
    """Prepare data for modeling"""
    update_status("Preparing data for modeling...", "cyan")
    
    # Define features and target
    target = 'made_purchase'
    
    # Exclude target and identifiers from features
    exclude_cols = [target, 'session_id', 'user_id', 'purchases_in_session']
    features = [col for col in data.columns if col not in exclude_cols]
    
    # Split the data
    X = data[features]
    y = data[target]
    
    # Identify categorical and numeric features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Standardize numerical features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance using SMOTE
    update_status("Applying SMOTE to handle class imbalance...", "cyan")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    update_status(f"Data prepared for modeling. Training set: {X_train_resampled.shape}", "green")
    update_progress(85, 100)
    
    return X_train_resampled, X_test, y_train_resampled, y_test, features, scaler, numeric_features, label_encoders

def train_and_evaluate_models(X_train, X_test, y_train, y_test, features):
    """Train multiple models and evaluate their performance"""
    update_status("Training and evaluating models...", "cyan")
    update_progress(85, 100)
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    # Train each model and collect results
    results = {}
    best_f1 = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        update_status(f"Training {name} model...", "yellow")
        
        # Simulate training time
        for i in range(5):
            time.sleep(0.1)  # Shorter sleep time for web demo
            update_progress(85 + (i * 0.5), 100)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        update_status(f"{name} Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}", "green")
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
    
    update_status(f"Best model based on F1 score: {best_model_name} (F1 = {best_f1:.4f})", "green")
    update_progress(90, 100)
    
    # Get feature importances for the best model
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        feature_importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
    else:
        importance_df = pd.DataFrame()
    
    return best_model, results, best_model_name, importance_df

def generate_user_flow_chart(data):
    """Generate user flow chart based on event transitions - fixed version"""
    try:
        # Check if we have event_type column
        if 'event_type' not in data.columns:
            return "Event type data not available"
        
        # Ensure we have timestamp column (the error was here)
        timestamp_col = 'timestamp' if 'timestamp' in data.columns else 'event_time'
        
        if timestamp_col not in data.columns:
            return "Timestamp data not available"
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by user_id and timestamp
        df = df.sort_values(['user_id', timestamp_col])
        
        # Get the next event for each user
        df['next_event'] = df.groupby('user_id')['event_type'].shift(-1)
        
        # Remove rows with no next event (end of user session)
        flow_df = df.dropna(subset=['next_event'])
        
        # Get top events
        top_events = flow_df['event_type'].value_counts().head(5).index.tolist()
        
        # Filter for just the top events
        flow_df = flow_df[flow_df['event_type'].isin(top_events) & 
                          flow_df['next_event'].isin(top_events)]
        
        # Count transitions
        transitions = flow_df.groupby(['event_type', 'next_event']).size().reset_index()
        transitions.columns = ['source', 'target', 'value']
        
        return transitions
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error creating user flow chart: {str(e)}"

def generate_model_plots(best_model_name, results, importance_df, X_test, y_test):
    """Generate plots for model performance"""
    plot_data = {}
    
    # Get best model results
    best_result = results[best_model_name]
    y_pred = best_result['y_pred']
    y_pred_proba = best_result['y_pred_proba']
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Save to a BytesIO object
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    plot_data['confusion_matrix'] = base64.b64encode(img_data.getvalue()).decode('ascii')
    plt.close(fig)
    
    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    ax.plot(recall, precision, marker='.')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)
    
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    plot_data['precision_recall'] = base64.b64encode(img_data.getvalue()).decode('ascii')
    plt.close(fig)
    
    # 3. Feature Importance
    if not importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(10)
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
        ax.set_title('Top 10 Feature Importance')
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png')
        img_data.seek(0)
        plot_data['feature_importance'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
    
    # 4. Model Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    f1_scores = [results[name]['f1_score'] for name in model_names]
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy')
    ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    plot_data['model_comparison'] = base64.b64encode(img_data.getvalue()).decode('ascii')
    plt.close(fig)
    
    # 5. Combined plot
    fig = plt.figure(figsize=(15, 12))
    
    # Confusion Matrix
    ax1 = fig.add_subplot(221)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion Matrix - {best_model_name}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Precision-Recall Curve
    ax2 = fig.add_subplot(222)
    ax2.plot(recall, precision, marker='.')
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(True)
    
    # Feature Importance
    ax3 = fig.add_subplot(223)
    if not importance_df.empty:
        top_features = importance_df.head(10)
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax3)
    ax3.set_title('Top 10 Feature Importance')
    
    # Model Comparison
    ax4 = fig.add_subplot(224)
    ax4.bar(x - width/2, accuracies, width, label='Accuracy')
    ax4.bar(x + width/2, f1_scores, width, label='F1 Score')
    ax4.set_title('Model Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    
    plt.tight_layout()
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    plot_data['combined'] = base64.b64encode(img_data.getvalue()).decode('ascii')
    plt.close(fig)
    
    return plot_data

def generate_user_analytics(data):
    """Generate user analytics with purchase behavior insights"""
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure event_time is datetime
        if 'event_time' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['event_time'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            return {"error": "No timestamp column found"}
        
        # User-level analytics
        user_analytics = {}
        
        # 1. User purchase frequency
        if 'event_type' in df.columns:
            # Count purchases per user
            purchase_counts = df[df['event_type'] == 'purchase'].groupby('user_id').size()
            total_users = df['user_id'].nunique()
            
            # Users who made a purchase
            purchasing_users = purchase_counts.index.nunique()
            
            user_analytics['purchase_metrics'] = {
                'total_users': int(total_users),
                'purchasing_users': int(purchasing_users),
                'conversion_rate': round(purchasing_users / total_users * 100, 2) if total_users > 0 else 0,
                'avg_purchases_per_buyer': round(purchase_counts.mean(), 2) if not purchase_counts.empty else 0
            }
            
            # Purchase distribution
            purchase_dist = purchase_counts.value_counts().sort_index().to_dict()
            user_analytics['purchase_distribution'] = {
                'purchases_per_user': {str(k): int(v) for k, v in purchase_dist.items()}
            }
        
        # 2. Time-based analytics
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Hour analysis
            hour_activity = df.groupby('hour').size()
            hour_purchases = df[df['event_type'] == 'purchase'].groupby('hour').size()
            
            user_analytics['time_analytics'] = {
                'hourly_activity': {str(k): int(v) for k, v in hour_activity.to_dict().items()},
                'hourly_purchases': {str(k): int(v) for k, v in hour_purchases.to_dict().items() if k in hour_purchases},
                'conversion_by_hour': {
                    str(h): round(hour_purchases.get(h, 0) / act * 100, 2) 
                    for h, act in hour_activity.to_dict().items() if act > 0
                }
            }
            
            # Weekend vs weekday
            weekend_data = df.groupby('weekend').agg({
                'user_id': 'nunique',
                'timestamp': 'count'
            })
            
            # Calculate purchase counts for weekend vs weekday
            purchase_by_weekend = df[df['event_type'] == 'purchase'].groupby('weekend').size()
            
            weekend_analytics = {}
            for i in [0, 1]:  # 0=weekday, 1=weekend
                if i in weekend_data.index:
                    users = weekend_data.loc[i, 'user_id']
                    events = weekend_data.loc[i, 'timestamp']
                    purchases = purchase_by_weekend.get(i, 0)
                    
                    weekend_analytics[str(i)] = {
                        'users': int(users),
                        'events': int(events),
                        'purchases': int(purchases),
                        'conversion_rate': round(purchases / events * 100, 2) if events > 0 else 0
                    }
            
            user_analytics['weekend_vs_weekday'] = weekend_analytics
        
        # 3. Device analytics
        if 'device' in df.columns:
            device_counts = df.groupby('device').size()
            device_purchases = df[df['event_type'] == 'purchase'].groupby('device').size()
            
            user_analytics['device_analytics'] = {
                'device_counts': {str(k): int(v) for k, v in device_counts.to_dict().items()},
                'device_purchases': {str(k): int(v) for k, v in device_purchases.to_dict().items() if k in device_purchases},
                'conversion_by_device': {
                    str(d): round(device_purchases.get(d, 0) / count * 100, 2)
                    for d, count in device_counts.to_dict().items() if count > 0
                }
            }
        
        # 4. Product & Category analytics
        if 'product_id' in df.columns and 'category_id' in df.columns:
            # Top viewed products
            product_views = df[df['event_type'] == 'view'].groupby('product_id').size()
            top_viewed = product_views.sort_values(ascending=False).head(10).to_dict()
            
            # Top purchased products
            product_purchases = df[df['event_type'] == 'purchase'].groupby('product_id').size()
            top_purchased = product_purchases.sort_values(ascending=False).head(10).to_dict()
            
            # Conversion by category
            if 'category_name' in df.columns:
                category_col = 'category_name'
            else:
                category_col = 'category_id'
                
            category_views = df[df['event_type'] == 'view'].groupby(category_col).size()
            category_purchases = df[df['event_type'] == 'purchase'].groupby(category_col).size()
            
            category_conversion = {
                str(cat): {
                    'views': int(views),
                    'purchases': int(category_purchases.get(cat, 0)),
                    'conversion_rate': round(category_purchases.get(cat, 0) / views * 100, 2)
                }
                for cat, views in category_views.to_dict().items() if views > 0
            }
            
            user_analytics['product_analytics'] = {
                'top_viewed_products': {str(k): int(v) for k, v in top_viewed.items()},
                'top_purchased_products': {str(k): int(v) for k, v in top_purchased.items()},
                'category_conversion': category_conversion
            }
        
        # 5. User session analytics
        if 'user_session' in df.columns or 'session_id' in df.columns:
            session_col = 'session_id' if 'session_id' in df.columns else 'user_session'
            
            # Session duration
            session_durations = df.groupby([session_col]).agg({
                'timestamp': [min, max]
            })
            
            # Flatten hierarchical index
            session_durations.columns = ['_'.join(col).strip() for col in session_durations.columns.values]
            
            # Calculate duration in minutes
            session_durations['duration_minutes'] = (
                session_durations['timestamp_max'] - session_durations['timestamp_min']
            ).dt.total_seconds() / 60
            
            # Create histogram of session durations
            duration_bins = [0, 1, 5, 10, 15, 30, 60, float('inf')]
            labels = ['<1 min', '1-5 mins', '5-10 mins', '10-15 mins', '15-30 mins', '30-60 mins', '>60 mins']
            
            session_durations['duration_category'] = pd.cut(
                session_durations['duration_minutes'], 
                bins=duration_bins, 
                labels=labels
            )
            
            duration_dist = session_durations['duration_category'].value_counts().sort_index().to_dict()
            
            # Session counts
            total_sessions = len(session_durations)
            avg_session_duration = session_durations['duration_minutes'].mean()
            
            # Events per session
            events_per_session = df.groupby(session_col).size()
            avg_events = events_per_session.mean()
            
            user_analytics['session_analytics'] = {
                'total_sessions': int(total_sessions),
                'avg_session_duration_minutes': round(avg_session_duration, 2),
                'avg_events_per_session': round(avg_events, 2),
                'session_duration_distribution': {str(k): int(v) for k, v in duration_dist.items()}
            }
        
        # Generate plots
        user_analytics['plots'] = generate_user_analytics_plots(df)
        
        return user_analytics
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': f'Error generating user analytics: {str(e)}'}

def generate_user_analytics_plots(df):
    """Generate plots for user analytics dashboard"""
    plot_data = {}
    
    try:
        # 1. Device Distribution Pie Chart
        if 'device' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            device_counts = df['device'].value_counts()
            ax.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('User Device Distribution')
            
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            plot_data['device_distribution'] = base64.b64encode(img_data.getvalue()).decode('ascii')
            plt.close(fig)
        
        # 2. Hourly Activity & Purchase Pattern
        if 'timestamp' in df.columns or 'event_time' in df.columns:
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'event_time'
            df[ts_col] = pd.to_datetime(df[ts_col])
            df['hour'] = df[ts_col].dt.hour
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # All activity by hour
            hourly_counts = df.groupby('hour').size()
            ax.plot(hourly_counts.index, hourly_counts.values, 'b-', linewidth=2, label='All Events')
            
            # Purchase activity by hour
            if 'event_type' in df.columns:
                purchase_counts = df[df['event_type'] == 'purchase'].groupby('hour').size()
                ax2 = ax.twinx()
                ax2.plot(purchase_counts.index, purchase_counts.values, 'r-', linewidth=2, label='Purchases')
                ax2.set_ylabel('Purchase Count', color='r')
                
                # Add legend for both y-axes
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Event Count', color='b')
            ax.set_title('Hourly Activity Pattern')
            ax.set_xticks(range(0, 24))
            ax.grid(True, alpha=0.3)
            
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            plot_data['hourly_activity'] = base64.b64encode(img_data.getvalue()).decode('ascii')
            plt.close(fig)
        
        # 3. User Purchase Funnel
        if 'event_type' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Count events by type
            event_counts = df['event_type'].value_counts().sort_values(ascending=False)
            
            # Create funnel-like display
            y_pos = range(len(event_counts))
            ax.barh(y_pos, event_counts, align='center', color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(event_counts.index)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Count')
            ax.set_title('User Purchase Funnel')
            
            # Add conversion rates
            for i, (event, count) in enumerate(event_counts.items()):
                if i < len(event_counts) - 1:
                    next_count = event_counts.iloc[i+1]
                    conversion = next_count / count * 100 if count > 0 else 0
                    ax.text(count + 5, i, f"{conversion:.1f}% →", va='center')
            
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            plot_data['purchase_funnel'] = base64.b64encode(img_data.getvalue()).decode('ascii')
            plt.close(fig)
            
        # 4. User Retention Chart (simplified cohort)
        if 'user_id' in df.columns and ('timestamp' in df.columns or 'event_time' in df.columns):
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'event_time'
            df[ts_col] = pd.to_datetime(df[ts_col])
            
            # Get first activity date for each user
            user_first_activity = df.groupby('user_id')[ts_col].min().reset_index()
            user_first_activity['cohort'] = user_first_activity[ts_col].dt.to_period('W')
            
            # Get all user activity with cohort
            df_with_cohort = pd.merge(df, user_first_activity[['user_id', 'cohort']], on='user_id')
            
            # Extract week and calculate user retention
            df_with_cohort['activity_week'] = df_with_cohort[ts_col].dt.to_period('W')
            df_with_cohort['weeks_since_join'] = (df_with_cohort['activity_week'] - df_with_cohort['cohort']).apply(lambda x: x.n)
            
            # Only take cohorts with full 4 weeks of data
            max_week = df_with_cohort['activity_week'].max()
            valid_cohorts = df_with_cohort[df_with_cohort['cohort'] <= (max_week - 3)]
            
            # Only use up to 5 cohorts for clarity
            top_cohorts = valid_cohorts['cohort'].value_counts().sort_index().tail(5).index
            valid_cohorts = valid_cohorts[valid_cohorts['cohort'].isin(top_cohorts)]
            
            # Count users per cohort and week
            cohort_data = valid_cohorts.groupby(['cohort', 'weeks_since_join'])['user_id'].nunique().reset_index()
            
            # Pivot and calculate retention
            cohort_pivot = cohort_data.pivot_table(index='cohort', columns='weeks_since_join', values='user_id')
            cohort_sizes = cohort_pivot[0]
            retention_pivot = cohort_pivot.div(cohort_sizes, axis=0) * 100
            
            # Plot retention heatmap
            if not retention_pivot.empty and len(retention_pivot.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(retention_pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
                ax.set_title('Cohort Retention Analysis (% of Users Returning)')
                ax.set_ylabel('User Cohort')
                ax.set_xlabel('Weeks Since First Visit')
                
                img_data = io.BytesIO()
                fig.savefig(img_data, format='png')
                img_data.seek(0)
                plot_data['user_retention'] = base64.b64encode(img_data.getvalue()).decode('ascii')
                plt.close(fig)
        
        # 5. Category Performance
        if 'category_id' in df.columns and 'event_type' in df.columns:
            category_col = 'category_name' if 'category_name' in df.columns else 'category_id'
            
            # Count views and purchases by category
            category_views = df[df['event_type'] == 'view'].groupby(category_col).size()
            category_purchases = df[df['event_type'] == 'purchase'].groupby(category_col).size()
            
            # Combine into a dataframe
            category_data = pd.DataFrame({'views': category_views})
            category_data['purchases'] = category_purchases
            category_data.fillna(0, inplace=True)
            
            # Calculate conversion rate
            category_data['conversion_rate'] = (category_data['purchases'] / category_data['views'] * 100)
            category_data.replace([np.inf, -np.inf], 0, inplace=True)
            category_data.fillna(0, inplace=True)
            
            # Sort by purchases and get top 10
            top_categories = category_data.sort_values('purchases', ascending=False).head(10)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot bars for views and purchases
            x = np.arange(len(top_categories.index))
            width = 0.35
            
            ax.bar(x - width/2, top_categories['views'], width, label='Views', color='skyblue')
            ax.bar(x + width/2, top_categories['purchases'], width, label='Purchases', color='orange')
            
            # Add conversion rate line on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(x, top_categories['conversion_rate'], 'r-', marker='o', linewidth=2, label='Conversion Rate')
            
            # Set labels and title
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax2.set_ylabel('Conversion Rate (%)')
            ax.set_title('Top Categories by Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(top_categories.index, rotation=45, ha='right')
            
            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            plot_data['category_performance'] = base64.b64encode(img_data.getvalue()).decode('ascii')
            plt.close(fig)
            
    except Exception as e:
        plot_data['error'] = str(e)
    
    return plot_data

def save_model(model, scaler, features, numeric_features, label_encoders):
    """Save the model and preprocessing components"""
    update_status("Saving model and components...", "cyan")
    update_progress(95, 100)
    
    model_package = {
        'model': model,
        'features': features,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'label_encoders': label_encoders
    }
    
    model_path = 'ecommerce_purchase_prediction_model.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model_package, file)
    
    update_status(f"Model saved as '{model_path}'", "green")
    update_progress(98, 100)
    
    return model_path

def create_prediction_function(model, scaler, features, numeric_features, label_encoders):
    """Create a function that can make predictions on new session data"""
    
    def predict_purchase_probability(session_data):
        """Predict purchase probability for a given session"""
        try:
            # Create DataFrame from session data
            session_df = pd.DataFrame([session_data])
            
            # Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in session_df.columns:
                    # Handle unseen categories
                    try:
                        session_df[col] = encoder.transform(session_df[col].astype(str))
                    except:
                        # Assign -1 or most common category for unseen values
                        session_df[col] = -1
            
            # Ensure all features are present
            for feature in features:
                if feature not in session_df.columns:
                    session_df[feature] = 0
            
            # Scale numeric features
            if numeric_features:
                present_numeric = [f for f in numeric_features if f in session_df.columns]
                if present_numeric:
                    session_df[present_numeric] = scaler.transform(session_df[present_numeric])
            
            # Make prediction using only the features the model was trained on
            X_pred = session_df[features]
            purchase_probability = model.predict_proba(X_pred)[0, 1]
            
            return purchase_probability
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    return predict_purchase_probability

def run_pipeline(task_id=None):
    """Run the complete ML pipeline"""
    global_data['task_id'] = task_id
    global_data['task_status'][task_id] = {'status': 'running', 'progress': 0}
    
    try:
        # Load data from Kaggle (or generate if needed)
        data = load_or_generate_data(sample_size=5000)  # Reduced sample size for faster processing
        global_data['data'] = data
        global_data['task_status'][task_id]['progress'] = 30
        
        # Generate user analytics
        user_analytics = generate_user_analytics(data)
        global_data['user_analytics'] = user_analytics
        global_data['task_status'][task_id]['progress'] = 40
        
        # Preprocess data
        processed_data = preprocess_data(data)
        global_data['processed_data'] = processed_data
        global_data['task_status'][task_id]['progress'] = 50
        
        # Prepare model data
        X_train, X_test, y_train, y_test, features, scaler, numeric_features, label_encoders = prepare_model_data(processed_data)
        global_data['X_test'] = X_test
        global_data['y_test'] = y_test
        global_data['scaler'] = scaler
        global_data['numeric_features'] = numeric_features
        global_data['label_encoders'] = label_encoders
        global_data['task_status'][task_id]['progress'] = 70
        
        # Train models
        best_model, results, best_model_name, importance_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)
        global_data['model'] = best_model
        global_data['results'] = results
        global_data['best_model_name'] = best_model_name
        global_data['importance_df'] = importance_df
        global_data['task_status'][task_id]['progress'] = 90
        
        # Save model
        model_path = save_model(best_model, scaler, features, numeric_features, label_encoders)
        
        # Create prediction function
        predict_function = create_prediction_function(best_model, scaler, features, numeric_features, label_encoders)
        global_data['predict_function'] = predict_function
        
        global_data['task_status'][task_id]['status'] = 'completed'
        global_data['task_status'][task_id]['progress'] = 100
        
        return {
            'success': True,
            'message': 'Pipeline completed successfully'
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        global_data['task_status'][task_id]['status'] = 'failed'
        global_data['task_status'][task_id]['error'] = str(e)
        
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', user_login=user_login, timestamp=session_timestamp)

@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    task_id = str(uuid.uuid4())
    
    # Run pipeline in a separate thread to avoid blocking
    thread = threading.Thread(target=run_pipeline, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/check_status/<task_id>')
def check_status(task_id):
    if task_id not in global_data['task_status']:
        return jsonify({'status': 'unknown', 'progress': 0})
    
    return jsonify(global_data['task_status'][task_id])

@app.route('/get_plots')
def get_plots():
    if global_data['model'] is None or global_data['results'] is None:
        return jsonify({'error': 'Model not trained yet'})
    
    plot_data = generate_model_plots(
        global_data['best_model_name'], 
        global_data['results'], 
        global_data['importance_df'], 
        global_data['X_test'], 
        global_data['y_test']
    )
    
    return jsonify(plot_data)

@app.route('/get_results')
def get_results():
    if global_data['results'] is None:
        return jsonify({'error': 'Model not trained yet'})
    
    # Format results for display
    results_data = {}
    for name, result in global_data['results'].items():
        results_data[name] = {
            'accuracy': float(result['accuracy']),
            'f1_score': float(result['f1_score']),
            'roc_auc': float(result['roc_auc'])
        }
    
    # Get feature importance
    feature_importance = []
    if global_data['importance_df'] is not None and not global_data['importance_df'].empty:
        top_features = global_data['importance_df'].head(10)
        for _, row in top_features.iterrows():
            feature_importance.append({
                'feature': row['Feature'],
                'importance': float(row['Importance'])
            })
    
    return jsonify({
        'best_model': global_data['best_model_name'],
        'results': results_data,
        'feature_importance': feature_importance
    })

@app.route('/predict', methods=['POST'])
def predict():
    if global_data['predict_function'] is None:
        return jsonify({'error': 'Model not trained yet'})
    
    try:
        # Get data from form
        session_data = request.get_json()
        
        # Convert numeric fields
        for key in ['session_events', 'avg_time_on_page', 'total_time_on_site', 
                   'cart_adds', 'cart_to_event_ratio']:
            if key in session_data:
                try:
                    session_data[key] = float(session_data[key])
                except:
                    session_data[key] = 0.0
        
        # Add current time features
        now = datetime.now()
        session_data['hour'] = now.hour
        session_data['day_of_week'] = now.weekday()
        session_data['is_weekend'] = 1 if session_data['day_of_week'] >= 5 else 0
        session_data['month'] = now.month
        
        # Make prediction
        probability = global_data['predict_function'](session_data)
        
        # Determine recommendation
        if probability >= 0.7:
            recommendation = "High purchase intent detected! Target with special offers to complete purchase."
        elif probability >= 0.4:
            recommendation = "Moderate purchase intent. Consider personalized product recommendations."
        else:
            recommendation = "Low purchase intent. Focus on engagement and brand awareness."
        
        return jsonify({
            'probability': float(probability),
            'probability_percent': f"{float(probability):.2%}",
            'recommendation': recommendation
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/get_advanced_plots')
def get_advanced_plots():
    if global_data['model'] is None or global_data['results'] is None:
        return jsonify({'error': 'Model not trained yet'})
    
    try:
        # Get necessary data
        best_model_name = global_data['best_model_name']
        best_model = global_data['model']
        results = global_data['results']
        X_test = global_data['X_test']
        y_test = global_data['y_test']
        
        # Original data for time/device distributions
        original_data = global_data['data']
        processed_data = global_data['processed_data']
        
        plot_data = {}
        
        # 1. ROC Curve for all models
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, result in results.items():
            y_pred_proba = result['y_pred_proba']
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for All Models')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['roc_curve'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 2. Learning Curve (using best model)
        from sklearn.model_selection import learning_curve
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a subsample for learning curve to speed up computation
        X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        y_sample = y_test.loc[X_sample.index]
        
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_sample, y_sample, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 5), 
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        ax.set_xlabel('Training examples')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Learning Curve - {best_model_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['learning_curve'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 3. Purchase Distribution by Time (Hour)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by hour and calculate purchase rate
        if 'hour' in processed_data.columns and 'made_purchase' in processed_data.columns:
            hour_group = processed_data.groupby('hour')['made_purchase'].agg(['mean', 'count'])
            hour_group.columns = ['purchase_rate', 'session_count']
            
            # Plot purchase rate
            ax.bar(hour_group.index, hour_group['purchase_rate'], color='skyblue')
            ax.set_xlabel('Hour of Day (24-hour format)')
            ax.set_ylabel('Purchase Rate')
            ax.set_title('Purchase Rate by Hour of Day')
            ax.set_xticks(range(0, 24))
            ax.grid(axis='y', alpha=0.3)
            
            # Add session count as a line on secondary axis
            ax2 = ax.twinx()
            ax2.plot(hour_group.index, hour_group['session_count'], color='orange', marker='o')
            ax2.set_ylabel('Session Count', color='orange')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, ['Purchase Rate', 'Session Count'], loc='upper right')
        else:
            ax.text(0.5, 0.5, "Hour data not available", ha='center', va='center', fontsize=14)
            
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['purchase_time_dist'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 4. Purchase Distribution by Device
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'device' in processed_data.columns and 'made_purchase' in processed_data.columns:
            device_group = processed_data.groupby('device')['made_purchase'].agg(['mean', 'count'])
            device_group.columns = ['purchase_rate', 'session_count']
            device_group = device_group.sort_values('session_count', ascending=False).head(10)
            
            # Create bar positions
            x = np.arange(len(device_group.index))
            width = 0.35
            
            # Create grouped bar chart
            ax.bar(x - width/2, device_group['purchase_rate'], width, label='Purchase Rate', color='skyblue')
            
            # Add session count axis
            ax2 = ax.twinx()
            ax2.bar(x + width/2, device_group['session_count'], width, label='Session Count', color='orange', alpha=0.7)
            
            # Labels and legend
            ax.set_xlabel('Device Type')
            ax.set_ylabel('Purchase Rate')
            ax2.set_ylabel('Session Count')
            ax.set_title('Purchase Rate by Device Type')
            ax.set_xticks(x)
            ax.set_xticklabels(device_group.index, rotation=45, ha='right')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, ['Purchase Rate', 'Session Count'], loc='upper right')
        else:
            ax.text(0.5, 0.5, "Device data not available", ha='center', va='center', fontsize=14)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['purchase_device_dist'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 5. Feature Correlation Heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Select only numeric columns
        numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 2:
            corr_matrix = X_test[numeric_cols].corr()
            
            # Plot heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                        linewidths=0.5, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title('Feature Correlation Heatmap')
        else:
            ax.text(0.5, 0.5, "Not enough numeric features for correlation matrix", 
                    ha='center', va='center', fontsize=14)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['correlation_heatmap'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 6. User Flow Chart (Sankey Diagram)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Try to create a simplified user flow
        try:
            if 'event_type' in original_data.columns:
                # Use the fixed function for user flow chart
                transitions = generate_user_flow_chart(original_data)
                
                if isinstance(transitions, pd.DataFrame) and not transitions.empty:
                    # If we have data for a directed graph visualization
                    import networkx as nx
                    
                    G = nx.DiGraph()
                    # Add nodes
                    events = set(transitions['source'].unique()) | set(transitions['target'].unique())
                    for event in events:
                        G.add_node(event)
                    
                    # Add edges with weights
                    for _, row in transitions.iterrows():
                        G.add_edge(row['source'], row['target'], weight=row['value'])
                    
                    # Create positions for nodes
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Get edge weights
                    weights = [G[u][v]['weight'] for u, v in G.edges()]
                    
                    # Normalize weights for width
                    max_weight = max(weights) if weights else 1
                    edge_width = [3 * w / max_weight for w in weights]
                    
                    # Draw graph
                    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, 
                            font_size=10, font_weight='bold', edge_color='gray', 
                            width=edge_width, ax=ax)
                    
                    ax.set_title('User Flow Between Events')
                else:
                    if isinstance(transitions, str):
                        ax.text(0.5, 0.5, transitions, ha='center', va='center', fontsize=14)
                    else:
                        ax.text(0.5, 0.5, "Not enough event transitions for flow chart", 
                                ha='center', va='center', fontsize=14)
            else:
                ax.text(0.5, 0.5, "Event type data not available", ha='center', va='center', fontsize=14)
        except Exception as e:
            print(f"Error creating user flow: {str(e)}")
            ax.text(0.5, 0.5, f"Could not generate user flow chart: {str(e)}", ha='center', va='center', fontsize=14)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['user_flow'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 7. Probability Distribution (best model)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        best_result = results[best_model_name]
        y_pred_proba = best_result['y_pred_proba']
        
        # Separate probabilities for actual positive and negative classes
        proba_pos = y_pred_proba[y_test == 1]
        proba_neg = y_pred_proba[y_test == 0]
        
        # Plot histograms
        ax.hist(proba_pos, bins=20, alpha=0.7, label='Actual Purchasers', color='green')
        ax.hist(proba_neg, bins=20, alpha=0.7, label='Non-Purchasers', color='red')
        
        ax.set_xlabel('Predicted Probability of Purchase')
        ax.set_ylabel('Count')
        ax.set_title(f'Probability Distribution - {best_model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['prob_distribution'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        # 8. Cumulative Gains Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort indices by predicted probability
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_test_sorted = y_test.iloc[sorted_indices]
        
        # Calculate cumulative gains
        total_positive = np.sum(y_test)
        cum_percent_of_total = np.cumsum(y_test_sorted) / total_positive * 100
        percent_of_population = np.arange(1, len(y_test) + 1) / len(y_test) * 100
        
        # Plot
        ax.plot(percent_of_population, cum_percent_of_total, label='Model')
        ax.plot([0, 100], [0, 100], 'k--', label='Baseline')
        
        ax.set_xlabel('Percentage of Population')
        ax.set_ylabel('Percentage of Positive Cases')
        ax.set_title('Cumulative Gains Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plot_data['cumulative_gains'] = base64.b64encode(img_data.getvalue()).decode('ascii')
        plt.close(fig)
        
        return jsonify(plot_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating plots: {str(e)}'})

@app.route('/get_random_session')
def get_random_session():
    if global_data['X_test'] is None:
        return jsonify({'error': 'No test data available'})
    
    # Get a random row from X_test
    random_index = np.random.randint(0, len(global_data['X_test']))
    random_sample = global_data['X_test'].iloc[random_index].to_dict()
    
    # Convert to proper types for display
    session_data = {
        'session_events': int(random_sample.get('session_events', 0)),
        'avg_time_on_page': round(float(random_sample.get('avg_time_on_page', 0)), 2),
        'total_time_on_site': round(float(random_sample.get('total_time_on_site', 0)), 2),
        'max_time_on_page': round(float(random_sample.get('max_time_on_page', 0)), 2),
        'cart_adds': int(random_sample.get('cart_adds', 0)),
        'cart_to_event_ratio': round(float(random_sample.get('cart_to_event_ratio', 0)), 2),
        'device': 'mobile',  # Default since we've encoded these
        'browser': 'Chrome'  # Default since we've encoded these
    }
    
    # Try to decode device and browser if we have the encoders
    if global_data['label_encoders']:
        if 'device' in global_data['label_encoders'] and 'device' in random_sample:
            try:
                device_value = global_data['label_encoders']['device'].inverse_transform([int(random_sample['device'])])[0]
                session_data['device'] = device_value
            except:
                pass
                
        if 'browser' in global_data['label_encoders'] and 'browser' in random_sample:
            try:
                browser_value = global_data['label_encoders']['browser'].inverse_transform([int(random_sample['browser'])])[0]
                session_data['browser'] = browser_value
            except:
                pass
    
    return jsonify(session_data)

@app.route('/get_user_analytics')
def get_user_analytics():
    """Return user analytics data for display"""
    if global_data['user_analytics'] is None:
        if global_data['data'] is not None:
            # Generate user analytics if data is available but analytics not yet computed
            user_analytics = generate_user_analytics(global_data['data'])
            global_data['user_analytics'] = user_analytics
            return jsonify(user_analytics)
        else:
            return jsonify({'error': 'No data available for user analytics'})
    
    return jsonify(global_data['user_analytics'])

@app.route('/download_sample_dataset')
def download_sample_dataset():
    """Download generated sample dataset"""
    if os.path.exists('ecommerce_sample_data.csv'):
        return send_file('ecommerce_sample_data.csv', 
                         download_name='ecommerce_sample_data.csv',
                         as_attachment=True)
    
    # If file doesn't exist, generate it first
    data = create_sample_ecommerce_dataset(num_records=5000)
    return send_file('ecommerce_sample_data.csv', 
                     download_name='ecommerce_sample_data.csv',
                     as_attachment=True)

@app.route('/simulate_session', methods=['POST'])
def simulate_session():
    """Simulate a user session with multiple events and predict purchase"""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        num_events = int(params.get('num_events', 5))
        include_purchase = params.get('include_purchase', False)
        
        # Generate a simulated session
        user_id = random.randint(2000, 3000)
        session_id = random.randint(50000, 60000)
        device = random.choice(['mobile', 'desktop', 'tablet'])
        browser = random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'])
        location = random.choice(['USA', 'UK', 'Germany', 'India', 'Canada'])
        
        # Start time (now)
        start_time = datetime.now()
        
        # Generate events
        events = []
        event_types = ['view'] * (num_events - 1)  # Start with all views
        
        # Add cart event with 40% probability
        if random.random() < 0.4:
            cart_idx = random.randint(1, num_events - 2)
            event_types[cart_idx] = 'cart'
            
            # Add purchase after cart with 30% probability or if explicitly requested
            if include_purchase or random.random() < 0.3:
                event_types.append('purchase')
            else:
                event_types.append('view')
        else:
            event_types.append('view')
        
        # Generate random products
        product_id = random.randint(1, 100)
        category_id = random.randint(1, 10)
        
        # Generate events with timestamps
        total_time_on_site = 0
        for i, event_type in enumerate(event_types):
            # Time difference between events (1-5 minutes)
            if i > 0:
                time_diff = random.randint(1, 5) * 60  # seconds
            else:
                time_diff = 0
                
            # Time on page depends on event type
            if event_type == 'view':
                time_on_page = random.randint(5, 120)
            elif event_type == 'cart':
                time_on_page = random.randint(30, 180)
            else:  # purchase
                time_on_page = random.randint(60, 300)
            
            timestamp = start_time + timedelta(seconds=total_time_on_site)
            
            events.append({
                'event_number': i + 1,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event_type,
                'user_id': user_id,
                'session_id': session_id,
                'product_id': product_id,
                'category_id': category_id,
                'device': device,
                'browser': browser,
                'location': location,
                'time_on_page': time_on_page
            })
            
            total_time_on_site += time_diff + time_on_page
        
        # Calculate session metrics
        session_events = len(events)
        avg_time_on_page = sum(e['time_on_page'] for e in events) / len(events)
        total_time_on_site = sum(e['time_on_page'] for e in events)
        cart_adds = sum(1 for e in events if e['event_type'] == 'cart')
        cart_to_event_ratio = cart_adds / session_events if session_events > 0 else 0
        
        # Create session data for prediction
        session_data = {
            'session_events': session_events,
            'avg_time_on_page': avg_time_on_page,
            'total_time_on_site': total_time_on_site,
            'cart_adds': cart_adds,
            'cart_to_event_ratio': cart_to_event_ratio,
            'device': device,
            'browser': browser,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0
        }
        
        # Make prediction if model is available
        prediction = None
        if global_data['predict_function'] is not None:
            probability = global_data['predict_function'](session_data)
            prediction = {
                'probability': float(probability),
                'probability_percent': f"{float(probability):.2%}"
            }
        
        # Return session data and events
        return jsonify({
            'session_metrics': session_data,
            'events': events,
            'prediction': prediction,
            'has_purchase': any(e['event_type'] == 'purchase' for e in events)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/get_sample_dataset_info')
def get_sample_dataset_info():
    """Get information about the sample dataset"""
    try:
        if os.path.exists('ecommerce_sample_data.csv'):
            df = pd.read_csv('ecommerce_sample_data.csv')
            
            # Get basic stats
            info = {
                'filename': 'ecommerce_sample_data.csv',
                'rows': len(df),
                'columns': len(df.columns),
                'file_size': f"{os.path.getsize('ecommerce_sample_data.csv') / (1024*1024):.2f} MB",
                'column_names': df.columns.tolist(),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'event_types': df['event_type'].unique().tolist() if 'event_type' in df.columns else [],
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
                'unique_products': df['product_id'].nunique() if 'product_id' in df.columns else 0,
                'unique_sessions': df['user_session'].nunique() if 'user_session' in df.columns else 0,
                'date_range': [
                    df['event_time'].min() if 'event_time' in df.columns else 'N/A',
                    df['event_time'].max() if 'event_time' in df.columns else 'N/A'
                ]
            }
            
            # Generate a preview (first 5 rows)
            preview = df.head(5).to_dict(orient='records')
            
            return jsonify({
                'info': info,
                'preview': preview
            })
        else:
            return jsonify({'error': 'Sample dataset not found'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': user_login,
        'python_version': platform.python_version(),
        'components': {
            'flask': flask.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': sklearn.__version__,
            'matplotlib': matplotlib.__version__
        },
        'data_loaded': global_data['data'] is not None,
        'model_trained': global_data['model'] is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    # Create a basic sample file if it doesn't exist
    if not os.path.exists('ecommerce_sample_data.csv'):
        print("Creating initial sample dataset...")
        create_sample_ecommerce_dataset(num_records=5000)
    
    app.run(debug=True)
