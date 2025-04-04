# E-Commerce Purchase Predictor

This project is a Flask-based web application designed to predict the likelihood of a user making a purchase based on their session data. It integrates data preprocessing, machine learning model training, and visualization tools to provide actionable insights. The application features a user-friendly interface with interactive charts, analytics, and real-time predictions.

---

## Features

### 1. **Data Handling**
- **Sample Dataset Generation**: Automatically generates a realistic e-commerce dataset with user sessions, events, and product interactions if no dataset is available.
- **Kaggle Dataset Integration**: Supports loading datasets from Kaggle using the Kaggle API for enhanced data flexibility.
- **Dataset Preview**: Displays dataset statistics, column information, and a preview of the first few rows.

### 2. **Data Preprocessing**
- **Feature Engineering**: Extracts session-level and user-level features such as:
  - `time_on_page`: Average time spent on a page.
  - `cart_to_event_ratio`: Ratio of cart events to total events.
  - `purchase_rate`: Percentage of sessions resulting in a purchase.
- **Handling Missing Data**: Fills missing values using forward-fill and median imputation techniques.
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset for improved model performance.

### 3. **Machine Learning Pipeline**
- **Model Training**: Trains multiple machine learning models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Feature Importance**: Identifies the most influential features for purchase predictions.
- **Model Comparison**: Evaluates models based on metrics such as:
  - Accuracy
  - F1 Score
  - ROC AUC (Receiver Operating Characteristic - Area Under Curve)

### 4. **Visualization**
- **Interactive Charts**:
  - Confusion Matrix
  - Precision-Recall Curve
  - Feature Importance
  - User Behavior Flow (Sankey Diagram)
- **Advanced Plots**:
  - Purchase Distribution by Time (Hourly)
  - Purchase Distribution by Device
  - Feature Correlation Heatmap
- **Cohort Analysis**: Displays user retention over time using heatmaps.

### 5. **User Analytics**
- **Device and Browser Insights**: Analyzes user behavior based on device and browser usage.
- **Time-Based Insights**: Identifies peak activity and purchase hours.
- **Session Analytics**: Provides metrics such as:
  - Average session duration
  - Events per session
  - Session duration distribution

### 6. **Prediction**
- **Real-Time Prediction**: Predicts the likelihood of a purchase for a given session using trained models.
- **Recommendations**: Provides actionable recommendations based on prediction probabilities:
  - High purchase intent: Suggests targeting with special offers.
  - Moderate purchase intent: Recommends personalized product recommendations.
  - Low purchase intent: Focuses on engagement and brand awareness.

### 7. **Simulation**
- **Session Simulation**: Simulates user sessions with multiple events (e.g., views, carts, purchases) and predicts purchase probabilities.

### 8. **API Endpoints**
- `/start_pipeline`: Starts the machine learning pipeline.
- `/check_status/<task_id>`: Checks the status of the pipeline.
- `/get_results`: Retrieves model results and feature importance.
- `/predict`: Predicts purchase probability for a given session.
- `/get_user_analytics`: Returns user analytics data.
- `/get_plots`: Provides visualization data.
- `/simulate_session`: Simulates a user session and predicts purchase probability.
- `/download_sample_dataset`: Downloads the generated sample dataset.
- `/get_sample_dataset_info`: Provides information about the sample dataset.
- `/api/health`: Health check endpoint for the API.

---

## Frontend Features

### HTML
- **Dynamic Content**: Displays results, analytics, and visualizations dynamically using Flask templates.

### CSS
- **Responsive Design**: Ensures the application is accessible on various devices.
- **Custom Animations**: Includes hover effects, transitions, and loading animations.

### JavaScript
- **`script.js`**:
  - Handles dynamic updates, animations, and API calls.
  - Manages interactive charts and visualizations.
  - Provides utility functions for smooth user experience.
- **`tabfix.js`**:
  - Fixes tab navigation issues.
  - Enhances modal functionality for enlarged plot views.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Node.js (optional, for advanced frontend development)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecommerce-purchase-predictor.git
   cd ecommerce-purchase-predictor 
    ```
2. Install the requirement
   ```
   pip install -r requirements.txt
    ```
3. Run the code
    ```   
   python app.py 
4. Open the application in your browser at
    ```
   http://localhost:5000. 
 ```  
ecommerce-purchase-predictor/
├── [app.py]            # Main Flask application
├── [ecommerce_sample_data.csv] # Sample dataset (auto-generated if missing)
├── [ecommerce_purchase_prediction_model.pkl] # Trained model (auto-generated)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
├── static/
│   ├── css/
│   │   └── [style.css]      # Application styles
│   ├── js/
│   │   ├── [script.js]       # Main JavaScript logic
│   │   └── [tabfix.js]      # Tab navigation fixes
│   └── images/             # Static images
├── templates/
│   └── [index.html]         # Main HTML template
└── README.md               # Project documentation
 ```  
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
This README provides a comprehensive overview of your project, including its features, installation steps, file structure, and API endpoints. Let me know if you ned further refinements!

> License
>
>This project is licensed under the MIT License. See the LICENSE file for details.
