"""
Model trainer script to create the random_forest_model.joblib file
This script creates a sample trained model based on the analysis from the notebooks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample training data based on the patterns from the notebooks"""
    np.random.seed(42)
    
    # Create sample data for 3 classes: 0=Non-Diabetes, 1=Pre-Diabetes, 2=Diabetes
    n_samples = 1000
    
    # Non-Diabetes (healthy individuals)
    non_diabetes = pd.DataFrame({
        'Age': np.random.normal(35, 10, 300).clip(18, 80),
        'Gender': np.random.choice([0, 1], 300),  # 0=Female, 1=Male
        'FBS': np.random.normal(85, 10, 300).clip(70, 100),  # Normal FBS < 100
        'wc': np.random.normal(80, 8, 300).clip(60, 120),
        'Hc': np.random.normal(95, 8, 300).clip(75, 130),
        'label': 0
    })
    
    # Pre-Diabetes 
    pre_diabetes = pd.DataFrame({
        'Age': np.random.normal(45, 12, 350).clip(25, 75),
        'Gender': np.random.choice([0, 1], 350),
        'FBS': np.random.normal(110, 8, 350).clip(100, 125),  # Pre-diabetes FBS 100-125
        'wc': np.random.normal(90, 10, 350).clip(70, 130),
        'Hc': np.random.normal(98, 8, 350).clip(80, 130),
        'label': 1
    })
    
    # Diabetes
    diabetes = pd.DataFrame({
        'Age': np.random.normal(55, 15, 350).clip(30, 85),
        'Gender': np.random.choice([0, 1], 350),
        'FBS': np.random.normal(160, 40, 350).clip(126, 300),  # Diabetes FBS >= 126
        'wc': np.random.normal(100, 12, 350).clip(80, 150),
        'Hc': np.random.normal(100, 10, 350).clip(85, 140),
        'label': 2
    })
    
    # Combine all data
    data = pd.concat([non_diabetes, pre_diabetes, diabetes], ignore_index=True)
    
    # Calculate WC/HC ratio
    data['WC/HC'] = data['wc'] / data['Hc']
    
    return data

def train_model():
    """Train the Random Forest model"""
    print("Creating sample training data...")
    df = create_sample_data()
    
    print("Preparing data for training...")
    # Select features
    feature_names = ['Age', 'Gender', 'FBS', 'wc', 'Hc', 'WC/HC']
    X = df[feature_names]
    y = df['label']
    
    print("Class distribution:")
    print(y.value_counts().sort_index())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print("Training Random Forest model...")
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetes', 'Pre-Diabetes', 'Diabetes']))
    
    # Save model with metadata
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'class_names': ['Non-Diabetes', 'Pre-Diabetes', 'Diabetes'],
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, 'random_forest_model.joblib')
    print("\nModel saved as 'random_forest_model.joblib'")
    print("Model training completed successfully!")

if __name__ == "__main__":
    train_model()
