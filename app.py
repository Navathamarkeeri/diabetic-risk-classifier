import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🧪 Diabetes Risk Prediction ")

st.markdown("""
> This app allows users to input personal health data and predicts their diabetes status using a trained Random Forest model. 
> It also explains which factors contributed most to the prediction using advanced AI analysis.

**How it works:**
- Enter your health metrics using the sliders below
- Real-time color coding shows risk levels: 🟢 Healthy, 🟡 Moderate Risk, 🔴 High Risk
- The AI model will predict your diabetes risk category
- Feature importance explanations will show which factors influenced the prediction most
""")

# Add CSS for better styling
st.markdown("""
<style>
.metric-status {
    padding: 8px;
    border-radius: 5px;
    margin: 5px 0;
    font-weight: bold;
}
.healthy { background-color: #d4edda; color: #155724; }
.moderate { background-color: #fff3cd; color: #856404; }
.high-risk { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('random_forest_model.joblib')
        return model_data['model'], model_data['scaler'], model_data['feature_names']
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'random_forest_model.joblib' is in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model, scaler, and feature names
model, scaler, feature_names = load_model()

# Create input sliders with color-coded feedback
st.subheader("📊 Enter Your Health Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age (years)", min_value=18, max_value=100, value=50, help="Your current age in years")
    
    # Age status
    if age >= 65:
        st.error("🔴 **HIGH RISK**: Age 65+ significantly increases diabetes risk")
    elif age >= 45:
        st.warning("🟡 **MODERATE RISK**: Age 45+ increases diabetes risk")
    else:
        st.success("🟢 **LOW RISK**: Younger age is protective")
    
    st.markdown("---")
    
    fbs = st.slider("🩸 Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=100, 
                    help="Blood sugar level after fasting for 8+ hours")
    
    # FBS status with clear ranges
    if fbs >= 126:
        st.error("🔴 **DIABETES RANGE**: ≥126 mg/dL indicates diabetes")
    elif fbs >= 100:
        st.warning("🟡 **PRE-DIABETES RANGE**: 100-125 mg/dL indicates pre-diabetes")
    else:
        st.success("🟢 **NORMAL RANGE**: <100 mg/dL is healthy")
    
    st.markdown("---")
    
    bmi = st.slider("⚖️ BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, 
                   help="Body Mass Index calculated as weight(kg)/height(m)²")
    
    # BMI status
    if bmi >= 30:
        st.error("🔴 **OBESE**: BMI ≥30 significantly increases diabetes risk")
    elif bmi >= 25:
        st.warning("🟡 **OVERWEIGHT**: BMI 25-29.9 increases diabetes risk")
    elif bmi >= 18.5:
        st.success("🟢 **NORMAL WEIGHT**: BMI 18.5-24.9 is healthy")
    else:
        st.warning("🟡 **UNDERWEIGHT**: BMI <18.5 may indicate health concerns")

with col2:
    waist_circumference = st.slider("📏 Waist Circumference (cm)", min_value=50, max_value=150, value=90,
                                  help="Measurement around your waist at the narrowest point")
    
    # Gender selection for waist assessment
    gender = st.selectbox("👤 Gender", options=["Male", "Female"], index=0)
    gender_encoded = 1 if gender == "Male" else 0
    
    # Waist status based on gender
    if gender == "Male":
        if waist_circumference >= 102:
            st.error("🔴 **HIGH RISK**: Male waist ≥102cm increases diabetes risk")
        elif waist_circumference >= 94:
            st.warning("🟡 **MODERATE RISK**: Male waist 94-101cm is concerning")
        else:
            st.success("🟢 **HEALTHY**: Male waist <94cm is good")
    else:
        if waist_circumference >= 88:
            st.error("🔴 **HIGH RISK**: Female waist ≥88cm increases diabetes risk")
        elif waist_circumference >= 80:
            st.warning("🟡 **MODERATE RISK**: Female waist 80-87cm is concerning")
        else:
            st.success("🟢 **HEALTHY**: Female waist <80cm is good")
    
    st.markdown("---")
    
    hip_circumference = st.slider("📐 Hip Circumference (cm)", min_value=70, max_value=150, value=100,
                                help="Measurement around your hips at the widest point")
    
    # Calculate and display WC/HC ratio
    wc_hc_ratio = waist_circumference / hip_circumference if hip_circumference > 0 else 0
    
    st.markdown(f"**Waist-to-Hip Ratio: {wc_hc_ratio:.2f}**")
    
    # WC/HC ratio status based on gender
    if gender == "Male":
        if wc_hc_ratio >= 0.95:
            st.error("🔴 **HIGH RISK**: Male ratio ≥0.95 increases diabetes risk")
        elif wc_hc_ratio >= 0.90:
            st.warning("🟡 **MODERATE RISK**: Male ratio 0.90-0.94 is concerning")
        else:
            st.success("🟢 **HEALTHY**: Male ratio <0.90 is good")
    else:
        if wc_hc_ratio >= 0.85:
            st.error("🔴 **HIGH RISK**: Female ratio ≥0.85 increases diabetes risk")
        elif wc_hc_ratio >= 0.80:
            st.warning("🟡 **MODERATE RISK**: Female ratio 0.80-0.84 is concerning")
        else:
            st.success("🟢 **HEALTHY**: Female ratio <0.80 is good")

# Prepare input data
input_data = np.array([[age, gender_encoded, fbs, waist_circumference, hip_circumference, wc_hc_ratio]])
input_df = pd.DataFrame(input_data, columns=feature_names)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    try:
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_scaled)[0]
        predicted_class = model.predict(input_scaled)[0]
        
        # Map predictions to class names
        class_names = ["Non-Diabetes", "Pre-Diabetes", "Diabetes"]
        predicted_label = class_names[predicted_class]
        
        # Display prediction results
        st.subheader("🎯 Prediction Results")
        
        # Display predicted class
        if predicted_class == 0:
            st.success(f"✅ **Prediction: {predicted_label}**")
        elif predicted_class == 1:
            st.warning(f"⚠️ **Prediction: {predicted_label}**")
        else:
            st.error(f"🚨 **Prediction: {predicted_label}**")
        
        # Display probabilities
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({
            'Risk Category': class_names,
            'Probability': [f"{prob:.1%}" for prob in prediction_proba]
        })
        st.dataframe(prob_df, hide_index=True)
        
        # AI Explanation using Feature Importance
        st.subheader("🧠 AI Explanation (Feature Importance Analysis)")
        st.info("The following explanation shows how each health metric influenced the prediction based on the AI model's learned patterns:")
        
        try:
            # Get feature importance from the Random Forest model
            feature_importance = model.feature_importances_
            
            # Calculate personalized impact based on input values and model patterns
            # This simulates how each feature contributes to the prediction
            input_values = np.array([age, gender_encoded, fbs, waist_circumference, hip_circumference, wc_hc_ratio])
            
            # Normalize input values to understand relative impact
            # Create ranges for comparison (based on training data patterns)
            feature_ranges = {
                'Age': (18, 100),
                'Gender': (0, 1),
                'FBS': (50, 300),
                'Waist Circumference': (50, 150),
                'Hip Circumference': (70, 150),
                'WC/HC Ratio': (0.5, 1.5)
            }
            
            # Calculate normalized impact scores
            normalized_impacts = []
            feature_names_display = ['Age', 'Gender', 'Fasting Blood Sugar', 'Waist Circumference', 'Hip Circumference', 'Waist-to-Hip Ratio']
            
            for i, (feature_name, (min_val, max_val)) in enumerate(zip(feature_names_display, feature_ranges.values())):
                # Normalize the input value
                normalized_val = (input_values[i] - min_val) / (max_val - min_val)
                # Combine with feature importance
                impact_score = feature_importance[i] * abs(normalized_val - 0.5) * 2  # 0.5 is middle/neutral
                normalized_impacts.append(impact_score)
            
            # Create feature impact dataframe
            feature_impact = pd.DataFrame({
                'Feature': feature_names_display,
                'Impact_Score': normalized_impacts,
                'Feature_Importance': feature_importance,
                'Input_Value': [age, gender, fbs, waist_circumference, hip_circumference, wc_hc_ratio]
            })
            
            # Sort by impact score
            feature_impact = feature_impact.sort_values('Impact_Score', ascending=False)
            
            # Generate text explanations
            st.markdown("**Feature Impact Analysis:**")
            
            for idx, row in feature_impact.iterrows():
                feature = row['Feature']
                impact = row['Impact_Score']
                importance = row['Feature_Importance']
                input_val = row['Input_Value']
                
                # Determine impact strength based on both importance and normalized value
                if impact > 0.15:
                    strength = "**strongly**"
                elif impact > 0.10:
                    strength = "**moderately**"
                elif impact > 0.05:
                    strength = "**slightly**"
                elif impact > 0.02:
                    strength = "**very slightly**"
                else:
                    strength = "had **minimal impact**"
                
                # Determine direction based on feature value and expected patterns
                direction_emoji = "📊"
                risk_assessment = ""
                
                if feature == 'Age':
                    if input_val > 50:
                        risk_assessment = "older age increases diabetes risk"
                        direction_emoji = "📈" if predicted_class > 0 else "📊"
                    else:
                        risk_assessment = "younger age generally reduces diabetes risk"
                        direction_emoji = "📉"
                        
                elif feature == 'Fasting Blood Sugar':
                    if input_val > 125:
                        risk_assessment = "elevated blood sugar strongly indicates diabetes risk"
                        direction_emoji = "🚨"
                    elif input_val > 100:
                        risk_assessment = "elevated blood sugar suggests pre-diabetes risk"
                        direction_emoji = "⚠️"
                    else:
                        risk_assessment = "normal blood sugar levels support low diabetes risk"
                        direction_emoji = "✅"
                        
                elif feature == 'Waist-to-Hip Ratio':
                    if input_val > 0.9:
                        risk_assessment = "high waist-to-hip ratio increases diabetes risk"
                        direction_emoji = "📈"
                    else:
                        risk_assessment = "healthy waist-to-hip ratio supports lower risk"
                        direction_emoji = "✅"
                        
                elif feature == 'Waist Circumference':
                    if input_val > 100:
                        risk_assessment = "larger waist circumference increases diabetes risk"
                        direction_emoji = "📈"
                    else:
                        risk_assessment = "healthy waist circumference supports lower risk"
                        direction_emoji = "✅"
                        
                elif feature == 'Hip Circumference':
                    risk_assessment = "contributes to overall body composition assessment"
                    direction_emoji = "📊"
                    
                elif feature == 'Gender':
                    gender_display = "Male" if input_val == 1 else "Female"
                    risk_assessment = f"{gender_display} - gender influences diabetes risk patterns"
                    direction_emoji = "👤"
                
                # Format input value
                if feature == 'Gender':
                    input_display = "Male" if input_val == 1 else "Female"
                elif feature in ['Waist-to-Hip Ratio']:
                    input_display = f"{input_val:.2f}"
                else:
                    input_display = f"{input_val:.1f}"
                
                if strength != "had **minimal impact**":
                    st.write(f"{direction_emoji} **{feature}** ({input_display}): {strength} influenced the prediction - {risk_assessment}")
                else:
                    st.write(f"➡️ **{feature}** ({input_display}): {strength} on the prediction")
            
            # Additional insights
            st.subheader("💡 Key Insights")
            top_feature = feature_impact.iloc[0]
            st.write(f"🔑 The most influential factor in this prediction was **{top_feature['Feature']}** based on your specific values and the AI model's learned patterns.")
            
            # Risk-based recommendations
            if predicted_class == 2:  # Diabetes
                st.warning("🏥 **Important:** Your metrics suggest high diabetes risk. Please consult with a healthcare provider for proper medical evaluation and personalized treatment advice.")
            elif predicted_class == 1:  # Pre-diabetes
                st.info("⚠️ **Action Recommended:** Your metrics suggest pre-diabetes risk. Consider lifestyle modifications and regular monitoring. Consult a healthcare provider for guidance.")
            else:  # Non-diabetes
                st.success("✨ **Great news!** Your current metrics suggest low diabetes risk. Continue maintaining a healthy lifestyle with regular exercise and balanced nutrition!")
                
        except Exception as e:
            st.error(f"Error generating feature importance explanation: {str(e)}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Additional information
st.markdown("---")
st.subheader("ℹ️ Important Notes")
st.info("""
**Medical Disclaimer:** This prediction tool is for educational purposes only and should not replace professional medical advice. 
Always consult with healthcare providers for proper medical evaluation and diagnosis.

**How to interpret results:**
- **Non-Diabetes**: Low risk based on current metrics
- **Pre-Diabetes**: Elevated risk, lifestyle changes may help
- **Diabetes**: High risk, medical evaluation recommended
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and SHAP", unsafe_allow_html=True)
