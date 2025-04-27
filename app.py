import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

#Load the trained ANN model
model = tf.keras.models.load_model('mental_health_ann_model.keras')

#Load the scaler
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit app
st.title("Mental Health Treatment Prediction App (ANN Model)")

st.markdown("""
This app predicts whether a person is likely to seek mental health treatment based on their personal and workplace factors.
""")

# User Inputs
st.header("Please answer the following:")

family_history = st.selectbox('Family History of Mental Health Issues?', ('Yes', 'No', 'Maybe'))
mental_health_history = st.selectbox('Past Mental Health Problems?', ('Yes', 'No', 'Maybe'))
growing_stress = st.selectbox('Feeling Growing Stress?', ('Yes', 'No', 'Maybe'))
mood_swings = st.selectbox('Mood Swings?', ('Low', 'Medium', 'High'))
coping_struggles = st.selectbox('Struggling to Cope?', ('Yes', 'No', 'Maybe'))
work_interest = st.selectbox('Decreased Interest in Work?', ('Yes', 'No', 'Maybe'))
social_weakness = st.selectbox('Social Weakness?', ('Yes', 'No', 'Maybe'))
mental_health_interview = st.selectbox('Willing to Talk to HR?', ('Yes', 'No', 'Maybe'))
care_options = st.selectbox('Are Care Options Available?', ('Yes', 'No', 'Maybe'))

#  Mapping inputs to numbers
mapping_binary = {'Yes': 1, 'No': 0, 'Maybe': 0.5}
mapping_mood = {'Low': 0, 'Medium': 0.5, 'High': 1}

inputs = np.array([
    mapping_binary[family_history],
    mapping_binary[mental_health_history],
    mapping_binary[growing_stress],
    mapping_mood[mood_swings],
    mapping_binary[coping_struggles],
    mapping_binary[work_interest],
    mapping_binary[social_weakness],
    mapping_binary[mental_health_interview],
    mapping_binary[care_options]
]).reshape(1, -1)

# Scale the inputs using the loaded scaler
inputs_scaled = scaler.transform(inputs)

#Make prediction
if st.button('Predict'):
    prediction = model.predict(inputs_scaled)
    probability = prediction[0][0]  

    st.write(f"**Prediction Probability: {probability:.2f}**")

    if probability > 0.54:
        st.success(f'✅ Likely to Seek Mental Health Treatment (Confidence: {probability*100:.1f}%)')
    elif probability < 0.45:
        st.warning(f'❌ Not Likely to Seek Mental Health Treatment (Confidence: {(1-probability)*100:.1f}%)')
    else:
        st.info('⚠️ Uncertain Prediction (around 50%) - Needs More Information')
