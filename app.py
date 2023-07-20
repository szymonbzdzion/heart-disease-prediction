import streamlit as st
import numpy as np
import pandas as pd
import pickle



st.title('Heart disease risk prediction :heart:')

with st.sidebar:
    st.markdown("""
                # Tool description:
                Machine learning model for predicting the risk of developing cardiovascular disease \n
                Gradient Boosting Classifer


                ## Variables description:
                1. Age - age of patients [years]
                2. Gender - gender of a patient [M - male, F - female]
                3. Chest pain type - [TA - typical angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
                4. Resting blood pressure - [mm Hg]
                5. Cholesterol - serum cholesterol level [mm/dl]
                6. Fasting blood sugar - [Yes: if FastBS > 120mg/dl; No: otherwise]
                7. Resting ECG - resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
                8. Max heart rate - maximum heart rate achieved [Numeric value]
                9. exercise-induced angina - [Y: Yes, N: No]
                10. oldpeak = ST [Numeric value measured in depression]
                11. ST_Slope - the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
            
                """)


with open('NewGradientBoost.pkl', 'rb') as f:
    model = pickle.load(f)

col1, col2, col3 = st.columns(3)



with col1:
    age = st.slider('Age', 0, 100)
with col2:
    gender = st.selectbox('Gender', ['M', 'F'])
with col3:
    chestpain = st.selectbox('Chest pain type:', ['TA', 'ATA', 'NAP', 'ASY'])
with col1:
    rest_bp = st.slider('Resting Blood Pressure: ', 0, 200)
with col2:
    choleterol = st.slider('Cholesterol level:', 0.0, 700.0)
with col3:
    fastingBS = st.select_slider('Fasting blood sugar', ['Yes', "No"])
with col1:
    rest_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
with col2:
    max_hr = st.slider('Max heart rate', 50, 210)
with col3:
    ex_angina = st.selectbox('Exercise induced angina', ['Y', 'N'])
with col1:
    oldpeak = st.slider('Oldpeak', -3.0, 7.0)
with col2:
    st_slope = st.selectbox('ST slope', ['Up', 'Flat', 'Down'])

columns = ["Age",	"Sex",	"ChestPainType",	"RestingBP",	"Cholesterol",	'FastingBS',"RestingECG",	"MaxHR",	"ExerciseAngina",	"Oldpeak",	"ST_Slope"]
pred_prop = []

if st.button('Predict'):
    row = np.array([age, gender, chestpain, rest_bp, choleterol, fastingBS ,rest_ecg, max_hr, ex_angina, oldpeak, st_slope])
    X = pd.DataFrame([row], columns=columns)
    X = model[0].transform(X)
    pred_prob = model[1].predict_proba(X)
    st.write(f'Probability of developing heart disease: {np.round(pred_prob[0][1], 2) * 100}%')

    

 





