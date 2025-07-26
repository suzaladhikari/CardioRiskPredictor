import streamlit as st
import pandas as pd
import joblib
import os 
## Lets divide the app into different headers first and then work out on each screen 

st.sidebar.title("Explore the options")
page = st.sidebar.selectbox("",["🏠 Home", " 📊 Prediction", " ℹ️ About the Model"," 👨‍💻 Developer"])

# Home Page 
if page == '🏠 Home':
    st.title("Cardio Vascular Disease Prediction")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("""Cardiovascular disease is one of the leading causes of death worldwide. What makes it especially dangerous is how suddenly it can occur, often without obvious warning signs.However, many preventive measures—ranging from physical activity to stress management—can significantly reduce the risk.This app helps you assess your likelihood of developing cardiovascular issues based on key health indicators.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("To explore and predict, please go to 📊 Prediction in the Navigation at the top right, of having CardioVascular Disease in the near future!")
    st.warning("⚠️ This is a demo. Do not rely on this prediction for medical advice.")


# Prediction Page

if page == ' 📊 Prediction':
    st.header("Predict the chance of having Heart Disease based on various factors. ")
    st.subheader("Just use the sidebar sliders and checkboxes to adjust the values, and the model will do the rest!")
    st.text("")
    st.markdown("---")
    st.subheader('🧍 Personal Info')
    gender = st.selectbox("What is your gender",['Male','Female'])
    gender = 1 if gender == 'Male' else 0
    expected_age_columns = [
    'Age_Category_18-24', 'Age_Category_25-29', 'Age_Category_30-34',
    'Age_Category_35-39', 'Age_Category_40-44', 'Age_Category_45-49',
    'Age_Category_50-54', 'Age_Category_55-59', 'Age_Category_60-64',
    'Age_Category_65-69', 'Age_Category_70-74', 'Age_Category_75-79',
    'Age_Category_80+'
    ]
    age = st.selectbox("Which Age range do you fall into?!", [
        "18-24","25-29","30-34","35-39","40-44","45-49","50-54",
        "55-59","60-64","65-69","70-74","75-79","80+"
    ])

    height_input = st.text_input("Can you please enter your height (in cm)")

    try:
        height_cm = int(height_input)
    except ValueError:
        height_cm = None
        st.warning("Please enter a valid weight in pounds.")
    weight_input = st.text_input("Can you please input your weight (in pounds)")

    try:
        weight_lbs = int(weight_input)
    except ValueError:
        weight_lbs = None
        st.warning("Please enter a valid weight in pounds.")


    # Calculate BMI if inputs are given
    bmi = None
    weight_kg = None
    height_m = None
    if height_cm and weight_lbs:
        try:
            height_m = float(height_cm) / 100
            weight_kg = float(weight_lbs) * 0.453592
            bmi = weight_kg / (height_m ** 2)
            st.success(f"Your BMI is: {bmi:.2f}")
        except ValueError:
            st.error("Please enter valid numbers for height and weight.")

    st.text("")
    st.markdown("---")
    st.subheader('🩺 Health History')

    general_health = st.selectbox("How is your general health",["Excellent", "Very good", "Good", "Fair", "Poor"])
    exercise_response = st.selectbox("Have you exercised in past 30 days",['Yes','No'])
    exercise = 1 if exercise_response == 'Yes' else 0
    st.text("")
    st.write("Do you have any of these conditions?")
    skin_cancer = st.checkbox("Skin Cancer")
    other_cancer = st.checkbox("Other Cancer")
    depression = st.checkbox("Depression")
    arthritis = st.checkbox("Arthritis")

    # Dropdown (selectbox) for Diabetes
    st.text("")
    diabetes = st.selectbox(
        "Do you have Diabetes?",
        ["No", "Yes"]
    )
    diabetes = 1 if diabetes == 'Yes' else 0
    # Optionally, convert checkboxes to 0/1
    skin_cancer = int(skin_cancer)
    other_cancer = int(other_cancer)
    depression = int(depression)
    arthritis = int(arthritis)
    
    st.text("")
    st.markdown("---")
    st.subheader('🚬 Lifestyle Habits')
    smoking_history = st.selectbox("Do you have the habit of smoking",['Yes','No'])
    smoking_history = 1 if smoking_history == 'Yes' else 0
    alcohol_consumption = st.text_input("Do you ever consume Alcohol, if yes enter the drinks per week else enter 0")
    fruit_consumption = st.text_input("How many times you consume fruits in a day ?")
    green_vegetables_consumption = st.text_input("How often do you eat green vegetables in a day?!")
    fried_potato_consumption = st.text_input("How often you have french fries as you meal in a week?!")
    alcohol_consumption = float(alcohol_consumption) if alcohol_consumption else 0
    fruit_consumption = float(fruit_consumption) if fruit_consumption else 0
    green_vegetables_consumption = float(green_vegetables_consumption) if green_vegetables_consumption else 0
    fried_potato_consumption = float(fried_potato_consumption) if fried_potato_consumption else 0

    st.text("")
    st.markdown("---")
    st.subheader("Thank you for all the details you have provided! To check if you have risk of having heart disease click Predict ")

    ## Data Preparation for the model 
    dataframe_to_be_used = {'General_Health':general_health,'Exercise':exercise,'Skin_Cancer':skin_cancer,'Other_Cancer':other_cancer,'Depression':depression,'Diabetes':diabetes, 'Arthritis':arthritis,'Sex':gender,'Height_(cm)':height_cm,'Weight_(kg)':weight_kg,'BMI':bmi,'Smoking_History':smoking_history, 'Alcohol_Consumption':alcohol_consumption, 'Fruit_Consumption':fruit_consumption,'Green_Vegetables_Consumption':green_vegetables_consumption,'FriedPotato_Consumption':fried_potato_consumption, 'SexBinary':gender,'Age_Category':age}
    dataframe = pd.DataFrame([dataframe_to_be_used])
    # Drop any age_category_... columns that may already exist
    for col in dataframe.columns:
        if col.startswith('Age_Category_'):
            dataframe.pop(col)

    dummied_data = pd.get_dummies(dataframe, columns=['Age_Category'], dtype = 'int')
    for col in expected_age_columns:
        if col not in dummied_data.columns:
            dummied_data[col] = 0
    dummied_data['General_Health'] = dummied_data['General_Health'].apply(lambda x: 'Poor' if x == 'Poor' else 'Good')
    dummied_data['General_Health'] = dummied_data['General_Health'].map({"Poor":1, "Good":0})
    to_be_predicted = dummied_data
    st.write(to_be_predicted)

    ## Prediction of the data 

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'modelPreparation', 'model.pkl')
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(to_be_predicted)
    if st.button("Predict"):
        if predicted[0] == "0":
            st.success("You are not at risk")
        else:
            st.error("You are at risk ")