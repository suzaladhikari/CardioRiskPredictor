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

    ## Prediction of the data 

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'modelPreparation', 'model.pkl')
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(to_be_predicted)
    if st.button("Predict"):
        if int(predicted[0]) == 0:
            st.success("Congratulations! You are not at risk")
        else:
            st.error("You are at risk!")

    
    #Disclamer 

    st.sidebar.markdown("""
    ---  
    ### ⚠️ Disclaimer

    This application was developed as part of a data science project and is based on a relatively small and synthetic dataset. While the model demonstrates how lifestyle and clinical features can be used to predict the risk of having the heart disease, it is **not trained on real clinical-scale data** and should **not be used for any actual medical decision-making**.

    Please explore and learn from this tool, but do not interpret its output as medical advice or a diagnostic reference. Always consult a qualified healthcare professional for real-world health concerns.
    """)
            


# About the model page 

if page == ' ℹ️ About the Model':
    st.header("Thank you so much for using the model !")
    st.error("This model is only for learning purpose, don't rely on this model for your medical evaluation! ")
    st.markdown("---")
    st.subheader("Hey there, from the developer of this app! You can know more about me in the   👨‍💻 Developer section of the navigation bar at the top right.")
    st.subheader(" Talking about the model, different five models were tested on the same data with 160000 rows and 30 columns or features and then the model performing better was thus selected.")
    #Logistic Regression
    st.markdown("---")

    st.header("Model One : Logistic Regression")
    st.subheader("The first model to be tested was logistic regression. The model performed good but not better compared to others.The accuracy scores of the model after all the pipeline adjustments and hyperparameter tuning were as follows:")
    st.write("Accuracy of the model : 74%")
    st.write("Precison of the model : 71%")
    st.write("Recall of the model: 73%")

    # Decision Tree Classifier
    st.markdown("---")
    st.header("Model Two : Decision Tree Classifier")
    st.subheader("The second model to be tested was Decision Tree Classifier.The accuracy scores of the model after all the pipeline adjustments and hyperparameter tuning were as follows:")
    st.write("Accuracy of the model : 73%")
    st.write("Precison of the model : 75%")
    st.write("Recall of the model: 61%")


    # Bagging Classifier
    st.markdown("---")
    st.header("Model Three :  Bagging Classifier")
    st.subheader("The third model to be tested was  Bagging Classifier.The estimator used in the model was the Decision Tree Classifier.The model outperformed every model and was selected as the main model for the prediciton. The accuracy scores of the model after all the pipeline adjustments and hyperparameter tuning were as follows:")
    st.write("Accuracy of the model : 80%")
    st.write("Precison of the model : 84%")
    st.write("Recall of the model: 74%")


    # Random Forest Classifier
    st.markdown("---")
    st.header("Model Four:  Random Forest Classifier")
    st.subheader("The fourth model to be tested was  Random Forest Classifier.The estimator used in the model was the Decision Tree Classifier by default. The accuracy scores of the model after all the pipeline adjustments and hyperparameter tuning were as follows:")
    st.write("Accuracy of the model : 79%")
    st.write("Precison of the model : 82%")
    st.write("Recall of the model: 71%")

    # XGBoost Classifier
    st.markdown("---")
    st.header("Model Five:   XGBoost Classifier")
    st.subheader("The fifth and last model to be tested was  XGBoost Classifier. The ScikitLearn API was used to train and test the data. The accuracy scores of the model after all the pipeline adjustments and hyperparameter tuning were as follows:")
    st.write("Accuracy of the model : 79%")
    st.write("Precison of the model : 82%")
    st.write("Recall of the model: 74%")

    st.markdown("---")

    #Disclamer 

    st.sidebar.markdown("""
    ---  
    ### ⚠️ Disclaimer

    This application was developed as part of a data science project and is based on a relatively small and synthetic dataset. While the model demonstrates how lifestyle and clinical features can be used to predict the risk of having the heart disease, it is **not trained on real clinical-scale data** and should **not be used for any actual medical decision-making**.

    Please explore and learn from this tool, but do not interpret its output as medical advice or a diagnostic reference. Always consult a qualified healthcare professional for real-world health concerns.
    """)


    ## Metrics used to evaluate the model's performance 

    st.header("Metrics used to choose the best model")
    st.text("")
    st.header("1. ROC-AUC-CURVE")
   
    st.subheader("The ROC-AUC curve was used in order to evaluate the best performance. The roc auc score of all the models were compared in the single chart !")
    st.image("app/rocauc.png")

    st.header("2. Precision-Recall Curve")
    st.subheader("The Precision-Recall Curve")
    st.subheader("The precision-recall curve of all the models has been shown in the diagram below!")
    st.image("app/precisionrecall.png")

    st.subheader("Based on the results, the roc auc score and area under the curve of precision recall curve, the bagging classifer outperforms every single of the model. Even though, the random forest classifier and XGBoost come closer, Bagging Classifier was considered the main model!.")

# About the developer

if page == ' 👨‍💻 Developer':
    st.title("Greetings 👋 ¡Hola! 👋 Bonjour")
    st.text("")
    st.subheader("Hi, I'm **Sujal Adhikari**, a sophomore at Caldwell University and an aspiring Data Scientist. This project means a lot to me—it’s my very first step into real-world machine learning, built from the ground up through hard work, persistence, and a deep desire to learn. There were challenges along the way, but every late night and every debugging session was worth it. I’m incredibly grateful to **Dr. Vlad Veksler** for his guidance and for helping me understand what it truly takes to grow in this field. Creating something meaningful using what I’ve learned so far has been both humbling and empowering. This is just the beginning, and I’m excited to keep learning, growing, and building. Thank you for being a part of this journey.")
    st.text("")
    st.subheader("Thank You 🙏 Gracias! 🙏 Merci")

    ##Side bar 
    st.sidebar.title("Meet the developer")
    st.sidebar.header("Sujal Adhikari")
    st.sidebar.write("New York Metropolitan Area")
    st.sidebar.write("Data Science | Data Analysis | Machine Learning")
    st.text("")
    st.sidebar.text("Thank you for trying out my first web app built with scikit-learn! This journey has had its challenges, but I'm proud to share something with real-life meaning. It's just the beginning—and purely educational for now. I’ll keep building and sharing more. Feel free to connect with me and follow my journey below!")
    st.sidebar.markdown("[Github](https://github.com/suzaladhikari)", unsafe_allow_html=True)
    st.sidebar.markdown("[Twitter](https://twitter.com/LifeOfSujal)", unsafe_allow_html=True)
    st.sidebar.markdown("sujal.adhikari.ds@gmail.com")
