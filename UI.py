import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # Assuming the model is stored as a .pkl file
from streamlit_option_menu import option_menu
# Assuming normalize_standardize_columns are columns that need normalization or standardization
normalize_standardize_columns = [
    "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
    "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", 
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"
]

# Load pre-trained model (Random Forest Classifier or any other model)
model = joblib.load("D:/projects/AlzheimerPredictionUI/random_forest_model.pkl")

def prepare_input_data(input_data, model):
    # Map categorical data
    input_data["Gender"] = 0 if input_data["Gender"] == "Male" else 1
    ethnicity_mapping = {"Caucasian": 0, "African American": 1, "Asian": 2, "Hispanic": 3}
    input_data["Ethnicity"] = ethnicity_mapping.get(input_data["Ethnicity"], 0)
    education_mapping = {"None": 0, "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    input_data["EducationLevel"] = education_mapping.get(input_data["EducationLevel"], 0)

    # Yes/No Features (convert "Yes"/"No" to 1/0)
    yes_no_features = [
        "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", 
        "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems", "Confusion", 
        "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"
    ]
    for feature in yes_no_features:
        input_data[feature] = 1 if input_data[feature] == "Yes" else 0

    # Ensure all columns are present for numeric features (based on your dataset columns)
    numeric_columns = [
        "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
        "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", 
        "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"
    ]
    
    # Normalize and standardize numeric columns
    numeric_features = {col: input_data[col] for col in numeric_columns if col in input_data}
    numeric_df = pd.DataFrame([numeric_features])

    # Normalize data
    min_max_scaler = MinMaxScaler()
    numeric_df[numeric_columns] = min_max_scaler.fit_transform(numeric_df[numeric_columns])
    
    # Standardize data
    standard_scaler = StandardScaler()
    numeric_df[numeric_columns] = standard_scaler.fit_transform(numeric_df[numeric_columns])
    
    # Merge normalized and standardized data with other features
    processed_data = {
        **numeric_df.iloc[0].to_dict(),
        **{key: input_data[key] for key in input_data if key not in numeric_columns}
    }

    # Ensure the processed data columns match the model's expected features
    try:
        model_columns = model.feature_names_in_  # Assuming the model has this attribute
        processed_data = {col: processed_data.get(col, 0) for col in model_columns}  # Add missing columns with default 0
    except AttributeError:
        # Handle case where model doesn't have feature names, simply use the keys in processed_data
        pass

    # Convert to DataFrame for prediction
    df = pd.DataFrame([processed_data])
    return df

# Streamlit UI
def main():
    # Apply custom CSS to make the menu span the page width and pin it to the top
    st.markdown(
        """
        <style>
        .nav-link {
            display: block;
            text-align: center;
            flex: 1;
        }
        div[data-testid="stHorizontalBlock"] {
            display: flex;
            justify-content: space-between;
            width: 100%;
            position: fixed;
            top: 0;
            z-index: 1000;
            background-color: white;
            border-bottom: 1px solid #ddd;
            padding: 10px 0;
        }
        .main-content {
            margin-top: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Create a horizontal menu spanning the page width
    page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction", "Visualisations"],
        icons=['house', 'book', 'bar-chart'],  # Add icons for each menu item
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("Alzheimer Prediction")


    if page == "Home":

    # Add an image to the top of the page
        st.image(
            "D:\projects\AlzheimerPredictionUI\\3d-medical-background-with-male-head-brain-dna-strands.jpg",
            caption="Alzheimer's Brain",
            use_container_width=True
        )

        # Title and welcome text
        st.title("Alzheimer Prediction")

        st.subheader("Welcome to the Alzheimer's Risk Prediction Tool")
        st.write(
            """
            This tool uses a machine learning model to predict the risk of Alzheimer's based on various health and lifestyle factors.
            Please fill out the form in the prediction Page to get your prediction.
            """
        )


       
        # Footer
        st.markdown(
            """
            <style>
            footer {visibility: hidden;}
            .reportview-container .main .block-container{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
                 </style>
           
            """,
            unsafe_allow_html=True
        )

    elif page == "Prediction":
    # Input form for user data
        with st.form(key="input_form"):
            age = st.number_input("Age", min_value=55, max_value=100, value=60)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian", "Hispanic"])
            education_level = st.selectbox("Education Level", options=["None", "High School", "Bachelor's", "Master's", "PhD"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0,step=0.0000001,format="%.5f")
            smoking = st.selectbox("Smoking", options=["Yes", "No"])
            alcohol_consumption = st.number_input("Alcohol Consumption", min_value=0.0, max_value=20.0, value=2.0,step=0.0000001,format="%.5f")
            physical_activity = st.number_input("Physical Activity", min_value=0.0, max_value=10.0, value=5.0,step=0.0000001,format="%.5f")
            diet_quality = st.number_input("Diet Quality", min_value=0.0, max_value=10.0, value=5.0,step=0.0000001,format="%.5f")
            sleep_quality = st.number_input("Sleep Quality", min_value=0.0, max_value=10.0, value=5.0,step=0.0000001,format="%.5f")
            family_history = st.selectbox("Family History of Alzheimer's", options=["Yes", "No"])
            cardiovascular_disease = st.selectbox("Cardiovascular Disease", options=["Yes", "No"])
            diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
            depression = st.selectbox("Depression", options=["Yes", "No"])
            head_injury = st.selectbox("Head Injury", options=["Yes", "No"])
            hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90.0, max_value=200.0, value=120.0 , step=0.0000001,format="%.5f")
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=60.0, max_value=120.0, value=80.0 , step=0.0000001,format="%.5f")
            cholesterol_total = st.number_input("Total Cholesterol", min_value=130.0, max_value=300.0, value=200.0,step=0.0000001,format="%.5f")
            cholesterol_ldl = st.number_input("LDL Cholesterol", min_value=50.0, max_value=200.0, value=100.0,step=0.0000001,format="%.5f")
            cholesterol_hdl = st.number_input("HDL Cholesterol", min_value=20.0, max_value=100.0, value=50.0,step=0.0000001,format="%.5f")
            cholesterol_triglycerides = st.number_input("Triglycerides", min_value=50.0, max_value=400.0, value=150.0,step=0.0000001,format="%.5f")
            mmse = st.number_input("MMSE", min_value=0.0, max_value=30.0, value=25.0,step=0.0000001)
            functional_assessment = st.number_input("Functional Assessment", min_value=0.0, max_value=10.0, value=5.0,step=0.0000001,format="%.5f")
            memory_complaints = st.selectbox("Memory Complaints", options=["Yes", "No"])
            behavioral_problems = st.selectbox("Behavioral Problems", options=["Yes", "No"])
            adl = st.number_input("Activities of Daily Living (ADL)", min_value=0.0, max_value=10.0, value=5.0,step=0.0000001,format="%.5f")
            confusion = st.selectbox("Confusion", options=["Yes", "No"])
            disorientation = st.selectbox("Disorientation", options=["Yes", "No"])
            personality_changes = st.selectbox("Personality Changes", options=["Yes", "No"])
            difficulty_completing_tasks = st.selectbox("Difficulty Completing Tasks", options=["Yes", "No"])
            forgetfulness = st.selectbox("Forgetfulness", options=["Yes", "No"])

            submit_button = st.form_submit_button(label="Predict")

        # When the user clicks "Predict", process the input data
        if submit_button:
            input_data = {
                "Age": age,
                "Gender": gender,
                "Ethnicity": ethnicity,
                "EducationLevel": education_level,
                "BMI": bmi,
                "Smoking": smoking,
                "AlcoholConsumption": alcohol_consumption,
                "PhysicalActivity": physical_activity,
                "DietQuality": diet_quality,
                "SleepQuality": sleep_quality,
                "FamilyHistoryAlzheimers": family_history,
                "CardiovascularDisease": cardiovascular_disease,
                "Diabetes": diabetes,
                "Depression": depression,
                "HeadInjury": head_injury,
                "Hypertension": hypertension,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "CholesterolTotal": cholesterol_total,
                "CholesterolLDL": cholesterol_ldl,
                "CholesterolHDL": cholesterol_hdl,
                "CholesterolTriglycerides": cholesterol_triglycerides,
                "MMSE": mmse,
                "FunctionalAssessment": functional_assessment,
                "MemoryComplaints": memory_complaints,
                "BehavioralProblems": behavioral_problems,
                "ADL": adl,
                "Confusion": confusion,
                "Disorientation": disorientation,
                "PersonalityChanges": personality_changes,
                "DifficultyCompletingTasks": difficulty_completing_tasks,
                "Forgetfulness": forgetfulness
            }

            # Prepare the input data
            processed_data = prepare_input_data(input_data, model)

            # Make prediction
            prediction = model.predict(processed_data)

# Display the result as a card above the page
            if prediction == 1:
                st.markdown(
                    """
                    <div style="width: 80%; margin: 20px auto; padding: 20px; #background-color: #ffcccc; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h2 style="color: red;">The Model Predicts That The Person Is At Risk Of Alzheimer's.</h2>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.warning("        Take care Of Your Health And Go To The Doctor         ")
            else:
                st.markdown(
                    """
                    <div style="width: 80%; margin: 20px auto; padding: 20px; #background-color: #ffffff; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h2 style="color: Aqua;">The Model Predicts That The Person Is Not At Risk Of Alzheimer's.</h2>
                    </div>
                    """
                    , unsafe_allow_html=True

                )
                st.success("Congratulations ,, You Are Done Sucess ,  And You don't have The risk of Alzheimer's ")
                st.balloons()


    elif page == "Visualisations":
        st.caption("Welcome in Visualisations Page You Can Show Some Visualisations About Alzheimer's Risk ")
        st.divider()

        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003146.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()

        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003237.png" , 
            ##caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003252.png",
            ##caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003319.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003333.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003146.png",
            ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003356.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003424.png",
            ##caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 002933.png",
            ##caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 003048.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.image(
            "D:\projects\AlzheimerPredictionUI\Screenshot 2024-12-22 002914.png",
           ## caption="This Visualisation Display The Relation Between The Gender And The Percentage Of People With Alzheimer's Risk",
            use_container_width=True
        )
        st.divider()
        st.chat_message("ai")
        st.markdown(
                    """
                    <div style="width: 80%; margin: 20px auto; padding: 20px; #background-color: #ffffff; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h2 style="color: white;">Thanks For Your Intersting.</h2>
                    </div>
                    """
                    , unsafe_allow_html=True

                )
        st.feedback()
        

        
if __name__ == "__main__":
    main()
