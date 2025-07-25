import streamlit as st

st.set_page_config(page_title="Hospital Readmission Risk Prediction", layout="wide")

st.title("🏥 Hospital Readmission Risk Prediction System")

menu = ["Patient Risk Prediction", "Insights Dashboard", "Admin/Analytics Panel"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Patient Risk Prediction":
    st.header("📋 Patient Risk Prediction")
    st.info("Input patient details to predict 30-day readmission risk.")

    import sys
    from pathlib import Path
    sys.path.append(str(Path("src/models").resolve().parent))
    from models.readmission_prediction import ReadmissionPredictor
    sys.path.append(str(Path("src/data_processing").resolve().parent))
    from data_processing.feature_engineering import engineer_features

    @st.cache_resource
    def get_trained_predictor():
        predictor = ReadmissionPredictor()
        df = predictor.load_and_prepare_data()
        predictor.preprocess_data(df)
        predictor.train_model(model_type='random_forest')
        return predictor

    predictor = get_trained_predictor()

    # Raw input form (matching CSV columns)
    with st.form("patient_form"):
        name = st.text_input("Name", "John Doe")
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        blood_type = st.text_input("Blood Type", "A+")
        medical_condition = st.text_input("Medical Condition", "Diabetes")
        date_of_admission = st.date_input("Date of Admission")
        doctor = st.text_input("Doctor", "Dr. Smith")
        hospital = st.text_input("Hospital", "General Hospital")
        insurance_provider = st.text_input("Insurance Provider", "HealthCare Inc.")
        billing_amount = st.number_input("Billing Amount", min_value=0.0, value=1000.0)
        room_number = st.text_input("Room Number", "101")
        admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent", "Other"])
        discharge_date = st.date_input("Discharge Date")
        medication = st.text_input("Medication", "Paracetamol")
        test_results = st.selectbox("Test Results", ["Normal", "Inconclusive", "Abnormal"])
        submitted = st.form_submit_button("Predict Readmission Risk")

    if submitted:
        # Build raw patient dict
        patient_raw = {
            'Name': str(name),
            'Age': int(age),
            'Gender': str(gender),
            'Blood Type': str(blood_type),
            'Medical Condition': str(medical_condition),
            'Date of Admission': str(date_of_admission),
            'Doctor': str(doctor),
            'Hospital': str(hospital),
            'Insurance Provider': str(insurance_provider),
            'Billing Amount': float(billing_amount),
            'Room Number': str(room_number),
            'Admission Type': str(admission_type),
            'Discharge Date': str(discharge_date),
            'Medication': str(medication),
            'Test Results': str(test_results)
        }
        import pandas as pd
        df_patient = pd.DataFrame([patient_raw])
        # Run through feature engineering
        try:
            df_engineered = engineer_features(df_patient)
            # Extract only the features needed for prediction
            features = predictor.feature_names
            patient_features = df_engineered[features].iloc[0].to_dict()
            # Debug: Show dtypes
            st.write('Feature values and types:', {k: (v, type(v)) for k, v in patient_features.items()})
            # Explicitly cast numeric features
            numeric_casts = {
                'Age': int,
                'Length_of_Stay': int,
                'Previous_Visits': int,
                'Billing Amount': float,
                'Test_Result_Score': float,
                'High_Risk_Doctor': int
            }
            for k, cast in numeric_casts.items():
                if k in patient_features:
                    try:
                        patient_features[k] = cast(patient_features[k])
                    except Exception:
                        pass
            result = predictor.predict_single(patient_features)
            st.subheader("Prediction Result")
            st.write(f"**Likely to be readmitted in 30 days:** {'Yes' if result['prediction'] else 'No'}")
            st.write(f"**Probability:** {result['probability']*100:.1f}%")
            st.write(f"**Risk Explanation:** {result['explanation']}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif choice == "Insights Dashboard":
    st.header("📊 Insights Dashboard")
    st.info("Visualize readmission trends and key insights.")
    # TODO: Display charts and recommendations
    st.write("(Charts and insights will appear here.)")

elif choice == "Admin/Analytics Panel":
    st.header("🛠️ Admin/Analytics Panel")
    st.info("Upload new data and monitor trends.")
    # TODO: Add file uploader and batch prediction
    st.write("(Admin tools and analytics will appear here.)") 