import numpy as np
import streamlit as st
import pandas as pd
import joblib
import sys
import os

model = joblib.load(r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\saved_model\fare_model.pkl")

def predict_dataset(df):

    original_df = df.copy()
    processed_df = preprocessing.preprocess_data(df)
    aligned_df = preprocessing.align_features(processed_df)

    predictions = model.predict(aligned_df)
    predicted_fares = np.expm1(predictions)
    
    original_df = (original_df.drop_duplicates().dropna().reset_index(drop=True))
    original_df["Predicted_Fare"] = (predicted_fares.round(2))

    return original_df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","model_training")))

import fare_dynamics 
import preprocessing 

st.set_page_config(page_title="Flight Fare Prediction & Recommendation",page_icon="✈️",layout="wide")
st.title("✈️ Flight Fare Prediction & Recommendation")

prediction_method = st.radio("Choose Prediction Method",
    [
        "✏️ Manual Flight Entry",
        "📂 Upload Dataset"
    ])

if prediction_method == "📂 Upload Dataset":

    st.subheader("📂 Dataset Upload")

    if st.button("📥 Use Sample Dataset"):

        sample_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                "..",
                "..",
                "dataset",
                "Test_set.csv"))

        df = pd.read_csv(sample_dataset_path)

        st.success("Sample Dataset Loaded Successfully ✅")
        st.dataframe(df.head())
        result_df = predict_dataset(df)
        
        st.success(f"✅ Predictions Generated Successfully ({len(result_df)} Records)")
        st.write("### Predicted Fare Preview")
        st.dataframe(result_df.head(10))
        st.session_state["prediction_results"] = result_df
        
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction Results",csv,"predicted_fares.csv","text/csv")

    with st.expander("📂 Upload Custom Dataset"):

        uploaded_file = st.file_uploader("Choose CSV File",type=["csv"])

        if uploaded_file is not None:

            try:

                df = pd.read_csv(uploaded_file)

            except Exception:

                st.error("""Dataset format is not supported.\nPlease upload a dataset following the Test_set.csv structure.""")
                st.stop()

            missing_columns = preprocessing.validate_dataset(df)

            if len(missing_columns) > 0:

                st.error("Dataset format is not supported.")

                with st.expander("📄 Required Dataset Format"):

                    required_columns = [
                        "Airline",
                        "Date_of_Journey",
                        "Source",
                        "Destination",
                        "Route",
                        "Dep_Time",
                        "Arrival_Time",
                        "Duration",
                        "Total_Stops",
                        "Additional_Info"
                    ]

                    for column in required_columns:

                        st.write(f"• {column}")

                st.write("### Missing Columns")

                for column in missing_columns:

                    st.write(f"• {column}")

            else:

                st.success("Dataset validated successfully ✅")
                st.dataframe(df.head())

                result_df = predict_dataset(df)
                st.success(f"✅ Predictions Generated Successfully ({len(result_df)} Records)")
                st.write("### Predicted Fare Preview")
                st.dataframe(result_df.head(10))
                st.session_state["prediction_results"] = result_df
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Prediction Results",csv,"predicted_fares.csv","text/csv")

elif prediction_method == "✏️ Manual Flight Entry":

    st.subheader("✏️ Enter Flight Details")

    airline = st.selectbox(
        "Airline",
        [
            "Air India",
            "GoAir",
            "IndiGo",
            "Jet Airways",
            "Jet Airways Business",
            "Multiple carriers",
            "Multiple carriers Premium economy",
            "SpiceJet",
            "Trujet",
            "Vistara",
            "Vistara Premium economy",
            "Air Asia"
        ]
    )

    source = st.selectbox(
        "Source",
        [
            "Banglore",
            "Chennai",
            "Delhi",
            "Kolkata",
            "Mumbai"
        ]
    )

    destination = st.selectbox(
        "Destination",
        [
            "Banglore",
            "Cochin",
            "Delhi",
            "Hyderabad",
            "Kolkata",
            "New Delhi"
        ]
    )

    journey_date = st.date_input("Journey Date")
    departure_time = st.time_input("Departure Time")
    arrival_time = st.time_input("Arrival Time")

    total_stops_display = st.selectbox(
        "Total Stops",
        [
            "Non-Stop",
            "1 Stop",
            "2 Stops",
            "3 Stops",
            "4 Stops"
        ]
    )
    stop_mapping = {
    "Non-Stop": "non-stop",
    "1 Stop": "1 stop",
    "2 Stops": "2 stops",
    "3 Stops": "3 stops",
    "4 Stops": "4 stops"
}
    total_stops = stop_mapping[total_stops_display]

    col1, col2 = st.columns(2)
    with col1:
        duration_hours = st.selectbox(
        "Hours",
        list(range(0,25))
    )
        
    with col2:
        duration_minutes = st.selectbox(
        "Minutes",
        [0,5,10,15,20,25,30,35,40,45,50,55]
    )
        
    current_fare = st.number_input("Current Fare (₹)",min_value=0.0,value=5000.0)

    if st.button("🚀 Predict Fare"):
        manual_df = pd.DataFrame(
        {
            "Airline": [airline],

            "Date_of_Journey": [journey_date.strftime("%d/%m/%Y")],

            "Source": [source],

            "Destination": [destination],

            "Route": [f"{source} -> {destination}"],

            "Dep_Time": [departure_time.strftime("%H:%M")],

            "Arrival_Time": [arrival_time.strftime("%H:%M")],

            "Duration": [f"{duration_hours}h {duration_minutes}m"],

            "Total_Stops": [total_stops],

            "Additional_Info": ["No info"]
        }
    )

        processed_df = preprocessing.preprocess_data(manual_df)
        aligned_df = preprocessing.align_features(processed_df)
        prediction = model.predict(aligned_df)
        predicted_fare = np.expm1(prediction[0])

        st.markdown("---")
    
        recommendation, percentage_change, risk_score, risk_label, savings = (
        fare_dynamics.fare_dynamics_engine(current_fare,predicted_fare))
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.metric(
        "💰 Predicted Fare",
        f"₹{predicted_fare:,.0f}"
    )
    
        with col2:
            st.metric(
        "📊 Price Movement Score",
        f"{risk_score:.2f}%"
    )
            st.write(risk_label)
            st.progress(int(risk_score))
        if recommendation == "BOOK NOW":
            st.error(
    f"""
    ## ✅ BOOK NOW

    Expected Fare Increase : {percentage_change:.2f}%

    Potential Additional Cost : ₹{savings:,.2f}
    """
)
        
        elif recommendation == "WAIT":
            st.success(
    f"""
    ## ⏳ WAIT

    Expected Fare Reduction : {abs(percentage_change):.2f}%

    Potential Savings : ₹{savings:,.2f}
    """
)
        
        else:
            st.info("⚖️ Fare expected to remain stable.")
        
        if percentage_change > 0:
            st.error(
        f"📈 Expected Fare Increase: +{percentage_change:.2f}%")
    
        elif percentage_change < 0:
            st.success(
        f"📉 Expected Fare Decrease: {abs(percentage_change):.2f}%")
        
        else:
            st.info("No major fare movement expected.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
        "Current Fare",
        f"₹{current_fare:,.0f}"
    )
        
        with col2:
             st.metric(
        "Difference",
        f"₹{abs(predicted_fare-current_fare):,.0f}"
    )
        with st.expander("📊 Why was this fare predicted?"):
                st.write(
        "The prediction was influenced by the following details:"
    )

                explanation_points = [
                f"✈️ Airline: {airline}",

                f"📍 Route: {source} → {destination}",

                f"⏳ Flight Duration: {duration_hours}h {duration_minutes}m",

                f"🛑 Total Stops: {total_stops_display}",

                f"📅 Travel Date: {journey_date.strftime('%d %B %Y')}"]

                for point in explanation_points:
                    st.write(point)

                st.info(
        """
        The fare prediction is based on historical flight patterns observed
        in similar journeys with comparable airlines, durations, stops and travel periods.
        """
    )