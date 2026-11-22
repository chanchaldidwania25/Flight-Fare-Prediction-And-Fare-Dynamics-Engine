import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Insights",page_icon="📊",layout="wide")

st.title("📊 Model Insights Dashboard")

if "prediction_results" in st.session_state:

    result_df = st.session_state["prediction_results"]

    st.header("📋 Latest Prediction Results")

    st.dataframe(
        result_df[
            [
                "Airline",
                "Source",
                "Destination",
                "Predicted_Fare"
            ]
        ].head(10)
    )

    avg_predicted = result_df["Predicted_Fare"].mean()
    highest_predicted = result_df["Predicted_Fare"].max()
    lowest_predicted = result_df["Predicted_Fare"].min()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Predicted Fare",
            f"₹{avg_predicted:,.0f}"
        )

    with col2:
        st.metric(
            "Highest Predicted Fare",
            f"₹{highest_predicted:,.0f}"
        )

    with col3:
        st.metric(
            "Lowest Predicted Fare",
            f"₹{lowest_predicted:,.0f}"
        )
    highest_airline_uploaded = (result_df.groupby("Airline")["Predicted_Fare"].mean().idxmax())

    st.info(
    f"{highest_airline_uploaded} has the highest predicted average fare among the uploaded flights."
)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Predicted Fare Distribution")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(result_df["Predicted_Fare"],kde=True,color="orange",ax=ax)
        st.pyplot(fig)

    with col2:

        st.subheader("✈ Airline vs Predicted Fare")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.boxplot(data=result_df,x="Airline",y="Predicted_Fare",palette="rocket",ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        st.markdown("---")

else:

    st.info(
        """
        No prediction results available yet.

        Generate predictions from the
        Flight Fare Prediction page to view
        uploaded dataset insights here.
        """
    )

st.markdown("---")

df = pd.read_csv(
    r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\dataset\Data_Train.csv"
)

st.header("📊 Key Findings")

st.markdown("""
- Flight duration has a significant impact on fare pricing.
- Airline selection noticeably affects ticket costs.
- Multi-stop routes show different pricing behaviour than non-stop routes.
- Certain destinations consistently attract higher fares.
- Fare prediction accuracy improves when route, duration and airline information are considered together.
""")

col1, col2 = st.columns(2)

with col1:

    st.subheader("💰 Price Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["Price"],kde=True,color="gold",ax=ax)
    st.pyplot(fig)

    avg_price = df["Price"].mean()
    max_price = df["Price"].max()

    st.info(
    f"""
    Average ticket fare is ₹{avg_price:,.0f}.

    The highest recorded fare reaches approximately ₹{max_price:,.0f},
    indicating the presence of premium fare segments and business-class routes.
    """
)

with col2:

    st.subheader("✈ Airline vs Price")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df,x="Airline",y="Price",palette="viridis",ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    highest_airline = (df.groupby("Airline")["Price"].mean().sort_values(ascending=False).index[0])
    highest_price = (df.groupby("Airline")["Price"].mean().max())

    st.info(
    f"{highest_airline} has the highest average fare of approximately ₹{highest_price:,.0f}."
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("⏳ Airline vs Duration")

    duration_df = df.copy()

    duration_df["Duration_Hours"] = (duration_df["Duration"].str.extract(r'(\d+)h')
                                     .fillna(0).astype(int))

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=duration_df,x="Airline",y="Duration_Hours",palette="magma",ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    longest_airline = (duration_df.groupby("Airline")["Duration_Hours"].mean().idxmax())
    avg_duration = (duration_df.groupby("Airline")["Duration_Hours"].mean().max())

    st.info(f"{longest_airline} operates the longest flights with an average duration of approximately {avg_duration:.1f} hours."
)

with col2:

    st.subheader("🛑 Airline vs Stops")

    stops_df = df.copy()
    stop_mapping = {
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    }

    stops_df["Stops"] = (stops_df["Total_Stops"].replace(stop_mapping))

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=stops_df,x="Airline",y="Stops",palette="coolwarm",ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    most_stops_airline = (stops_df.groupby("Airline")["Stops"].mean().idxmax())
    avg_stops = (stops_df.groupby("Airline")["Stops"].mean().max())

    st.info(
    f"{most_stops_airline} operates routes with an average of {avg_stops:.1f} stops."
)
st.markdown("---")

st.subheader("🎯 Feature Importance")

feature_df = pd.DataFrame({

        "Feature": [
            "Jet Airways Business",
            "Destination Hyderabad",
            "Jet Airways",
            "Duration",
            "IndiGo",
            "Total Stops",
            "Destination Delhi",
            "Journey Month",
            "Departure Hour",
            "Arrival Hour"
        ],

        "Importance": [
            0.449,
            0.213,
            0.122,
            0.098,
            0.051,
            0.027,
            0.025,
            0.020,
            0.005,
            0.006
        ]
    })

feature_df = feature_df.sort_values(by="Importance",ascending=False)

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=feature_df,x="Importance",y="Feature",color="limegreen",ax=ax)
st.pyplot(fig)

st.info(
    f"""
    The most influential fare driver is {feature_df.iloc[0]['Feature']}.

    Airline choice, destination and journey duration collectively
    contribute most to fare prediction accuracy.
    """
)

st.markdown("---")
with st.expander("⚙️ Technical Model Information"):

    st.subheader("🏆 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "Log XGBoost"
        ],
        "R2": [
            0.584,
            0.815,
            0.836,
            0.843
        ]
    })

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=comparison_df,x="Model",y="R2",palette="plasma",ax=ax)
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            "₹1134"
        )

    with col2:
        st.metric(
            "RMSE",
            "₹1811"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.843"
        )

    st.success(
        """
        ✅ Final Selected Model: Log-XGBoost

        The model was evaluated against multiple machine learning
        algorithms including Linear Regression, Random Forest
        and standard XGBoost.

        Log-XGBoost achieved the best overall performance
        with the highest R² score while maintaining lower
        prediction errors and stronger generalisation on
        previously unseen flight records.

        The logarithmic transformation also improved the
        handling of fare outliers, making the predictions
        more stable and reliable.

        Based on accuracy, robustness and consistency,
        Log-XGBoost was selected as the final deployment model.
        """
    )