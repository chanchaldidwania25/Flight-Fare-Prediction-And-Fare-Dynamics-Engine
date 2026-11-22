import streamlit as st

st.set_page_config(page_title="Flight Fare Prediction & Fare Dynamics Engine",
    page_icon="✈️",
    layout="wide")
st.title("✈️ Flight Fare Prediction & Fare Dynamics Engine")
st.markdown("""
### Smart Flight Fare Forecasting & Travel Decision Support System

Predict future flight fares, analyse fare movements and receive
intelligent booking recommendations using Machine Learning.

Built around real-world booking dynamics — from lead-time surges and route behavior 
to carrier patterns — this system turns raw fare estimates into 
smarter travel choices.
""")

st.markdown("---")

st.header("🚀 Project Overview")

st.write(
"""
This application helps travellers make informed booking decisions 
by forecasting future airfare trends.

Using machine learning and historical flight data, the system estimates ticket prices, 
measures expected fare movement and provides actionable recommendations 
such as BOOK NOW or WAIT.
"""
)

st.markdown("---")

st.header("✨ Key Features")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ✅ Flight Fare Prediction

    Predict airfare for selected routes
    using machine learning.
    """)

    st.markdown("""
    ✅ Fare Dynamics Engine

    Book Now / Wait recommendations
    based on fare movement analysis.
    """)

with col2:

    st.markdown("""
    ✅ Risk & Savings Analysis

    Understand potential savings and
    expected fare movements.
    """)

    st.markdown("""
    ✅ Dataset Prediction

    Upload flight datasets and generate
    bulk fare predictions.
    """)
st.markdown("---")

st.header("🔄 How It Works")

st.info(
"""
✈️ Flight Details → ⚙️ Feature Engineering → 🤖 Prediction Model → 💰 Fare Prediction → 📈 Price Movement Analysis → ✅ Booking Recommendation
"""
)
st.markdown("---")

st.header("📌 Navigation Guide")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ### ✏️ Flight Fare Prediction

    Predict fares, analyse fare movement, upload datasets
    and download prediction reports.
    """)

with col2:

    st.markdown("""
    ### 📊 Model Insights

    Explore visual analytics, feature importance,
    dataset trends and model performance.
    """)

st.markdown("---")

st.header("🏆 Final Deployed Model")

st.markdown(
    """
    Log-Transformed XGBoost was selected as the final deployment model
    after comparing multiple machine learning algorithms.

    The model demonstrated stronger predictive performance,
    improved handling of fare outliers and better
    generalisation on unseen flight records.
    """
)