import pandas as pd
import joblib

REQUIRED_COLUMNS = [
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
def validate_dataset(df):

    missing_columns = []

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns

def preprocess_data(df):

    df = df.drop_duplicates()
    df = df.dropna()

    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'],dayfirst=True)
    df['Journey_Day'] = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month

    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'],format="%H:%M")
    df['Dep_Hour'] = df['Dep_Time'].dt.hour
    df['Dep_Minute'] = df['Dep_Time'].dt.minute

    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'].str[:5],format="%H:%M")
    df['Arrival_Hour'] = df['Arrival_Time'].dt.hour
    df['Arrival_Minute'] = df['Arrival_Time'].dt.minute

    df["Duration_Hours"] = (df["Duration"].str.extract(r'(\d+)h').fillna(0).astype(int))
    df["Duration_Minutes"] = (df["Duration"].str.extract(r'(\d+)m').fillna(0).astype(int))

    df["Duration_Total_Minutes"] = ((df["Duration_Hours"] * 60)+ df["Duration_Minutes"])

    df["Total_Stops"] = (df["Total_Stops"].replace({'non-stop': 0,'1 stop': 1,
            '2 stops': 2,'3 stops': 3,'4 stops': 4}).astype(int))

    df = df.drop(["Date_of_Journey","Dep_Time","Arrival_Time","Duration","Duration_Hours",
            "Duration_Minutes","Route","Additional_Info"],axis=1)

    df = pd.get_dummies(df,columns=["Airline","Source","Destination"],drop_first=True)

    return df

def align_features(df):

    expected_columns = joblib.load(
        r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\saved_model\feature_columns.pkl"
    )

    for column in expected_columns:

        if column not in df.columns:

            df[column] = 0

    df = df[expected_columns]

    return df

