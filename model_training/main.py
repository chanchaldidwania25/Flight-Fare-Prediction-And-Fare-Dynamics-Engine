#Importing dependencies needed 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ( mean_squared_error,mean_absolute_error,r2_score)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

#Reteriving dataset

df = pd.read_csv(r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\dataset\Data_Train.csv"
)
print(df.head())
#Perforimg Preprocessing 
print(df.shape,"\n",df.columns,"\n")
print(df.info(),"\n",df.isna().any())
print(df['Arrival_Time'].head(20))
print(df["Arrival_Time"].str.contains(" ").sum())
print(df.nunique(),"\n",df.duplicated().sum())
print(df[df.duplicated()].head())
df=df.drop_duplicates()
print(df.shape)
print(df[df["Route"].isna()])
df.dropna(inplace=True)
print(df.shape)
print(df.isna().any())

#Performing Feature Engineering

df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],dayfirst=True)
df['Journey_Day']= df['Date_of_Journey'].dt.day
df['Journey_Month']= df['Date_of_Journey'].dt.month
print(df[['Date_of_Journey','Journey_Day','Journey_Month']].head())
df['Dep_Time']=pd.to_datetime(df['Dep_Time'],format="%H:%M")
df['Dep_Hour']=df['Dep_Time'].dt.hour
df['Dep_Minute']=df['Dep_Time'].dt.minute
print(df[['Dep_Time','Dep_Hour','Dep_Minute']].head())
print(df["Arrival_Time"].head(20).to_string())
df['Arrival_Time']=pd.to_datetime(df['Arrival_Time'].str[:5],format="%H:%M")
df['Arrival_Hour']=df['Arrival_Time'].dt.hour
df['Arrival_Minute']=df['Arrival_Time'].dt.minute
print(df[['Arrival_Time','Arrival_Hour','Arrival_Minute']].head())
print(df["Duration"].head(20).to_string())
print(df[df["Duration"].str.contains("m") == False]["Duration"].head(20))
df["Duration_Hours"] = df["Duration"].str.extract(r'(\d+)h')
df["Duration_Hours"] = df["Duration_Hours"].fillna(0).astype(int)
df["Duration_Minutes"] = df["Duration"].str.extract(r'(\d+)m')
df["Duration_Minutes"] = df["Duration_Minutes"].fillna(0).astype(int)
print(df[["Duration","Duration_Hours","Duration_Minutes"]].head(20))
df['Duration_Total_Minutes']=((df["Duration_Hours"]*60)+ df["Duration_Minutes"])
print(df["Duration_Total_Minutes"].head(20))
print(df["Total_Stops"].unique())
df['Total_Stops']= df["Total_Stops"].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}).astype(int)
print(df["Total_Stops"].head(20))
print(df['Airline'].value_counts())
print(df["Source"].value_counts())
print("\n")
print(df["Destination"].value_counts())
print(df.columns)
print(df["Route"].head(10))
print(df["Additional_Info"].value_counts())
df=df.drop(["Date_of_Journey","Dep_Time","Arrival_Time","Duration","Duration_Hours","Duration_Minutes","Route","Additional_Info"],axis=1)
print(df.columns)
print(df.dtypes)
print(df["Total_Stops"].head())
print(df["Total_Stops"].dtype)
df = pd.get_dummies(df,columns=["Airline", "Source", "Destination"],drop_first=True)
print(df.columns)
print(df.shape,df.dtypes)

# LINEAR REGRESSION MODEL

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr_model=LinearRegression()
lr_model.fit(X_train,y_train)

y_pred=lr_model.predict(X_test)
print(y_pred[:10])

lr_mae = mean_absolute_error(y_test, y_pred)
lr_mse = mean_squared_error(y_test, y_pred)
lr_rmse = lr_mse ** 0.5
lr_r2 = r2_score(y_test, y_pred)

print("\n\nLinear Regression Model stats:-")
print("MAE:",lr_mae,"\n","MSE:",lr_mse,"\n","RMSE:",lr_rmse,"\n","R2 Score:",lr_r2)

#RANDOM FOREST MODEL

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
rf_model=RandomForestRegressor()
rf_model.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)
print(y_pred[:10])

rf_mae = mean_absolute_error(y_test, y_pred)
rf_mse = mean_squared_error(y_test, y_pred)
rf_rmse = rf_mse ** 0.5
rf_r2 = r2_score(y_test, y_pred)

print("\n\n\nRandom Forest Model stats:-")
print("MAE:",rf_mae,"\n","MSE:",rf_mse,"\n","RMSE:",rf_rmse,"\n","R2 Score:",rf_r2)

print(rf_model.score(X_train, y_train))
print(rf_model.score(X_test, y_test))

#XGBOOST MODEL

xgb_model=XGBRegressor(n_estimators=100,random_state=42)
xgb_model.fit(X_train,y_train)
xgb_y_pred=xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test,xgb_y_pred)
xgb_mse = mean_squared_error(y_test,xgb_y_pred)
xgb_rmse = xgb_mse ** 0.5
xgb_r2 = r2_score(y_test,xgb_y_pred)
print("\n\n\nXGBOOST Model stats:-")
print("MAE:",xgb_mae,"\n","MSE:",xgb_mse,"\n","RMSE:",xgb_rmse,"\n","R2 Score:",xgb_r2)

print("\nTrain R2:", xgb_model.score(X_train, y_train))
print("Test R2:", xgb_model.score(X_test, y_test))

#FINE TUNING XGBOOST

feature_importance = pd.DataFrame({"Feature": X.columns,
        "Importance": xgb_model.feature_importances_})

feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print(feature_importance.head(10))

xgb_model = XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,
    colsample_bytree=0.8,random_state=42)

xgb_model.fit(X_train,y_train)
xgb_y_pred=xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test,xgb_y_pred)
xgb_mse = mean_squared_error(y_test,xgb_y_pred)
xgb_rmse = xgb_mse ** 0.5
xgb_r2 = r2_score(y_test,xgb_y_pred)
print("\n\nXGBOOST Model stats:-")
print("MAE:",xgb_mae,"\n","MSE:",xgb_mse,"\n","RMSE:",xgb_rmse,"\n","R2 Score:",xgb_r2)

print("\nTrain R2:", xgb_model.score(X_train, y_train))
print("Test R2:", xgb_model.score(X_test, y_test))

# LOG TRANSFORM XGBOOST MODEL

print("Reached log transform section")
X = df.drop("Price", axis=1)
y_log= np.log1p(df["Price"])
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X,y_log,test_size=0.2,
    random_state=42)
xgb_log = XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,
    colsample_bytree=0.8,random_state=42)
xgb_log.fit(X_train_log,y_train_log)
log_predictions = xgb_log.predict(X_test_log)
actual_predictions = np.expm1(log_predictions)

actual_y_test = np.expm1(y_test_log)
log_mae = mean_absolute_error(actual_y_test,actual_predictions)
log_mse = mean_squared_error(actual_y_test,actual_predictions)
log_rmse = log_mse ** 0.5
log_r2 = r2_score(actual_y_test,actual_predictions)
print("\nLOG TRANSFORMED XGBOOST")

print("MAE :", log_mae)
print("MSE :", log_mse)
print("RMSE:", log_rmse)
print("R2  :", log_r2)

print("Train R2:",xgb_log.score(X_train_log, y_train_log))
print("Test R2:",xgb_log.score(X_test_log, y_test_log))
print("Completed log transform section")

#Fare Dynamics Engine

def fare_dynamics_engine(current_fare, predicted_fare):

    difference = predicted_fare - current_fare
    percentage_change = (difference / current_fare) * 100

    # Recommendation Logic

    if percentage_change > 10:
        recommendation = "BOOK NOW"

    elif percentage_change < -10:
        recommendation = "WAIT"

    else:
        recommendation = "STABLE"

    # Risk Assessment

    absolute_percentage = abs(percentage_change)

    if absolute_percentage < 5:
        risk_score = "LOW"

    elif absolute_percentage < 15:
        risk_score = "MEDIUM"

    else:
        risk_score = "HIGH"

    # Savings / Loss

    savings_amount = abs(difference)

    return {
        "Current Fare": round(current_fare, 2),
        "Predicted Fare": round(predicted_fare, 2),
        "Difference": round(difference, 2),
        "Percentage Change": round(percentage_change, 2),
        "Recommendation": recommendation,
        "Risk Score": risk_score,
        "Potential Savings/Loss": round(savings_amount, 2)
    }

# TESTING FARE DYNAMICS ENGINE

test_result = fare_dynamics_engine(current_fare=7000,predicted_fare=9000)

print("\nFARE DYNAMICS TEST")

for key, value in test_result.items():
    print(f"{key}: {value}")

import joblib

joblib.dump(
    xgb_log,
    r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\saved_model\fare_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    r"C:\Users\chanc\OneDrive\Desktop\Internship Project\Flight_Fare_Prediction\saved_model\feature_columns.pkl"
)
print("Model Saved Successfully")