# ==========================
# STEP 1 : Load Dataset
# ==========================

import pandas as pd
df=pd.read_csv("Housing_price_Dataset.csv")

print("Sample Rows")
print(df.head())

print("\n Data shape")
print(df.shape)

print("\n Data Set Info")
print(df.info())

print("\nSummary Statics")
print(df.describe(include="all"))

print("\nMissing values")
print(df.isnull().sum())

# ==========================
# STEP 2 : Data Cleaning
# ==========================


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["mainroad"]=le.fit_transform(df["mainroad"])
df["guestroom"]=le.fit_transform(df["guestroom"])
df["basement"]=le.fit_transform(df["basement"])
df["hotwaterheating"]=le.fit_transform(df["hotwaterheating"])
df["airconditioning"]=le.fit_transform(df["airconditioning"])
df["prefarea"]=le.fit_transform(df["prefarea"])
df=pd.get_dummies(df,columns=["furnishingstatus"],drop_first=True,dtype=int)

print("\n AFTER ENCODING")
print(df.head())

print('\n Data type')
print(df.dtypes)

# ==========================
# STEP 3 : Model Training
# ==========================


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Features and Tareget 

X=df.drop("price",axis=1)
Y=df["price"]
print("\nFeatures")
print(X.head())
print("\nTarget")
print(Y.head())

# train-test-split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Feature Scaling

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Model Training

model=LinearRegression()
model.fit(X_train,Y_train)

# prediction 

y_pred=model.predict(X_test)

print("\nActual price")
print(Y_test.head())
print("\n predicted price")
print(y_pred[:5])

# Evalution 

print("MAE:",mean_absolute_error(Y_test,y_pred))
print("MSE",mean_squared_error(Y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(Y_test,y_pred)))
print("R2 SCORE:",r2_score(Y_test,y_pred))

# vasualization

plt.figure(figsize=(8,6))
plt.scatter(Y_test,y_pred)
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()])
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs predicted House price")
plt.savefig("Actual_vs_predicted_house.png")
plt.show()


# ==========================
# STEP 4 : User Prediction
# ==========================

print("\n Pridicted your House price")
try:
    area=float(input("enter Area"))
    bedrooms=int(input("enter no of rooms"))
    bathrooms=int(input("enter no of bathroom"))
    stories=int(input("enter no of stories"))
    parking=int(input("enter no of parking"))
    mainroad=input("enter mainroad (yes/no)")
    guestroom=input("enter guest rooom (yes/no)")
    basement=input("enter basement (yes/no)")
    hotwaterheating=input("enter hotwaterheating (yes/no)")
    airconditioning=input("enter airconditioning (yes/no)")
    prefarea=input("enter prefarea (yes/no)")
    furnishingstatus_semi_furnished=input("enter furnishingstatus_semi_furnished (yes/no):")
    furnishingstatus_unfurnishe=input("enter furnishingstatus_unfurnishe (yes/no):")
    mainroad=1 if mainroad.lower()=="yes" else  0
    guestroom= 1 if guestroom.lower()=="yes" else 0
    basement= 1 if basement.lower()=="yes" else 0
    hotwaterheating= 1 if hotwaterheating.lower()=="yes" else 0
    airconditioning= 1 if airconditioning.lower()=="yes" else 0
    prefarea= 1 if prefarea.lower()=="yes" else 0
    furnishingstatus_semi_furnished=1 if furnishingstatus_semi_furnished.lower()=="yes" else 0
    furnishingstatus_unfurnishe = 1 if furnishingstatus_unfurnishe.lower()=="yes" else 0

    user_input=pd.DataFrame([{
        'area':area,
        'bedrooms':bedrooms, 
        'bathrooms':bathrooms, 
        'stories':stories, 
        'mainroad':mainroad, 
        'guestroom':guestroom,
        'basement':basement, 
        'hotwaterheating':hotwaterheating,
        'airconditioning':airconditioning,
        'parking':parking, 
        'prefarea':prefarea,
        'furnishingstatus_semi-furnished':furnishingstatus_semi_furnished, 
        'furnishingstatus_unfurnished':furnishingstatus_unfurnishe,
       }])
    user_input=user_input[X.columns]
    user_scaled=scaler.transform(user_input)
    prediction=model.predict(user_scaled)

    print("\n predicted House Price:",prediction[0])

except Exception as e:
    print("Error",e)