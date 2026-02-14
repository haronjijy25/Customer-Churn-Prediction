# =============================================================
#        CUSTOMER CHURN PREDICTION & REVENUE RISK ENGINE
# =============================================================
# Author  : Haron Jijy
# Stack   : Python + MySQL + Pandas + ML (Random Forest)
# Purpose : Predict customer churn and calculate revenue risk
# =============================================================


# =========================
# 1. IMPORT LIBRARIES
# =========================
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# 2. CONNECT TO DATABASE
# =========================
engine = create_engine("mysql+mysqlconnector://root:2007@localhost/churn_project")

print("\n[✓] Connected to MySQL")


# =========================
# 3. LOAD DATA
# =========================
df = pd.read_sql("SELECT * FROM customers", engine)

print("[✓] Data Loaded:", df.shape)
print(df.head())


# =========================
# 4. BASIC METRICS
# =========================
churn_rate = df["Churn"].mean() * 100
revenue_risk_total = (df["TotalSpend"] * df["Churn"]).sum()

print(f"\n📊 Overall Churn Rate : {churn_rate:.2f}%")
print(f"💰 Total Revenue Risk : {revenue_risk_total:,.2f}")


# =============================================================
#                    EXPLORATORY DATA ANALYSIS
# =============================================================

plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure()
sns.barplot(x="Complaints", y="Churn", data=df)
plt.title("Churn vs Complaints")
plt.show()

plt.figure()
sns.boxplot(x="Churn", y="Tenure", data=df)
plt.title("Churn vs Tenure")
plt.show()

plt.figure()
sns.boxplot(x="Churn", y="LastPurchaseDaysAgo", data=df)
plt.title("Churn vs Recency")
plt.show()

plt.figure()
sns.histplot(df["TotalSpend"], bins=30)
plt.title("Customer Value Distribution")
plt.show()


# =============================================================
#                     MACHINE LEARNING MODEL
# =============================================================

# Prepare features
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Convert categorical to numeric
X = pd.get_dummies(X, columns=["Region"], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n[✓] Data Split Complete")

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("[✓] Model Trained")

# Predictions
y_pred = model.predict(X_test)

print("\n📈 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =============================================================
#                   FEATURE IMPORTANCE
# =============================================================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n🔥 Top Churn Factors:")
print(importance.head(10))


# =============================================================
#                 REVENUE RISK ENGINE (CORE)
# =============================================================

# Predict churn probability for ALL customers
df_features = pd.get_dummies(df.drop(columns=["CustomerID", "Churn"]), columns=["Region"], drop_first=True)

df["Churn_Probability"] = model.predict_proba(df_features)[:, 1]

# Revenue at risk per customer
df["Revenue_At_Risk"] = df["TotalSpend"] * df["Churn_Probability"]

# Sort high risk customers
risk_customers = df.sort_values(by="Revenue_At_Risk", ascending=False)

print("\n💀 Top 10 High-Risk Customers:")
print(risk_customers[["CustomerID", "TotalSpend", "Churn_Probability", "Revenue_At_Risk"]].head(10))


# =============================================================
#                    EXPORT RESULTS (FOR POWER BI / GITHUB)
# =============================================================

df.to_csv("customer_churn_predictions.csv", index=False)
print("\n[✓] Results exported → customer_churn_predictions.csv")


# =============================================================
#                          END
# =============================================================

