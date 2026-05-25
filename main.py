import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("customer_churn.csv")

# -------------------------------
# Display Dataset
# -------------------------------
print("\n===== DATASET PREVIEW =====\n")
print(df.head())

# -------------------------------
# Encode Categorical Columns
# -------------------------------
categorical_columns = [
    'Gender',
    'SubscriptionType',
    'PaymentMethod',
    'ContractType',
    'Churn'
]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# -------------------------------
# Dataset Information
# -------------------------------
print("\n===== DATASET INFO =====\n")
print(df.info())

# Create numeric-only dataframe
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# -------------------------------
# Visualization 1:
# Churn Distribution
# -------------------------------
plt.figure(figsize=(6,4))

sns.countplot(x='Churn', data=df)

plt.title("Customer Churn Distribution")

plt.savefig("outputs/churn_distribution.png")

plt.show()

# -------------------------------
# Visualization 2:
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

# -------------------------------
# Visualization 3:
# Login Frequency vs Churn
# -------------------------------
plt.figure(figsize=(6,4))

sns.boxplot(
    x='Churn',
    y='LoginFrequency',
    data=df
)

plt.title("Login Frequency vs Churn")

plt.savefig("outputs/login_vs_churn.png")

plt.show()

print("\nGraphs generated successfully!")

# -------------------------------
# Machine Learning Model
# -------------------------------

# Features and Target
X = df.drop(['CustomerID', 'Churn'], axis=1)

y = df['Churn']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Classification Report
report = classification_report(y_test, predictions)

# -------------------------------
# Print Results
# -------------------------------
print("\n===== MODEL RESULTS =====\n")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(report)

# -------------------------------
# Save Results
# -------------------------------
with open("outputs/model_accuracy.txt", "w") as file:

    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

    file.write(report)

print("\nResults saved in outputs folder!")