import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================================================================
# LOAD DATASET
# ======================================================================

df = pd.read_csv(r"C:\Users\HARIHARAN SRINIVASAN\Downloads\hema\dataset1.csv")

print(df.head())

# ======================================================================
# DATA PREPROCESSING
# ======================================================================

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nShape:", df.shape)

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

print("\nShape after removing null values:", df.shape)

# ======================================================================
# SPLIT INPUT FEATURES AND OUTPUT
# ======================================================================

# Change this column name if your dataset uses another target name
target_column = "Output"

X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nInput Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# ======================================================================
# CONVERT CATEGORICAL DATA
# ======================================================================

# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True)

# Convert target labels such as Low, Medium, High into numbers
le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded Target Classes:")
print(le.classes_)

# ======================================================================
# TRAIN TEST SPLIT
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ======================================================================
# RANDOM FOREST CLASSIFICATION MODEL
# ======================================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================================================
# PREDICTION
# ======================================================================

prediction = model.predict(X_test)

# ======================================================================
# MODEL EVALUATION
# ======================================================================

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)


print("\nClassification Report:")

labels = np.unique(np.concatenate([y_test, prediction]))

print(classification_report(
    y_test,
    prediction,
    labels=labels,
    target_names=[str(x) for x in labels],
    zero_division=0
))
# ======================================================================
# TRAIN AND TEST SCORE
# ======================================================================

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)

# ======================================================================
# SAVE MODEL + ENCODER + FEATURE COLUMNS (needed by the Flask app)
# ======================================================================

# X.columns AFTER get_dummies is the exact column order the model expects.
# We save it so the Flask app can build the input row in the same order.
joblib.dump(model, "model/soil_fertility_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("\nModel, label encoder, and feature columns saved to /model folder.")
