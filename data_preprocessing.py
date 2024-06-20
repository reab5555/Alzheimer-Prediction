import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load the data (assuming it's in a CSV file)
data = pd.read_csv(r"alzheimers_dataset.csv")

# Separate features and target
X = data.drop(['PatientID', 'Diagnosis'], axis=1)
y = data['Diagnosis']

# Print the number of samples for each class
print("Class distribution before resampling:")
print(y.value_counts())

# Define feature types
numeric_features = ['Age', 'ADL', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'SystolicBP', 'AlcoholConsumption',
                    'PhysicalActivity', 'DietQuality', 'SleepQuality', 'BMI']
binary_features = ['Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
                   'HeadInjury', 'Hypertension', 'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
                   'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']
ordinal_features = ['Ethnicity', 'EducationLevel']

# One-hot encode ordinal features
X = pd.get_dummies(X, columns=ordinal_features)

# Scale numerical features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Undersample the majority class if classes are uneven
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Print the number of samples for each class after resampling
print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts())

# Convert all columns to float
X_resampled = X_resampled.astype(float)

# Keep data as numpy arrays for splitting
X_numpy = X_resampled.values
y_numpy = y_resampled.values

# Save the processed data
X_resampled.to_csv("X_resampled.csv", index=False)
pd.Series(y_resampled).to_csv("y_resampled.csv", index=False)
