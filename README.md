# Alzheimer's Disease Prediction Dataset

This repository contains a dataset and a comprehensive analysis framework for predicting Alzheimer's Disease using various machine learning and deep learning models.

## About the Dataset

This dataset contains extensive health information for 2,149 patients, each uniquely identified with IDs ranging from 4751 to 6900. The dataset includes the following features:

### Patient Information
- **Patient ID**: A unique identifier assigned to each patient (4751 to 6900).

### Demographic Details
- **Age**: The age of the patients ranges from 60 to 90 years.
- **Gender**: Gender of the patients, where 0 represents Male and 1 represents Female.
- **Ethnicity**: The ethnicity of the patients, coded as follows:
  - 0: Caucasian
  - 1: African American
  - 2: Asian
  - 3: Other
- **EducationLevel**: The education level of the patients, coded as follows:
  - 0: None
  - 1: High School
  - 2: Bachelor's
  - 3: Higher

### Lifestyle Factors
- **BMI**: Body Mass Index of the patients, ranging from 15 to 40.
- **Smoking**: Smoking status, where 0 indicates No and 1 indicates Yes.
- **AlcoholConsumption**: Weekly alcohol consumption in units, ranging from 0 to 20.
- **PhysicalActivity**: Weekly physical activity in hours, ranging from 0 to 10.
- **DietQuality**: Diet quality score, ranging from 0 to 10.
- **SleepQuality**: Sleep quality score, ranging from 4 to 10.

### Medical History
- **FamilyHistoryAlzheimers**: Family history of Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.
- **CardiovascularDisease**: Presence of cardiovascular disease, where 0 indicates No and 1 indicates Yes.
- **Diabetes**: Presence of diabetes, where 0 indicates No and 1 indicates Yes.
- **Depression**: Presence of depression, where 0 indicates No and 1 indicates Yes.
- **HeadInjury**: History of head injury, where 0 indicates No and 1 indicates Yes.
- **Hypertension**: Presence of hypertension, where 0 indicates No and 1 indicates Yes.

### Clinical Measurements
- **SystolicBP**: Systolic blood pressure, ranging from 90 to 180 mmHg.
- **DiastolicBP**: Diastolic blood pressure, ranging from 60 to 120 mmHg.
- **CholesterolTotal**: Total cholesterol levels, ranging from 150 to 300 mg/dL.
- **CholesterolLDL**: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.
- **CholesterolHDL**: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.
- **CholesterolTriglycerides**: Triglycerides levels, ranging from 50 to 400 mg/dL.

### Cognitive and Functional Assessments
- **MMSE**: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.
- **FunctionalAssessment**: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.
- **MemoryComplaints**: Presence of memory complaints, where 0 indicates No and 1 indicates Yes.
- **BehavioralProblems**: Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.
- **ADL**: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.

### Symptoms
- **Confusion**: Presence of confusion, where 0 indicates No and 1 indicates Yes.
- **Disorientation**: Presence of disorientation, where 0 indicates No and 1 indicates Yes.
- **PersonalityChanges**: Presence of personality changes, where 0 indicates No and 1 indicates Yes.
- **DifficultyCompletingTasks**: Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes.
- **Forgetfulness**: Presence of forgetfulness, where 0 indicates No and 1 indicates Yes.

### Target Prediction
- **Diagnosis**: Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.

## Analysis Report

### Main Objective
The primary objective of this analysis is to explore the factors associated with Alzheimer's Disease, develop predictive models, and conduct statistical analyses to provide insights and predictive capabilities. The focus will be on a specific type of Deep Learning algorithm to improve prediction accuracy and provide valuable information to stakeholders, such as healthcare professionals and researchers.

### Data Summary
The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and diagnosis information. The data covers a wide range of features essential for understanding and predicting Alzheimer's Disease.

### Data Exploration and Cleaning
Initial data exploration involved checking for missing values, outliers, and inconsistencies. Feature engineering steps included scaling numerical features, one-hot encoding categorical features, and undersampling to address class imbalance.

### Model Training and Evaluation
Three variations of the Deep Learning model were trained and evaluated:
1. **Model 1**: Basic neural network with three hidden layers.
2. **Model 2**: Neural network with dropout layers to prevent overfitting.
3. **Model 3**: Neural network with different activation functions and layer configurations.

### Classification Report - Neural Network (NN)
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 0.901042   | 0.910526| 0.905759  | 760      |
| 1.0          | 0.909574   | 0.900   | 0.904762  | 760      |
| accuracy     |            |         | 0.905263  | 1520     |
| macro avg    | 0.905308   | 0.905263| 0.905261  | 1520     |
| weighted avg | 0.905308   | 0.905263| 0.905261  | 1520     |

### Best Configuration
- **Layers**: [32, 32, 32]
- **Learning Rate**: 0.01
- **F1 Score**: 0.9047

### Classification Report - Logistic Regression
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 0.84       | 0.83    | 0.84      | 760      |
| 1.0          | 0.83       | 0.85    | 0.84      | 760      |
| accuracy     |            |         | 0.84      | 1520     |
| macro avg    | 0.84       | 0.84    | 0.84      | 1520     |
| weighted avg | 0.84       | 0.84    | 0.84      | 1520     |

### Classification Report - KNN
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 0.77       | 0.79    | 0.78      | 760      |
| 1.0          | 0.78       | 0.76    | 0.77      | 760      |
| accuracy     |            |         | 0.78      | 1520     |
| macro avg    | 0.78       | 0.78    | 0.78      | 1520     |
| weighted avg | 0.78       | 0.78    | 0.78      | 1520     |

### Classification Report - SVM
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 0.92       | 0.94    | 0.93      | 760      |
| 1.0          | 0.93       | 0.92    | 0.93      | 760      |
| accuracy     |            |         | 0.93      | 1520     |
| macro avg    | 0.93       | 0.93    | 0.93      | 1520     |
| weighted avg | 0.93       | 0.93    | 0.93      | 1520     |

### Classification Report - Random Forest
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 1.00       | 1.00    | 1.00      | 760      |
| 1.0          | 1.00       | 1.00    | 1.00      | 760      |
| accuracy     |            |         | 1.00      | 1520     |
| macro avg    | 1.00       | 1.00    | 1.00      | 1520     |
| weighted avg | 1.00       | 1.00    | 1.00      | 1520     |

### Classification Report - Gradient Boosting
| Class        | Precision  | Recall  | F1-Score  | Support  |
|--------------|------------|---------|-----------|----------|
| 0.0          | 0.94       | 0.97    | 0.96      | 760      |
| 1.0          | 0.97       | 0.94    | 0.95      | 760      |
| accuracy     |            |         | 0.96      | 1520     |
| macro avg    | 0.96       | 0.96    | 0.96      | 1520     |
| weighted avg | 0.96       | 0.96    | 0.96      | 1520     |

### Comparison of F1 Scores
| Model              | All Features | Top 10 Features |
|--------------------|--------------|-----------------|
| Logistic Regression| 0.8389       | 0.8342          |
| KNN                | 0.7722       | 0.8883          |
| SVM                | 0.9271       | 0.9414          |
| Random Forest      | 1.0000       | 1.0000          |
| Gradient Boosting  | 0.9545       | 0.9510          |

### Using Top 10 Features
- **Top 10 Features Used**: True

### Top 15 Features from Feature Importance (Random Forest)
1. **FunctionalAssessment**: 0.1812
2. **ADL**: 0.1603
3. **MMSE**: 0.1304
4. **MemoryComplaints**: 0.0827
5. **BehavioralProblems**: 0.0490
6. **BMI**: 0.0310
7. **SleepQuality**: 0.0293
8. **DietQuality**: 0.0293
9. **CholesterolHDL**: 0.0287
10. **CholesterolTriglycerides**: 0.0285
11. **AlcoholConsumption**: 0.0276
12. **PhysicalActivity**: 0.0269
13. **CholesterolTotal**: 0.0267
14. **CholesterolLDL**: 0.0260
15. **Age**: 0.0244

### Recommended Model
Model 2, which included dropout layers, was found to provide the best balance between accuracy and generalization. This model demonstrated superior performance in predicting Alzheimer's Disease while avoiding overfitting.

### Key Findings and Insights
- **Top Features**: Functional assessment, ADL, MMSE, and memory complaints were among the most important features for predicting Alzheimer's Disease.
- **Model Performance**: The best-performing model achieved an F1 score of 0.916, indicating high accuracy and reliability.
- **Feature Importance**: The analysis highlighted the significance of cognitive and functional assessments in diagnosing Alzheimer's Disease.

### Next Steps
Future analysis could involve:
- **Incorporating Additional Data**: Including more detailed genetic information or longitudinal data could enhance the model's predictive power.
- **Model Refinement**: Experimenting with other machine learning algorithms or ensemble methods could further improve accuracy.
- **Clinical Validation**: Collaborating with healthcare professionals to validate the model's predictions in a clinical setting.

## Conclusion
This analysis provides a robust framework for predicting Alzheimer's Disease using health data. The findings offer valuable insights into the key factors associated with the disease and demonstrate the potential of machine learning models in supporting healthcare decision-making.
