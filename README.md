<img src="images/icon.jpeg" width="100" alt="alt text">

# Alzheimer's Disease Prediction

This repository contains a dataset and a comprehensive analysis framework for predicting Alzheimer's Disease using various machine learning and deep learning models.

## Main Objective
The primary objective of this analysis is to explore the factors associated with Alzheimer's Disease, develop predictive models, and conduct statistical analyses to provide insights and predictive capabilities. The focus will be on a specific type of Deep Learning algorithm to improve prediction accuracy and provide valuable information to stakeholders, such as healthcare professionals and researchers.

## Data Summary
The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and diagnosis information. The data covers a wide range of features essential for understanding and predicting Alzheimer's Disease.   

## About the Dataset

This dataset contains extensive health information for 2,149 patients, each uniquely identified with IDs ranging from 4751 to 6900. The dataset includes the following features:

### Patient Information
- **Patient ID**: A unique identifier assigned to each patient (4751 to 6900) (omitted)..

### Demographic Details
- **Age**: The age of the patients ranges from 60 to 90 years.
- **Gender**: Gender of the patients, where 0 represents Male and 1 represents Female.
- **Ethnicity**: The ethnicity of the patients, coded as follows:
- **EducationLevel**: The education level of the patients, coded as follows:
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

## Data Exploration and Cleaning
Initial data exploration involved checking for missing values, outliers, and inconsistencies. Feature engineering steps included scaling numerical features, one-hot encoding categorical features, and undersampling to address class imbalance.

Class distribution before resampling:
| Diagnosis | Samples |
|-----------|---------|
| 0         | 1389    |     
| 1         | 760     |

Class distribution after resampling:
| Diagnosis | Samples |
|-----------|---------|
| 0         | 760    |     
| 1         | 760     |


## Features Selection:
The SelectKBest method from the sklearn.feature_selection module is a feature selection technique that selects the top k features based on a statistical measure of their relevance to the target variable. 

Selected As top 10 Features:
- CardiovascularDisease
- CholesterolHDL
- MMSE
- FunctionalAssessment
- MemoryComplaints
- BehavioralProblems
- ADL
- Ethnicity_1
- Ethnicity_2
- EducationLevel_0   

We can also view this as a correlation matrix and see the top 5 features:   

<img src="images/heatmap.png" width="400" alt="alt text">

----------------------------------------------------------------------------

## Model Training and Evaluation
We used a neural network model and conducted a grid search to find the best hyperparameters, including the number of layers, nodes, and learning rate. The grid search was performed using various configurations. Stratified cross-validation with 8 folds and downsampling for preprocessing were utilized to ensure balanced class distribution.   
  
We also experimented with 5 simplier and less complex models such as Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forrest, and Gradient Boosing with cross-validation with 8 folds.   

## Key Findings and Insights

### Neural Network (NN) (All Features)
| Class        | Precision | Recall | F1-Score | Support  |
|--------------|-----------|--------|----------|----------|
| No Alzheimer | 0.842     | 0.818  | 0.830    | 760      |
| Alzheimer    | 0.823     | 0.847  | 0.835    | 760      |

### Best Configuration
- **Layers**: [128, 128, 128]
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **F1 Score**: 0.834

----------------------------------------------------------------------------

<img src="images/learning_curve.png" width="500" alt="alt text">

<img src="images/confusion_matrix.png" width="350" alt="alt text">

<img src="images/roc.png" width="350" alt="alt text">

----------------------------------------------------------------------------

### Comparison of F1 Scores
| Complexity | Model                   | Parameters (All Features) | Parameters (Top 10 Features) | F1 Score (All Features) | F1 Score (Top 10 Features) |
|------------|-------------------------|---------------------------|------------------------------|-------------------------|----------------------------|
| 1          | Logistic Regression     | C = 10                    | C = 10                       | 0.812                   | 0.821                      |
| 2          | K-Nearest Neighbors     | k = 50                    | k = 3                        | 0.757                   | 0.823                      |
| 3          | Support Vector Machine  | C = 1 (Linear)            | C = 5 (Linear)               | 0.815                   | 0.888                      |
| 4          | Random Forest           | Estimators = 500          | Estimators = 200             | 0.922                   | 0.928                      |
| 5          | Gradient Boosting       | Estimators = 100          | Estimators = 100             | 0.919                   | 0.922                      |
| 6          | Neural Network          | Layers = [128, 128, 128]  |                              | 0.834                   |                            |

Overall, it seems that the classification task is not too complex, as smaller and less complex model can yield better results.   

### Top 10 Features Importance (Random Forest)
These are the most constributing variables or features that predict positive diagnosis of Alzheimer:
|       | Feature                    | Importance |
|-------|----------------------------|------------|
| 1     | Functional Assessment      | 0.1812     |
| 2     | ADL                        | 0.1603     |
| 3     | MMSE                       | 0.1304     |
| 4     | Memory Complaints          | 0.0827     |
| 5     | Behavioral Problems        | 0.0490     |
| 6     | BMI                        | 0.0310     |
| 7     | Sleep Quality              | 0.0293     |
| 8     | Diet Quality               | 0.0293     |
| 9     | Cholesterol HDL            | 0.0287     |
| 10    | Cholesterol Triglycerides  | 0.0285     |


### Recommended Model
The best-performing model was the Random Forest, especially when using the top 10 features, achieving an F1 score of 0.9285. This model demonstrated superior performance in predicting Alzheimer's Disease.   

## Conclusion
This analysis provides a robust framework for predicting Alzheimer's Disease using health data. The findings offer valuable insights into the key factors associated with the disease and demonstrate the potential of machine learning models in supporting healthcare decision-making.   
   
Future analysis could involve:
- **Incorporating Additional Data**: Including more detailed genetic information or longitudinal data could enhance the model's predictive power.
- **Model Refinement**: Experimenting with other machine learning algorithms or ensemble methods, as well as setting different Neural Network hyperparameters that could further improve accuracy.
- **Clinical Validation**: Collaborating with healthcare professionals to validate the model's predictions in a clinical setting.


## Setup

Scripts:   
- `data_preprocessing.py`: Loads and preprocesses the data, including scaling and resampling.   
- `feature_selection.py`: Selects the top 10 features using `SelectKBest`.   
- `simple_model_evaluation.py`: Evaluates simple models using all features and the top 10 features, saving the best F1 scores.   
- `feature_importance.py`: Trains a Random Forest model and prints the feature importance.   
- `layer_configurations.py`: Generates layer configurations and learning rates for the neural network.   
- `neural_network.py`: Trains and evaluates a flexible neural network model using the best configurations.   
- `alzheimer_train_main.py`: Main script to run all the above scripts in the correct order.   

1. Clone the repository:
    ```bash
    git clone https://github.com/reab5555/alzheimers-disease-prediction.git
    cd alzheimers-disease-prediction
    ```

2. Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn tqdm matplotlib seaborn tabulate torch
    ```

### How to Run

1. Ensure the data file `alzheimers_dataset.csv` is in the correct path as specified in the `data_preprocessing.py` script.

2. Run the main script:
    ```bash
    python alzheimer_train_main.py
    ```

### Notes

- Ensure the data file `alzheimers_dataset.csv` is in the correct path as specified in the `data_preprocessing.py` script.
- Modify the file paths in the scripts if your directory structure is different.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
