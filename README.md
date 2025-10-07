# 🤖 Machine Learning Project Portfolio

This repository contains a curated portfolio of eight machine learning projects demonstrating proficiency across core ML domains, including Classification (Binary, Multi-Class), Regression, and Deep Learning for Computer Vision.

Each project notebook (.ipynb files) details the complete pipeline: Data Loading, Exploratory Data Analysis (EDA), Feature Engineering, Multi-Model Training, Performance Evaluation, and Visualization.

📂 Projects Overview
The portfolio is structured around models built using structured tabular data and a dedicated computer vision task.

Tabular Data Projects (Classification & Regression)
Project

Objective

ML Type

Key Data Features

Best Model

Key Metric Result

Forest Cover Type Prediction

Predict 1 of 7 forest types.

Multi-Class Classification

Topography, Hydrology, Soil/Wilderness Indicators.

Random Forest / LightGBM

High Accuracy (Elevation is key feature).

Liver Cirrhosis Stage Detection

Predict the disease stage (1, 2, or 3).

Multi-Class Classification

Clinical Biomarkers (Bilirubin, Albumin, Copper).

Random Forest

Strong F1-Score (Reliable clinical staging).

Mobile Phone Price Prediction

Predict price range (Low/Med/High/Very High).

Multi-Class Classification

RAM, Battery Power, Pixel Resolution.

LightGBM Classifier

High Macro F1-Score.

Vehicle Price Prediction

Predict vehicle price (continuous value).

Regression

Make, Model, Year, Mileage, Engine Specs.

LightGBM Regressor

High R 
2
  (≈0.90+) and Low MAE.

Heart Disease Prediction

Predict heart disease presence (Binary).

Binary Classification

Age, Sex, Cholesterol, Chest Pain Type.

Random Forest Classifier

Excellent ROC AUC (≈0.97).

Thyroid Cancer Detection

Predict cancer recurrence (Binary).

Binary Classification

Pathology, Stage, Lymph Node Involvement.

Support Vector Classifier (SVC)

High Cross-Validation Accuracy (≈89%).

Lung Cancer Survival

Predict patient survival (Binary).

Binary Classification

Cancer Stage, Smoking Status, Comorbidities.

LightGBM / Logistic Regression

AUC ≈0.50 (Indicates data limitation for static prediction).

Computer Vision Project
Project

Objective

ML Type

Approach

Key Result

Animal Species Classification

Classify 15 different animal species from images.

Deep Learning

Transfer Learning (InceptionV3) with custom classification head.

Validation Accuracy ≈95%.

🛠️ Key Methodologies Demonstrated
Data Preparation and Feature Engineering
Imputation & Handling Missingness: Used median/mode imputation to handle missing values in clinical and commercial datasets.

Feature Transformation: Techniques included mapping circular features (e.g., trigonometric transformation of Aspect) and calculating age features (e.g., car_age).

Encoding & Scaling: Employed OneHotEncoder and OrdinalEncoder for categorical variables, and StandardScaler to normalize numerical data.

Imbalance Handling: Utilized class_weight='balanced' in models for imbalanced tasks (like Lung Cancer Survival) to prioritize meaningful metrics over skewed accuracy.

Multi-Model Comparison and Selection
Every project involved training and evaluating multiple distinct model types to ensure the optimal algorithm was selected based on performance and robustness:

Model Diversity: Comparison included Linear Models (Logistic Regression, Ridge), Instance-Based Models (KNN), and advanced Tree-Based Ensembles (Random Forest, LightGBM).

Model Tuning: Focus on selecting models that generalize well, often relying on Cross-Validation scores.

Evaluation and Visualization
Performance was analyzed using extensive visualizations for true insight:

Key Performance Indicators (KPIs): Used task-appropriate metrics: ROC AUC, F1-Score (Macro), R 
2
 , and Mean Absolute Error (MAE).

Visual Diagnostics: Comprehensive use of matplotlib and seaborn to generate:

Feature Importance Plots to understand model drivers.

ROC Curves for clear trade-off analysis in binary classification.

Confusion Matrices for detailed error analysis (e.g., False Positives/Negatives).

Distribution and Correlation Plots for initial EDA.

Deep Learning and Transfer Learning
The Animal Classification project demonstrated Transfer Learning, leveraging the pre-trained weights of InceptionV3 to achieve high accuracy quickly, minimizing the need for massive, domain-specific training data.

🚀 How to Run the Notebooks
Clone the Repository:

git clone [repository-link]

Open in Colab/Jupyter: Upload the desired *.ipynb file (e.g., heart_disease.ipynb) to your Google Colab or local Jupyter environment.

Data: Each notebook is set up to read its corresponding dataset. Ensure your data file is in the same directory or the path is adjusted.

Run Cells: Execute the notebook cells sequentially to reproduce the data analysis, modeling, and evaluation pipeline.
