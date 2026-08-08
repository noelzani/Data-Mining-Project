#  CEN 352 – Artificial Intelligence  
##  Term Project: Mental Health Prediction System

---

##  Group Members and Roles

**Nikol Dalipi**  
- Data preprocessing and dataset analysis  
- Rule-based system design  
- Evaluation and result interpretation  

**Noel Zani**  
- Random Forest model implementation  
- Model training and testing  
- Streamlit application development  

> The project was developed collaboratively; responsibilities may overlap.

---

##  Project Overview

This project focuses on the design and evaluation of an intelligent agent for employee attrition prediction.  
The goal is to classify employees into one of the following categories:

- 🔴 Stressed  
- 🟠 At Risk
- 🟢 Healthy

The project implements and evaluates two different AI techniques taught in the course and compares their performance on the same dataset.

---

##  AI Techniques Implemented

###  Rule-Based System (Logical Reasoning)

A rule-based system was developed using manually defined rules and thresholds derived from training data statistics.

Key characteristics:
- Deterministic and interpretable decisions  
- Explicit representation of domain knowledge  
- Symbolic reasoning approach  

Why this method was chosen:  
Rule-based systems are a core topic in the course and provide transparent decision logic.

---

###  Random Forest Classifier (Statistical Learning)

A Random Forest classifier was implemented and trained on the dataset after preprocessing and class balancing.

Key characteristics:
- Ensemble learning using multiple decision trees  
- Handles non-linear feature interactions  
- Robust to overfitting and noise  

Why this method was chosen:  
Random Forest is a powerful statistical learning algorithm covered in the course syllabus.

---

##  Model Usage and Comparison

The two AI techniques are implemented and evaluated independently.  
They are not combined into a hybrid system.

This allows:
- Fair comparison of symbolic and statistical approaches  
- Clear analysis of strengths and weaknesses  
- Compliance with the requirement of using multiple AI techniques  

---

##  Dataset Description

The dataset contains employee-related features such as workload indicators, satisfaction levels, and performance attributes.

Dataset characteristics:
- Realistic and socially relevant  
- Highly imbalanced  
- Suitable for both rule-based and machine learning approaches  

---

##  Evaluation Methodology

Because the dataset is imbalanced, accuracy alone is not sufficient.

Evaluation metrics used:
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

These metrics align with the Performance (P) component of the PEAS framework.

---

## Project Structure

AI-Project-Demo/  
│  
├── src/  
│   ├── preprocessing.py  
│   ├── rule_based.py  
│   ├── random_forest.py  
│   └── app.py  
│  
├── models/  
│   └── random_forest_model.pkl  
│   └── encoding_metadata.pkl  
│   └── label_encoder.pkl  
│   └── feature_names.pkl  
│   └── scaler.pkl  
│   └── test_results.pkl  
│  
├── data/  
│   └── mental_health_social_media_dataset.csv  
│  
├── requirements.txt  
└── README.md  

---

##  Virtual Environment (venv)

A virtual environment is used to isolate project dependencies and ensure reproducibility.

Why venv is used:
- Prevents dependency conflicts  
- Ensures consistent library versions  
- Makes the project reproducible  

---

##  How to Run the Project

1. Clone the repository  
   $ git clone <repository-link>  
   $ cd cen352-term-project-2025-26-nikol-noel 

2. Create and activate a virtual environment
   $ python -m venv venv  
   $ source venv/bin/activate        (macOS/Linux)  
   $ venv\Scripts\activate           (Windows)  

3. Install dependencies  
   $ pip install -r requirements.txt  

4. Run the models  
   $ python src/main.py  

5. Run the Streamlit application  
   $ streamlit run src/app.py  

---

##  Agent Design

Agent types:
- Symbolic agent (Rule-Based System)  
- Learning agent (Random Forest)  

Environment: Partially observable, static  
Sensors: Dataset features  
Actuators: Classification decisions  

---

##  Future Work

- Add additional classifiers for comparison  
- Improve rule calibration  
- Introduce explainability techniques  
- Explore hybrid integration as an extension  

---

##  Libraries Used

- scikit-learn  
- pandas  
- numpy  
- imbalanced-learn  
- streamlit  
- joblib  

All code was developed by the project members.

---

##  Compliance Summary

✔ Two AI techniques implemented  
✔ Python-based solution  
✔ Quantitative evaluation   
✔ Reproducible execution instructions provided  
