
# Classification and Ensemble Learning

Implementation and evaluation of classification models for student performance prediction.

This project was developed as part of the Machine Learning and Deep Learning course in the Artificial Intelligence Engineering program at Universidad de San Andrés.

## Overview

The objective of this project is to build predictive models to identify students at risk of low academic performance.

The dataset contains student-level information collected across multiple schools and semesters. A central challenge of the project is evaluating how model performance changes depending on the validation strategy, especially when schools differ from each other.

The project includes both binary and multiclass classification, with a focus on robust evaluation, grouped data, class imbalance and ensemble learning.

## Main Topics

- Exploratory Data Analysis
- Data preprocessing
- Binary classification
- Multiclass classification
- Logistic Regression from scratch
- L2 regularization
- Random split validation
- Group split by school
- Temporal split validation
- Cross-validation
- GroupKFold
- Linear Discriminant Analysis
- Random Forest
- Feature importance
- Class imbalance handling
- Undersampling
- Oversampling
- SMOTE
- Cost re-weighting
- Final model selection

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Precision-Recall AUC

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── classification_ensemble_learning.ipynb
└── src/
    ├── models.py
    ├── preprocessing.py
    ├── metrics.py
    └── validation.py

Data Availability

The datasets used in this project are not included in the repository due to course distribution restrictions.

The notebook is provided with executed outputs so the full workflow, experiments, visualizations and results can be reviewed without rerunning the entire project.

To run the project locally, place the required dataset files inside the data/ directory following the paths expected by the notebook.

Notes

This project emphasizes the importance of choosing realistic validation strategies.

In particular, group-based validation by school provides a more realistic estimate of generalization when deploying a model to unseen institutions. The project also studies how class imbalance affects performance and how different rebalance strategies change the behavior of the classifier.

Key Learnings
Why random splits can overestimate model performance
How grouped data affects validation and generalization
How temporal validation differs from random validation
How class imbalance changes model behavior
How ensemble methods can improve predictive performance
How to compare classification models using multiple metrics
